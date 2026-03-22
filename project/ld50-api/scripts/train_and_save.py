"""
Aligned with ld50_starter.ipynb (CELL 6 merge/scale, CELL 7 XGBoost, CELL 8–9 SHAP, CELL 10 test).
Saves model bundle + report.json for the web UI.

Run: py -3.13 project/ld50-api/scripts/train_and_save.py
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR

API_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(API_ROOT))

from app.featurize import (  # noqa: E402
    ADV_COLS,
    COLS_TO_SCALE,
    MORGAN_NBITS,
    get_advanced_descriptors,
    smiles_to_morgan,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = PROJECT_ROOT / "notebooks" / "data" / "ld50_zhu.tab"
ARTIFACTS_DIR = API_ROOT / "artifacts"
BUNDLE_PATH = ARTIFACTS_DIR / "model_bundle.joblib"
REPORT_PATH = ARTIFACTS_DIR / "report.json"

MOLECULE_INDEX_WATERFALL = 42  # CELL 9: Case Study — validation row index


def tdc_random_split(df: pd.DataFrame, fold_seed: int = 42, frac: tuple[float, float, float] = (0.7, 0.1, 0.2)):
    train_frac, val_frac, test_frac = frac
    test = df.sample(frac=test_frac, replace=False, random_state=fold_seed)
    train_val = df.loc[~df.index.isin(test.index)]
    val = train_val.sample(frac=val_frac / (1.0 - test_frac), replace=False, random_state=1)
    train = train_val.loc[~train_val.index.isin(val.index)]
    return (
        train.reset_index(drop=True),
        val.reset_index(drop=True),
        test.reset_index(drop=True),
    )


def build_X(smiles_series: pd.Series) -> pd.DataFrame:
    # CELL 3: Morgan matrices; CELL 5: advanced descriptors — merged like CELL 6.
    rows_m = [smiles_to_morgan(s) for s in smiles_series]
    rows_a = [get_advanced_descriptors(s) for s in smiles_series]
    morgan_df = pd.DataFrame(rows_m, columns=list(range(MORGAN_NBITS)))
    adv_df = pd.DataFrame(rows_a, columns=ADV_COLS)
    return pd.concat([morgan_df, adv_df], axis=1)


def column_labels(columns: list) -> list[str]:
    out: list[str] = []
    for c in columns:
        if isinstance(c, int) or (isinstance(c, str) and str(c).isdigit()):
            out.append(f"Morgan bit {int(c)}")
        else:
            out.append(str(c))
    return out


def eval_split(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    return {
        "n": int(len(y_true)),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "r2": float(r2_score(y_true, y_pred)),
    }


def compute_shap_plots(
    model: xgb.XGBRegressor,
    X_valid: pd.DataFrame,
    valid_df: pd.DataFrame,
    y_valid: np.ndarray,
    y_pred_valid: np.ndarray,
    feature_labels: list[str],
) -> dict:
    import shap

    Xv = X_valid.values.astype(np.float64)
    explainer = shap.TreeExplainer(model)
    sv = explainer.shap_values(Xv)
    base = float(explainer.expected_value if np.ndim(explainer.expected_value) == 0 else explainer.expected_value[0])

    mean_abs = np.abs(sv).mean(axis=0)
    top_g = np.argsort(mean_abs)[-10:][::-1]
    shap_global = [
        {"feature": feature_labels[i], "mean_abs_shap": float(mean_abs[i])}
        for i in top_g
    ]

    mi = MOLECULE_INDEX_WATERFALL
    molecule_id = str(valid_df["Drug_ID"].iloc[mi])
    sv_row = sv[mi]
    fv_row = Xv[mi]
    top_w = np.argsort(np.abs(sv_row))[-10:][::-1]
    waterfall_features = [
        {
            "feature": feature_labels[i],
            "shap": float(sv_row[i]),
            "value": float(fv_row[i]),
        }
        for i in top_w
    ]
    pred_row = float(y_pred_valid[mi])
    true_row = float(y_valid[mi])

    return {
        "global_top10": shap_global,
        "waterfall": {
            "molecule_index": mi,
            "molecule_id": molecule_id,
            "expected_value": base,
            "predicted_log_ld50": pred_row,
            "actual_log_ld50": true_row,
            "features": waterfall_features,
        },
    }


def run_benchmark_cell_7_5(
    X_train: pd.DataFrame,
    X_valid: pd.DataFrame,
    y_train: np.ndarray,
    y_valid: np.ndarray,
    y_pred_xgb_valid: np.ndarray,
    xgb_fit_seconds: float,
) -> dict:
    """
    CELL 7.5: compare Random Forest, SVR, and XGBoost on the same validation set.
    sklearn needs string column names (as in the notebook).
    """
    Xt = X_train.copy()
    Xv = X_valid.copy()
    Xt.columns = Xt.columns.astype(str)
    Xv.columns = Xv.columns.astype(str)

    rows: list[dict] = []

    # Random Forest — full training set (same as notebook intent).
    rf = RandomForestRegressor(n_estimators=200, max_depth=10, random_state=42, n_jobs=-1)
    t0 = time.perf_counter()
    rf.fit(Xt, y_train)
    pred_rf = rf.predict(Xv)
    t_rf = time.perf_counter() - t0
    rows.append(
        {
            "model": "Random Forest",
            "r2_validation": round(float(r2_score(y_valid, pred_rf)), 4),
            "mae_validation": round(float(mean_absolute_error(y_valid, pred_rf)), 4),
            "train_time_s": round(float(t_rf), 2),
        }
    )

    # SVR RBF — full-data fit is often intractable (O(n²) kernel); notebook uses full X_train.
    # We train on a fixed random subset so the pipeline finishes in reasonable time; validation predictions use full X_valid.
    svr_n_train = min(3000, len(Xt))
    rng = np.random.RandomState(42)
    svr_idx = rng.choice(len(Xt), size=svr_n_train, replace=False)
    svr = SVR(kernel="rbf", C=1.0, epsilon=0.1, cache_size=2000)
    t0 = time.perf_counter()
    svr.fit(Xt.iloc[svr_idx], y_train[svr_idx])
    pred_svr = svr.predict(Xv)
    t_svr = time.perf_counter() - t0
    rows.append(
        {
            "model": "SVR (RBF)",
            "r2_validation": round(float(r2_score(y_valid, pred_svr)), 4),
            "mae_validation": round(float(mean_absolute_error(y_valid, pred_svr)), 4),
            "train_time_s": round(float(t_svr), 2),
            "train_rows_used": int(svr_n_train),
        }
    )

    # XGBoost — already trained; report CELL 7 fit time and validation metrics (no refit).
    rows.append(
        {
            "model": "XGBoost",
            "r2_validation": round(float(r2_score(y_valid, y_pred_xgb_valid)), 4),
            "mae_validation": round(float(mean_absolute_error(y_valid, y_pred_xgb_valid)), 4),
            "train_time_s": round(float(xgb_fit_seconds), 2),
        }
    )

    rows.sort(key=lambda r: r["r2_validation"], reverse=True)
    rank = {m: i + 1 for i, m in enumerate(r["model"] for r in rows)}

    return {
        "description": "Validation-set comparison (same features as CELL 6–7). R² / MAE on log(LD50).",
        "r2_threshold": 0.6,
        "svr_note": (
            f"SVR (RBF) was trained on {svr_n_train} random train molecules (seed=42); "
            "full RBF SVR on ~5k×1200 features is often impractically slow."
        ),
        "leaderboard": rows,
        "rank_by_model": rank,
    }


def main() -> None:
    if not DATA_PATH.is_file():
        raise FileNotFoundError(f"Dataset not found: {DATA_PATH}")

    raw = pd.read_csv(DATA_PATH, sep="\t")
    df = raw.rename(columns={"ID": "Drug_ID", "X": "Drug"})
    train_df, valid_df, test_df = tdc_random_split(df)

    print("Featurizing train...")
    X_train = build_X(train_df["Drug"])
    print("Featurizing valid...")
    X_valid = build_X(valid_df["Drug"])
    print("Featurizing test...")
    X_test = build_X(test_df["Drug"])

    y_train = train_df["Y"].to_numpy(dtype=np.float64)
    y_valid = valid_df["Y"].to_numpy(dtype=np.float64)
    y_test = test_df["Y"].to_numpy(dtype=np.float64)

    # CELL 6: fit mean/std ONLY on train; transform valid/test without refitting.
    scaler = StandardScaler()
    X_train = X_train.copy()
    X_valid = X_valid.copy()
    X_test = X_test.copy()
    X_train[COLS_TO_SCALE] = scaler.fit_transform(X_train[COLS_TO_SCALE])
    X_valid[COLS_TO_SCALE] = scaler.transform(X_valid[COLS_TO_SCALE])
    X_test[COLS_TO_SCALE] = scaler.transform(X_test[COLS_TO_SCALE])

    feat_labels = column_labels(list(X_train.columns))

    # CELL 7: XGBoost — aggressive hyperparameters to fight chemical noise (notebook).
    model = xgb.XGBRegressor(
        n_estimators=1500,
        learning_rate=0.03,
        max_depth=7,
        subsample=0.8,
        colsample_bytree=0.4,
        reg_alpha=1.0,
        reg_lambda=2.0,
        random_state=42,
        n_jobs=-1,
        tree_method="hist",
    )
    print("Training (CELL 7: eval_set train+valid, early_stopping if supported)...")
    fit_kw: dict = {
        "eval_set": [(X_train, y_train), (X_valid, y_valid)],
        "verbose": 100,
    }
    t_xgb0 = time.perf_counter()
    try:
        fit_kw["early_stopping_rounds"] = 50
        model.fit(X_train, y_train, **fit_kw)
    except TypeError:
        fit_kw.pop("early_stopping_rounds", None)
        model.fit(X_train, y_train, **fit_kw)
    xgb_fit_seconds = time.perf_counter() - t_xgb0

    y_pred_train = model.predict(X_train)
    y_pred_valid = model.predict(X_valid)
    y_pred_test = model.predict(X_test)

    metrics_payload = {
        "description": "r2_score, mean_absolute_error, RMSE = sqrt(MSE) — same as notebook prints.",
        "split_rule": "Random 70/10/20, seed=42 then val random_state=1 (TDC create_fold).",
        "train": eval_split(y_train, y_pred_train),
        "valid": eval_split(y_valid, y_pred_valid),
        "test": eval_split(y_test, y_pred_test),
    }

    # Plots match the notebook: validation scatter (CELL 7), SHAP (CELL 8–9). No extra histogram/test scatter.
    plots: dict = {
        "scatter_validation": {
            "y_true": y_valid.tolist(),
            "y_pred": y_pred_valid.tolist(),
        },
    }

    print("Benchmark (CELL 7.5: Random Forest vs SVR vs XGBoost)...")
    benchmark = run_benchmark_cell_7_5(
        X_train,
        X_valid,
        y_train,
        y_valid,
        y_pred_valid,
        xgb_fit_seconds,
    )
    print("FINAL MODEL RANKING (validation R2, highest first):")
    for row in benchmark["leaderboard"]:
        print(
            f"  {row['model']}: R2={row['r2_validation']}, MAE={row['mae_validation']}, time={row['train_time_s']}s"
        )

    sample_rows = (
        train_df[["Drug_ID", "Drug", "Y"]].head(5).to_dict(orient="records")
    )

    print("Computing SHAP (CELL 8-9: TreeExplainer + validation set)...")
    try:
        plots["shap"] = compute_shap_plots(
            model, X_valid, valid_df, y_valid, y_pred_valid, feat_labels
        )
    except Exception as e:
        print(f"SHAP skipped: {e}")
        plots["shap"] = None

    report = {
        "metrics": metrics_payload,
        "plots": plots,
        "dataset": {
            "split_sizes": {
                "train": int(len(train_df)),
                "valid": int(len(valid_df)),
                "test": int(len(test_df)),
            },
            "sample_rows": sample_rows,
        },
        "model": {
            "name": "XGBoost regressor",
            "params": {
                "n_estimators": 1500,
                "learning_rate": 0.03,
                "max_depth": 7,
                "subsample": 0.8,
                "colsample_bytree": 0.4,
                "reg_alpha": 1.0,
                "reg_lambda": 2.0,
                "early_stopping_rounds": 50,
                "tree_method": "hist",
            },
        },
        "notebook": {
            "reference": "ld50_starter.ipynb — CELL 3 Morgan, CELL 5 descriptors, CELL 6 scale, CELL 7–10 train/SHAP/test, CELL 7.5 benchmark",
        },
        "benchmark": benchmark,
        "features": {
            "morgan": {"radius": 2, "n_bits": MORGAN_NBITS},
            "physicochemical": COLS_TO_SCALE,
            "maccs_bits": 167,
        },
    }

    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    bundle = {
        "model": model,
        "scaler": scaler,
        "cols_to_scale": COLS_TO_SCALE,
    }
    joblib.dump(bundle, BUNDLE_PATH)
    REPORT_PATH.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Saved: {BUNDLE_PATH}")
    print(f"Saved: {REPORT_PATH}")
    print("--- Validation ---")
    print(f"R2   : {metrics_payload['valid']['r2']:.4f}")
    print(f"MAE  : {metrics_payload['valid']['mae']:.4f}")
    print(f"RMSE : {metrics_payload['valid']['rmse']:.4f}")
    print("--- Test ---")
    print(f"R2   : {metrics_payload['test']['r2']:.4f}")
    print(f"MAE  : {metrics_payload['test']['mae']:.4f}")
    print(f"RMSE : {metrics_payload['test']['rmse']:.4f}")


if __name__ == "__main__":
    main()
