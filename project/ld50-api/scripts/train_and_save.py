"""
Même pipeline que ld50_starter.ipynb : split TDC, featurisation, scaler, XGBoost.
Sauvegarde le modèle + métriques train / valid / test (comme les prints du notebook).

Run : py -3.13 project/ld50-api/scripts/train_and_save.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler

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
METRICS_PATH = ARTIFACTS_DIR / "metrics.json"


def tdc_random_split(df: pd.DataFrame, fold_seed: int = 42, frac: tuple[float, float, float] = (0.7, 0.1, 0.2)):
    """Identique à tdc.utils.split.create_fold (utilisé par get_split du notebook TDC)."""
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
    rows_m = [smiles_to_morgan(s) for s in smiles_series]
    rows_a = [get_advanced_descriptors(s) for s in smiles_series]
    morgan_df = pd.DataFrame(rows_m, columns=list(range(MORGAN_NBITS)))
    adv_df = pd.DataFrame(rows_a, columns=ADV_COLS)
    return pd.concat([morgan_df, adv_df], axis=1)


def eval_split(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """Comme le notebook : R², MAE, RMSE = sqrt(MSE)."""
    return {
        "n": int(len(y_true)),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "r2": float(r2_score(y_true, y_pred)),
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

    scaler = StandardScaler()
    X_train = X_train.copy()
    X_valid = X_valid.copy()
    X_test = X_test.copy()
    X_train[COLS_TO_SCALE] = scaler.fit_transform(X_train[COLS_TO_SCALE])
    X_valid[COLS_TO_SCALE] = scaler.transform(X_valid[COLS_TO_SCALE])
    X_test[COLS_TO_SCALE] = scaler.transform(X_test[COLS_TO_SCALE])

    # Hyperparamètres comme ld50_starter.ipynb (cellule XGBoost).
    # XGBoost 3.x : pas d'early_stopping_rounds dans fit() ici ; n_estimators=1500 comme plafond notebook.
    model = xgb.XGBRegressor(
        n_estimators=1500,
        learning_rate=0.03,
        max_depth=7,
        subsample=0.8,
        colsample_bytree=0.3,
        reg_alpha=1.0,
        reg_lambda=2.0,
        random_state=42,
        n_jobs=-1,
        tree_method="hist",
    )
    print("Training (eval_set = train + valid, comme le notebook)...")
    model.fit(
        X_train,
        y_train,
        eval_set=[(X_train, y_train), (X_valid, y_valid)],
        verbose=100,
    )

    y_pred_train = model.predict(X_train)
    y_pred_valid = model.predict(X_valid)
    y_pred_test = model.predict(X_test)

    metrics_payload = {
        "description": "Métriques sklearn comme le notebook (MAE, RMSE=sqrt(MSE), R²).",
        "split_rule": "random 70/10/20, seed=42 puis val random_state=1 (create_fold TDC)",
        "train": eval_split(y_train, y_pred_train),
        "valid": eval_split(y_valid, y_pred_valid),
        "test": eval_split(y_test, y_pred_test),
    }

    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    bundle = {
        "model": model,
        "scaler": scaler,
        "cols_to_scale": COLS_TO_SCALE,
    }
    joblib.dump(bundle, BUNDLE_PATH)
    METRICS_PATH.write_text(json.dumps(metrics_payload, indent=2), encoding="utf-8")
    print(f"Saved: {BUNDLE_PATH}")
    print(f"Saved: {METRICS_PATH}")
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
