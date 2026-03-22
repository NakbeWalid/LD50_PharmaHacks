# LD50 local web report (Angular + Python)

The UI is **English** and shows the same story as `ld50_starter.ipynb`: dataset intro, roadmap, splits, model description, **metrics**, and **figures** (target distribution, validation/test scatter plots, SHAP global and local waterfall for validation row 42).

There is **no SMILES prediction** for arbitrary new molecules — only **reported performance and plots** from the training script.

## 1. Train once (generates `report.json`)

Same pipeline as the notebook: TDC-style 70/10/20 split, Morgan (r=2, 1024) + scaled physicochemical descriptors + MACCS (167), XGBoost.

```powershell
cd c:\Users\HP\LD50_PharmaHacks\project\ld50-api
py -3.13 -m pip install -r requirements.txt
py -3.13 scripts\train_and_save.py
```

Creates:

- `ld50-api/artifacts/model_bundle.joblib` (gitignored by default)
- `ld50-api/artifacts/report.json` — metrics + data for all charts (including SHAP)

## 2. API (FastAPI)

Pick a free port (**8765** is the default in `ld50-ui/proxy.conf.json`). Port **8000** may hit `WinError 10013` on some Windows setups.

```powershell
cd c:\Users\HP\LD50_PharmaHacks\project\ld50-api
py -3.13 -m uvicorn app.main:app --reload --host 127.0.0.1 --port 8765
```

- Health: http://127.0.0.1:8765/health  
- Report JSON: http://127.0.0.1:8765/report  

If you change the API port, edit **`ld50-ui/proxy.conf.json`** (`target`) and **restart** `npm start`.

## 3. Angular UI

```powershell
cd c:\Users\HP\LD50_PharmaHacks\project\ld50-ui
npm install
npm start
```

Open http://localhost:4200 — the app requests **`GET /api/report`** (proxied to uvicorn).

## After retraining in the notebook

Re-run `train_and_save.py` so `report.json` matches your latest model and plots.
