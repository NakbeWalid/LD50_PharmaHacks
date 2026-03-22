# Tester le modèle LD50 en local (Angular + API Python)

L’interface affiche uniquement les **performances** (train / validation / test), comme les métriques du notebook — **pas de prédiction** sur de nouvelles molécules.

## 1. Entraîner et exporter métriques + modèle (une fois)

Même pipeline que le notebook : split TDC 70/10/20, Morgan 1024 (r=2) + 5 descripteurs + MACCS (167), `StandardScaler` sur les 5 descripteurs, XGBoost (hyperparamètres alignés sur `ld50_starter.ipynb`).

```powershell
cd c:\Users\HP\LD50_PharmaHacks\project\ld50-api
py -3.13 -m pip install -r requirements.txt
py -3.13 scripts\train_and_save.py
```

Cela crée :

- `ld50-api/artifacts/model_bundle.joblib` (non versionné par défaut)
- `ld50-api/artifacts/metrics.json` — MAE, RMSE (√MSE), R² pour train, valid, test

## 2. Lancer l’API (FastAPI)

Choisis un port libre (**8080**, **8765**, etc.) : sous Windows, le **8000** peut déclencher `WinError 10013`. Le fichier **`ld50-ui/proxy.conf.json`** pointe par défaut vers **8765** ; si tu lances uvicorn sur un autre port, mets le même dans `target`.

```powershell
cd c:\Users\HP\LD50_PharmaHacks\project\ld50-api
py -3.13 -m uvicorn app.main:app --reload --host 127.0.0.1 --port 8765
```

- Ex. santé : http://127.0.0.1:8765/health  
- Ex. métriques : http://127.0.0.1:8765/metrics  
- Ex. docs : http://127.0.0.1:8765/docs  

En dev, l’UI utilise un **proxy Angular** (`ld50-ui/proxy.conf.json`) : les requêtes vont vers `/api/...` et sont renvoyées vers uvicorn. **Modifie `target` dans ce fichier** si ton API n’est pas sur `http://127.0.0.1:8765`. Puis **redémarre** `npm start` après toute modification du proxy.

## 3. Lancer l’interface Angular

Autre terminal :

```powershell
cd c:\Users\HP\LD50_PharmaHacks\project\ld50-ui
npm start
```

Ouvre http://localhost:4200 — l’app appelle `GET /api/metrics`, relayé par le proxy vers ton uvicorn (évite le décalage de port et le « 0 Unknown Error » si le front pointait vers le mauvais port).

## Après un nouvel entraînement dans le notebook

Ré-exécute `train_and_save.py` pour régénérer `metrics.json` et le bundle, ou adapte le notebook pour exporter les mêmes artefacts (mêmes colonnes et scaler sur `MolWt`, `LogP`, `HBD`, `HBA`, `TPSA`).
