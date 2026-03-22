from __future__ import annotations

import json
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

ARTIFACTS_DIR = Path(__file__).resolve().parent.parent / "artifacts"
METRICS_PATH = ARTIFACTS_DIR / "metrics.json"

app = FastAPI(title="LD50 — métriques (local)", version="2.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:4200",
        "http://127.0.0.1:4200",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
def startup() -> None:
    pass


@app.get("/health")
def health():
    ok = METRICS_PATH.is_file()
    return {"status": "ok" if ok else "no_metrics", "metrics": str(METRICS_PATH)}


@app.get("/metrics")
def metrics():
    """Métriques train / valid / test générées par scripts/train_and_save.py (logique notebook)."""
    if not METRICS_PATH.is_file():
        raise HTTPException(
            status_code=503,
            detail=(
                f"Fichier manquant: {METRICS_PATH}. "
                "Exécute: py -3.13 scripts/train_and_save.py depuis ld50-api/"
            ),
        )
    return json.loads(METRICS_PATH.read_text(encoding="utf-8"))
