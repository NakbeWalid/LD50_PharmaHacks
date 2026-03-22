from __future__ import annotations

import json
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

ARTIFACTS_DIR = Path(__file__).resolve().parent.parent / "artifacts"
REPORT_PATH = ARTIFACTS_DIR / "report.json"


def _load_report() -> dict:
    if not REPORT_PATH.is_file():
        raise HTTPException(
            status_code=503,
            detail=(
                f"Missing file: {REPORT_PATH}. "
                "Run: py -3.13 scripts/train_and_save.py from ld50-api/"
            ),
        )
    return json.loads(REPORT_PATH.read_text(encoding="utf-8"))


app = FastAPI(title="LD50 report API (local)", version="3.0.0")

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


@app.get("/health")
def health():
    ok = REPORT_PATH.is_file()
    return {"status": "ok" if ok else "no_report", "report": str(REPORT_PATH)}


@app.get("/report")
def report():
    return _load_report()
