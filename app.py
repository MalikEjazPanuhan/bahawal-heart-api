"""
BAHAWAL CARDIOLOGIST HOSPITAL — Single-service deploy
=====================================================
Hosts BOTH the React frontend (as static files) AND the FastAPI model
on a single Render Web Service. One URL, no SPA 404s, no cross-origin issues.

Endpoints:
  GET  /                 → hospital React app
  GET  /assets/*         → bundled JS/CSS/images
  GET  /health           → liveness check
  GET  /docs             → Swagger UI
  POST /predict          → 8 model features → risk probability
  POST /predict-full     → full hospital form JSON
"""

import json
import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime, date, timezone
import uuid

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from pydantic import BaseModel, Field
from starlette.staticfiles import StaticFiles

# ---------------------------------------------------------------------------
# Load model + metadata
# ---------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent
ARTIFACT_DIR = BASE_DIR / "model_artifacts"

model = joblib.load(ARTIFACT_DIR / "heart_rf_model.joblib")
feature_columns: list[str] = joblib.load(ARTIFACT_DIR / "feature_columns.joblib")
with open(ARTIFACT_DIR / "feature_defaults.json") as f:
    feature_defaults: dict[str, int] = json.load(f)

# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------
app = FastAPI(
    title="BAHAWAL CARDIOLOGIST HOSPITAL — Heart Risk API",
    description="Random Forest classifier predicting heart disease risk from 8 clinical features.",
    version="2.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------
class PredictRequest(BaseModel):
    age:      int   = Field(..., ge=1,   le=120)
    trestbps: int   = Field(..., ge=50,  le=250)
    chol:     int   = Field(..., ge=100, le=600)
    cp:       int   = Field(..., ge=0,   le=3)
    ca:       int   = Field(..., ge=0,   le=4)
    thalach:  int   = Field(..., ge=50,  le=250)
    thal:     int   = Field(..., ge=0,   le=3)
    oldpeak:  float = Field(..., ge=0,   le=7)


class PredictResponse(BaseModel):
    risk_probability: float
    risk_level:       str
    model_version:    str
    disclaimer:       str


PredictResponse.model_config = {"protected_namespaces": ()}  # type: ignore[attr-defined]


class PatientInfo(BaseModel):
    full_name:     str
    email:         str
    phone:         str
    date_of_birth: str


class MedicalData(BaseModel):
    trestbps: float
    chol:     float
    cp:       float
    ca:       float
    thalach:  float
    thal:     float
    oldpeak:  float


class HospitalFormRequest(BaseModel):
    patient_info: PatientInfo
    medical_data: MedicalData


class HospitalFormResponse(BaseModel):
    success:           bool
    risk_probability:  float
    risk_score:        int
    risk_level:        str
    risk_category:     str
    has_heart_disease: bool
    recommendation:    str
    recommendations:   str
    alert_level:       str
    timestamp:         str
    request_id:        str
    disclaimer:        str
    model_version:     str


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _build_feature_row(payload: dict) -> pd.DataFrame:
    row = {**feature_defaults, **payload}
    row["trestbps_capped"] = min(int(row["trestbps"]), 200)
    return pd.DataFrame([row], columns=feature_columns)


def _classify(p: float) -> str:
    if p < 0.30: return "Low"
    if p < 0.60: return "Moderate"
    return "High"


def _age_from_dob(dob: str) -> int:
    try:
        d = datetime.fromisoformat(dob.replace("Z", "")).date()
    except ValueError:
        d = date.fromisoformat(dob)
    today = date.today()
    return today.year - d.year - ((today.month, today.day) < (d.month, d.day))


# ---------------------------------------------------------------------------
# API routes (registered FIRST so they take priority over SPA fallback)
# ---------------------------------------------------------------------------
@app.get("/api-info")
def root_info() -> dict:
    return {
        "service": "BAHAWAL CARDIOLOGIST HOSPITAL — Heart Risk API",
        "model":   "Random Forest (Cleveland dataset, 1,025 patients)",
        "version": "2.0.0",
        "endpoints": {
            "predict":     "POST /predict       (8 model features)",
            "predict_full":"POST /predict-full  (hospital form JSON)",
            "model_info":  "GET  /model-info",
            "health":      "GET  /health",
            "docs":        "GET  /docs",
            "ui":          "GET  /             → the full hospital React app",
        },
    }


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "model_loaded": model is not None}


@app.get("/model-info")
def model_info() -> dict:
    return {
        "model_type":         type(model).__name__,
        "n_features":         model.n_features_in_,
        "feature_order":      feature_columns,
        "feature_defaults":   feature_defaults,
        "feature_importances": {
            name: float(imp) for name, imp in zip(feature_columns, model.feature_importances_)
        },
    }


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest) -> PredictResponse:
    try:
        X = _build_feature_row(req.model_dump())
        proba = float(model.predict_proba(X)[0][1])
        return PredictResponse(
            risk_probability=round(proba, 4),
            risk_level=_classify(proba),
            model_version="2.0.0-rf",
            disclaimer=(
                "This is a risk estimation tool only. "
                "It does NOT provide medical diagnosis. "
                "All final decisions must be made by qualified healthcare professionals."
            ),
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {e!s}") from e


@app.post("/predict-full", response_model=HospitalFormResponse)
def predict_full(req: HospitalFormRequest) -> HospitalFormResponse:
    try:
        age = _age_from_dob(req.patient_info.date_of_birth)
        md = req.medical_data.model_dump()

        payload = {
            "age":      age,
            "trestbps": float(md["trestbps"]),
            "chol":     float(md["chol"]),
            "cp":       float(md["cp"]),
            "ca":       float(md["ca"]),
            "thalach":  float(md["thalach"]),
            "thal":     float(md["thal"]),
            "oldpeak":  float(md["oldpeak"]),
        }
        X = _build_feature_row(payload)
        proba = float(model.predict_proba(X)[0][1])

        if proba < 0.5:
            category, alert, rec = "Moderate", "normal", "Schedule a routine check-up with your doctor"
        elif proba < 0.7:
            category, alert, rec = "High", "high", "Consult a cardiologist within 2 weeks"
        else:
            category, alert, rec = "Very High", "critical", "IMMEDIATE cardiology consultation required!"

        return HospitalFormResponse(
            success=True,
            risk_probability=round(proba, 4),
            risk_score=int(round(proba * 100)),
            risk_level=category,
            risk_category=category,
            has_heart_disease=proba >= 0.5,
            recommendation=rec,
            recommendations=rec,
            alert_level=alert,
            timestamp=datetime.now(timezone.utc).isoformat(),
            request_id=f"{int(datetime.now(timezone.utc).timestamp() * 1000)}-{uuid.uuid4().hex[:6]}",
            disclaimer=(
                "This is a risk estimation tool only. "
                "It does NOT provide medical diagnosis. "
                "All final decisions must be made by qualified healthcare professionals."
            ),
            model_version="2.0.0-rf",
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {e!s}") from e


# ---------------------------------------------------------------------------
# Frontend (React SPA) — served with SPA fallback so sub-routes don't 404
# ---------------------------------------------------------------------------
STATIC_DIR = BASE_DIR / "static_frontend"
if STATIC_DIR.exists():
    # Serve hashed assets directly (JS, CSS, images)
    app.mount("/assets", StaticFiles(directory=str(STATIC_DIR / "assets")), name="assets")

    @app.get("/", response_class=HTMLResponse)
    @app.get("/{path:path}", response_class=HTMLResponse, include_in_schema=False)
    def spa_fallback(path: str = ""):
        # Don't intercept real API/docs/health paths
        if path.startswith(("predict", "health", "model-info", "docs", "openapi", "api-info", "assets")):
            return JSONResponse({"detail": "Not Found"}, status_code=404)
        index = STATIC_DIR / "index.html"
        if index.exists():
            return FileResponse(index)
        return JSONResponse({"error": "frontend dist not found"}, status_code=404)
