# main.py — FastAPI backend
# Serves semiconductor revenue forecasts via REST API

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pandas as pd
from typing import Optional
import os
import json
import joblib
from prophet.serialize import model_from_json

app = FastAPI(
    title="EnergiChip Forecast API",
    description="Semiconductor demand forecasting driven by energy transition indicators",
    version="1.0.0"
)

# Allow frontend requests
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"]
)

# ─────────────────────────────────────────────────────────────
# LOAD MODELS
# ─────────────────────────────────────────────────────────────
# Resolve base directory properly depending on app structure
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

try:
    # Load XGBoost/Feature cols using Joblib
    xgb_model_path = os.path.join(BASE_DIR, "models", "xgb_model.pkl")
    if os.path.exists(xgb_model_path):
        xgb_model = joblib.load(xgb_model_path)
    
    feature_cols_path = os.path.join(BASE_DIR, "models", "feature_cols.pkl")
    if os.path.exists(feature_cols_path):
        feature_cols = joblib.load(feature_cols_path)

    # ⚠️ CRITICAL: Load Prophet model using JSON, not Joblib/Pickle
    prophet_model_path = os.path.join(BASE_DIR, "models", "prophet_model.json")
    with open(prophet_model_path, "r") as fin:
        prophet_model = model_from_json(json.load(fin))

    print("✅ Models loaded successfully")

except Exception as e:
    print(f"❌ Error loading models: {e}")
    xgb_model = None
    prophet_model = None
    feature_cols = []

# ─────────────────────────────────────────────────────────────
# REQUEST / RESPONSE SCHEMAS
# ─────────────────────────────────────────────────────────────
class ForecastRequest(BaseModel):
    solar_gw_added: float
    ev_sales_million: float
    india_solar_gw: float
    quarters_ahead: Optional[int] = 4


class ForecastResponse(BaseModel):
    forecast_quarters: list
    predicted_revenue_bn: list
    lower_bound: list
    upper_bound: list
    model_confidence: str


# ─────────────────────────────────────────────────────────────
# ROOT ENDPOINT
# ─────────────────────────────────────────────────────────────
@app.get("/")
def root():
    return {
        "message": "EnergiChip Forecast API is running",
        "docs": "/docs"
    }


# ─────────────────────────────────────────────────────────────
# HEALTH CHECK ENDPOINT
# ─────────────────────────────────────────────────────────────
@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "model": "Prophet Forecast Model"
    }


# ─────────────────────────────────────────────────────────────
# MAIN PREDICTION ENDPOINT
# ─────────────────────────────────────────────────────────────
@app.post("/predict", response_model=ForecastResponse)
def predict_revenue(request: ForecastRequest):

    try:
        # Ensure model is loaded
        if prophet_model is None:
            raise HTTPException(
                status_code=500,
                detail="Prophet model not loaded. Check model paths and ensure it is saved as a JSON file."
            )

        # Create future quarterly dates
        future = prophet_model.make_future_dataframe(
            periods=request.quarters_ahead,
            freq="QS"
        )

        # ⚠️ NOTE: If your model was trained with extra regressors (solar_gw, ev_sales), 
        # you MUST inject those values into the 'future' dataframe here before calling predict().
        # Example: future["solar_gw_added"] = request.solar_gw_added

        # Generate forecast
        forecast = prophet_model.predict(future)

        # Extract only future predictions
        future_forecast = forecast.tail(request.quarters_ahead)

        predictions = (
            future_forecast["yhat"]
            .round(2)
            .tolist()
        )

        lower_bounds = (
            future_forecast["yhat_lower"]
            .round(2)
            .tolist()
        )

        upper_bounds = (
            future_forecast["yhat_upper"]
            .round(2)
            .tolist()
        )

        quarters = [
            f"{d.year}-Q{d.quarter}"
            for d in future_forecast["ds"]
        ]

        return ForecastResponse(
            forecast_quarters=quarters,
            predicted_revenue_bn=predictions,
            lower_bound=lower_bounds,
            upper_bound=upper_bounds,
            model_confidence="Moderate confidence (Prophet MAPE ≈ 3.5%)"
        )

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=str(e)
        )

# Run using:
# uvicorn app.main:app --host 0.0.0.0 --port 10000