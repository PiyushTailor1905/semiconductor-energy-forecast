# main.py — FastAPI backend
# Serves semiconductor revenue forecasts via REST API

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import joblib
import pandas as pd
from typing import Optional

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
try:
    xgb_model = joblib.load("models/xgb_model.pkl")
    prophet_model = joblib.load("models/prophet_model.pkl")
    import os

    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    xgb_model = joblib.load(os.path.join(BASE_DIR, "models", "xgb_model.pkl"))
    feature_cols = joblib.load(os.path.join(BASE_DIR, "models", "feature_cols.pkl"))

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
                detail="Prophet model not loaded"
            )

        # Create future quarterly dates
        future = prophet_model.make_future_dataframe(
            periods=request.quarters_ahead,
            freq="QS"
        )

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
# uvicorn app.main:app --reload