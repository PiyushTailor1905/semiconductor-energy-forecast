EnergiChip Forecast — Semiconductor Energy Demand Intelligence

An end-to-end machine learning system that forecasts semiconductor market demand using global energy transition indicators such as solar capacity expansion, EV adoption, and clean energy trends.

The system combines time-series forecasting + machine learning + interactive dashboarding to simulate how energy transitions influence semiconductor demand globally and in India.

# Problem Statement

Semiconductor demand is strongly influenced by macro trends like electrification and renewable energy growth.
This project models that relationship and builds a forecasting system that can:

Predict future semiconductor market revenue
Simulate energy-driven demand scenarios
Compare India vs Global semiconductor trends

# Key Idea

Energy Transition → Industrial Electrification → Semiconductor Demand Growth

We model semiconductor revenue as a function of:

Solar capacity additions (GW)
EV adoption (millions of units)
Historical demand trends
# System Architecture
Data Sources → Preprocessing → Feature Engineering → Model Training
      ↓
 XGBoost + Prophet Models
      ↓
 FastAPI Backend (Prediction API)
      ↓
 Streamlit Frontend (Dashboard UI)
 
Features
Time-series forecasting of semiconductor demand
India vs Global comparison view
Scenario-based simulation (solar + EV inputs)
Hybrid modeling: XGBoost + Prophet
Confidence interval-based forecasting
Modular ML pipeline (production-style structure)

# Tech Stack

Python
Pandas, NumPy
Scikit-learn, XGBoost
Prophet
FastAPI
Streamlit
Plotly
Joblib

# Project Structure
semiconductor-energy-forecast/
│
├── data/
├── notebooks/
├── src/
│   ├── data_loader.py
│   ├── preprocessing.py
│   ├── features.py
│   ├── train.py
│   ├── evaluate.py
│
├── app/
│   └── main.py
│
├── frontend/
│   └── app.py
│
├── models/
├── requirements.txt
└── README.md

# How to Run Locally
1. Install dependencies
pip install -r requirements.txt
2. Train models
python src/train.py
3. Start backend API
uvicorn app.main:app --reload
4. Start frontend dashboard
streamlit run frontend/app.py

# Models Used
XGBoost Regressor
Captures non-linear relationships
Best for structured tabular forecasting
Primary prediction engine
Prophet
Trend + seasonality modeling
Confidence intervals
Interpretability for stakeholders

# Results (Sample)
XGBoost: strong directional accuracy (R² varies by split)
Prophet: stable trend forecasting with low error on smoothed series

# Key Learnings
Time-series forecasting must respect chronological splits
Feature engineering is more important than model complexity
Real-world forecasting requires hybrid approaches
Deployment architecture matters as much as model accuracy

# Future Improvements
Replace synthetic data with real semiconductor supply chain data
Add real-time API integration (FRED / World Bank live sync)
Improve uncertainty estimation (Bayesian models)
Add anomaly detection for supply shocks

# Author
# Piyush Tailor
