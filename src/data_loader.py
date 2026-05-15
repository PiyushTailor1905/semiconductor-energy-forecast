# data_loader.py
# Purpose: Load data from all sources into clean DataFrames
# Why: Centralizing data loading makes the project modular and professional

import pandas as pd
import requests
import os

# ─── FRED DATA LOADER ────────────────────────────────────────────────────────
def load_fred_series(series_id, api_key=None):
    """
    Load a single data series from FRED (Federal Reserve Economic Data).
    series_id: the FRED series code, e.g. 'PPIACO' for Producer Price Index
    Returns: a pandas Series with DatetimeIndex
    """
    # FRED has a free API — get your key at fred.stlouisfed.org/docs/api/api_key.html
    # For demo purposes, we use the requests library directly
    url = f"https://api.stlouisfed.org/fred/series/observations"
    params = {
        "series_id": series_id,
        "api_key": api_key or os.getenv("FRED_API_KEY"),
        "file_type": "json",
        "observation_start": "2010-01-01"
    }
    if api_key is None:
        raise ValueError("FRED API key not found.")
    response = requests.get(url, params=params, timeout=10)
    response.raise_for_status()
    data = response.json()
    
    # Convert to DataFrame
    df = pd.DataFrame(data["observations"])
    df["date"] = pd.to_datetime(df["date"])
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    df = df.set_index("date")[["value"]]
    df.columns = [series_id]
    return df

# ─── WORLD BANK LOADER ───────────────────────────────────────────────────────
def load_worldbank_indicator(indicator, country_code, start_year=2010):
    """
    Load World Bank data for a specific indicator and country.
    indicator: WB indicator code, e.g. 'TX.VAL.TECH.CD' for tech exports
    country_code: ISO2 code, e.g. 'IN' for India, 'WLD' for World
    Returns: pandas DataFrame with year and value columns
    """
    url = f"https://api.worldbank.org/v2/country/{country_code}/indicator/{indicator}"
    params = {
        "format": "json",
        "per_page": 100,
        "date": f"{start_year}:2024"
    }
    response = requests.get(url, params=params)
    data = response.json()
    
    # World Bank returns [metadata, data_list]
    records = data[1] if len(data) > 1 else []
    rows = []
    for record in records:
        rows.append({
            "year": int(record["date"]),
            "value": record["value"],
            "country": record["country"]["value"]
        })
    
    df = pd.DataFrame(rows)
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    df = df.sort_values("year").reset_index(drop=True)
    return df

# ─── MANUAL DATA LOADER ──────────────────────────────────────────────────────
def load_csv_data(filepath, date_col=None):
    """
    Load any CSV file from the data/raw folder.
    date_col: if provided, parse this column as datetime index
    """
    df = pd.read_csv(filepath)
    if date_col and date_col in df.columns:
        df[date_col] = pd.to_datetime(df[date_col])
        df = df.set_index(date_col)
    return df

# ─── DEMO DATA GENERATOR ─────────────────────────────────────────────────────
def generate_synthetic_semiconductor_data():
    """
    IMPORTANT: Use this ONLY if real data is unavailable for certain series.
    Generates realistic synthetic data based on known market trends.
    Label it clearly as 'synthetic/estimated' in your project.
    """
    import numpy as np
    
    # Date range: quarterly from 2010 to 2024
    dates = pd.date_range(start="2010-01-01", end="2024-01-01", freq="QS")
    n = len(dates)
    
    np.random.seed(42)
    
    # Semiconductor market revenue (USD billions) — realistic trend
    # Global market grew from ~300B in 2010 to ~530B in 2023
    trend = np.linspace(300, 530, n)
    # Add realistic cyclicality — semiconductor cycles are ~4 years
    cycle = 40 * np.sin(2 * np.pi * np.arange(n) / 16)
    # Add noise
    noise = np.random.normal(0, 15, n)
    revenue = trend + cycle + noise
    
    # Solar GW additions (global) — grew from ~17 GW in 2010 to ~290 GW in 2023
    solar_gw = np.linspace(17, 290, n) + np.random.normal(0, 5, n)
    
    # EV sales (millions of units) — near zero in 2010, ~14M in 2023
    ev_sales = np.geomspace(0.01, 14, n) + np.random.normal(0, 0.3, n)
    ev_sales = np.maximum(ev_sales, 0)
    
    # India solar capacity (GW)
    india_solar = np.geomspace(0.5, 73, n) + np.random.normal(0, 1, n)
    
    df = pd.DataFrame({
        "date": dates,
        "global_semiconductor_revenue_bn": revenue,
        "global_solar_gw_added": solar_gw,
        "global_ev_sales_mn": ev_sales,
        "india_solar_capacity_gw": india_solar,
    }).set_index("date")
    
    return df
if __name__ == "__main__":
    df = generate_synthetic_semiconductor_data()
    print(df.head())