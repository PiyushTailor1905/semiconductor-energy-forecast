# features.py
# Purpose: Create features that help the model learn patterns
# Why: Raw data alone is rarely enough. We create features that capture
#      trends, seasonality, momentum, and cross-variable relationships.

import pandas as pd
import numpy as np

def add_lag_features(df, target_col, lags=[1, 2, 4, 8]):
    """
    Lag features: the value of the target at previous time steps.
    Why: The best predictor of next quarter's revenue is often 
         last quarter's revenue and the quarter a year ago.
    lags: list of quarters to look back
    """
    for lag in lags:
        df[f"{target_col}_lag{lag}"] = df[target_col].shift(lag)
    return df

def add_rolling_features(df, target_col, windows=[4, 8]):
    """
    Rolling statistics: moving average and moving standard deviation.
    Why: Captures medium-term trends and volatility patterns.
    windows: window sizes in quarters
    """
    for window in windows:
        df[f"{target_col}_rolling_mean_{window}q"] = (
            df[target_col].rolling(window=window).mean()
        )
        df[f"{target_col}_rolling_std_{window}q"] = (
            df[target_col].rolling(window=window).std()
        )
    return df

def add_growth_rates(df, target_col):
    """
    Quarter-over-Quarter and Year-over-Year growth rates.
    Why: Semiconductor market participants think in terms of growth rates,
         not absolute values. Including this helps the model learn industry patterns.
    """
    # Quarter-over-quarter change
    df[f"{target_col}_qoq_growth"] = df[target_col].pct_change(1) * 100
    # Year-over-year change (4 quarters back)
    df[f"{target_col}_yoy_growth"] = df[target_col].pct_change(4) * 100
    return df

def add_time_features(df):
    """
    Temporal features: quarter, year, trend index.
    Why: Models need to know WHERE in time they are. 
         Semiconductor demand has strong seasonality (Q4 is typically strong).
    """
    df["year"] = df.index.year
    df["quarter"] = df.index.quarter
    # Trend index: 0, 1, 2, 3, ... — captures long-term growth direction
    df["trend_index"] = np.arange(len(df))
    # Cyclical encoding of quarter (so Q4 and Q1 are close together)
    df["quarter_sin"] = np.sin(2 * np.pi * df["quarter"] / 4)
    df["quarter_cos"] = np.cos(2 * np.pi * df["quarter"] / 4)
    return df

def add_interaction_features(df):
    """
    Interaction features: combinations of variables that have joint explanatory power.
    Why: Solar + EV together drive power semiconductor demand more than either alone.
    """
    if "global_solar_gw_added" in df.columns and "global_ev_sales_mn" in df.columns:
        # Combined clean energy intensity
        df["clean_energy_momentum"] = (
            df["global_solar_gw_added"] * df["global_ev_sales_mn"]
        )
    return df

def build_feature_matrix(df, target_col="global_semiconductor_revenue_bn"):
    """
    Master function: applies all feature engineering steps.
    Drops rows with NaN (caused by lagging/rolling operations).
    """
    df = add_lag_features(df, target_col)
    df = add_rolling_features(df, target_col)
    df = add_growth_rates(df, target_col)
    df = add_time_features(df)
    df = add_interaction_features(df)
    
    # Drop rows where lag features are NaN (first N rows)
    df = df.dropna()
    
    print(f"Features created: {list(df.columns)}")
    print(f"Dataset shape after feature engineering: {df.shape}")
    
    return df
if __name__ == "__main__":
    from src.preprocessing import create_master_dataset

    df = create_master_dataset()

    featured_df = build_feature_matrix(df)

    print(featured_df.head())