# preprocessing.py
# Purpose: Clean, merge, and align all data sources into one master DataFrame
# Why: Raw data from different sources has different frequencies, missing values,
#      and date formats. This step standardizes everything.

import pandas as pd
import numpy as np

def resample_to_quarterly(df, method="mean"):
    """
    Resample any DataFrame to quarterly frequency.
    method: 'mean' for continuous variables, 'sum' for cumulative ones
    """
    if method == "mean":
        return df.resample("QS").mean()
    elif method == "sum":
        return df.resample("QS").sum()
    else:
        return df.resample("QS").last()

def handle_missing_values(df, method="interpolate"):
    """
    Fill missing values in a time series DataFrame.
    interpolate: linear interpolation — best for smooth time series
    forward_fill: carry last known value forward — good for sparse data
    """
    if method == "interpolate":
        # Time-weighted linear interpolation
        df = df.interpolate(method="time")
    elif method == "forward_fill":
        df = df.ffill()
    
    # After interpolation, fill any remaining NaN at edges
    df = df.bfill().ffill()
    return df

def normalize_features(df, columns, method="minmax"):
    """
    Normalize feature columns so they're on a similar scale.
    This is important for models like LSTM but less critical for XGBoost.
    We'll track the scaler parameters to inverse-transform predictions later.
    """
    from sklearn.preprocessing import MinMaxScaler, StandardScaler
    
    scalers = {}
    df_scaled = df.copy()
    
    for col in columns:
        if method == "minmax":
            scaler = MinMaxScaler()
        else:
            scaler = StandardScaler()
        
        df_scaled[col] = scaler.fit_transform(df[[col]]).flatten()
        scalers[col] = scaler  # Save to inverse-transform later
    
    return df_scaled, scalers

def merge_global_india_data(global_df, india_df):
    """
    Merge global and India datasets on their date index.
    Uses outer join to keep all dates, then interpolates gaps.
    """
    merged = pd.merge(
        global_df, india_df,
        left_index=True, right_index=True,
        how="outer",
        suffixes=("_global", "_india")
    )
    merged = handle_missing_values(merged)
    return merged

def create_master_dataset():
    """
    Orchestrates the full data preparation pipeline.
    Returns the master DataFrame ready for feature engineering.
    """
    from src.data_loader import generate_synthetic_semiconductor_data
    
    # Load base data (replace with real data when available)
    df = generate_synthetic_semiconductor_data()
    df = df.sort_index()
    
    print(f"Dataset shape: {df.shape}")
    print(f"Date range: {df.index.min()} to {df.index.max()}")
    print(f"Missing values:\n{df.isnull().sum()}")
    
    # Handle missing values
    df = handle_missing_values(df)
    
    print("\n✅ Master dataset created successfully")
    return df

if __name__ == "__main__":
    df = create_master_dataset()
    print(df.head())