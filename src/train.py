# train.py
# Purpose: Train forecasting models and save them
# Why: We need to train once and serve predictions many times via API

import pandas as pd
import numpy as np
import joblib
import os
import json  # Added for native Prophet serialization
from prophet.serialize import model_to_json  # Added for native Prophet serialization
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import xgboost as xgb

def time_series_train_test_split(df, test_size=0.2):
    """
    Split time series data — NEVER use random split for time series.
    Why: Random splitting creates data leakage. The future cannot be used 
         to predict the past. Always split chronologically.
    test_size: fraction of data to hold out as test set
    """
    split_idx = int(len(df) * (1 - test_size))
    train = df.iloc[:split_idx]
    test = df.iloc[split_idx:]
    print(f"Train: {train.index.min()} to {train.index.max()} ({len(train)} rows)")
    print(f"Test:  {test.index.min()} to {test.index.max()} ({len(test)} rows)")
    return train, test

def train_xgboost_model(df, target_col, feature_cols):
    """
    Train XGBoost regressor — our primary model.
    Why XGBoost: handles non-linear relationships, works well with tabular data,
    fast to train, interpretable via feature importance, industry standard.
    """
    train, test = time_series_train_test_split(df)
    
    X_train = train[feature_cols]
    y_train = train[target_col]
    X_test = test[feature_cols]
    y_test = test[target_col]

    missing_cols = [col for col in feature_cols if col not in df.columns]

    if missing_cols:
        raise ValueError(f"Missing feature columns: {missing_cols}")
        
    # XGBoost parameters — tuned for time series
    model = xgb.XGBRegressor(
        n_estimators=200,        # Number of trees
        learning_rate=0.05,      # Step size — lower = more careful learning
        max_depth=4,             # Tree depth — lower prevents overfitting
        subsample=0.8,           # Row sampling per tree
        colsample_bytree=0.8,    # Feature sampling per tree
        min_child_weight=3,      # Minimum samples in a leaf
        random_state=42,
        verbosity=0
    )
    
    model.fit(
        X_train, y_train,
        eval_set=[(X_test, y_test)],
        verbose=False
    )
    
    # Evaluate
    y_pred = model.predict(X_test)
    metrics = evaluate_model(y_test, y_pred)
    
    print("\n📊 XGBoost Evaluation:")
    for k, v in metrics.items():
        print(f"  {k}: {v:.4f}")
    
    return model, metrics, (X_test, y_test, y_pred)

def train_prophet_model(df, target_col):
    """
    Train Facebook Prophet — our interpretable trend/seasonality model.
    Why Prophet: excellent for trend decomposition, handles seasonality automatically,
    generations confidence intervals, easy to explain to non-technical stakeholders.
    Prophet expects specific column names: 'ds' for date, 'y' for target.
    """
    from prophet import Prophet
    
    # Prepare data in Prophet format
    prophet_df = df[[target_col]].reset_index()
    prophet_df.columns = ["ds", "y"]
    
    split_idx = int(len(prophet_df) * 0.8)
    train_df = prophet_df.iloc[:split_idx]
    test_df = prophet_df.iloc[split_idx:]
    
    model = Prophet(
        yearly_seasonality=True,
        weekly_seasonality=False,   # Quarterly data — no weekly patterns
        daily_seasonality=False,    # Quarterly data — no daily patterns
        changepoint_prior_scale=0.05,  # Flexibility of trend changes
        seasonality_prior_scale=10,
        interval_width=0.95         # 95% confidence intervals
    )
    
    model.fit(train_df)
    
    # Forecast over test period + 8 quarters future
    future = model.make_future_dataframe(periods=8, freq="QS")
    forecast = model.predict(future)
    
    # Evaluate on test period only

    # Merge forecast with actual test dates
    test_forecast = pd.merge(
        test_df,
        forecast[["ds", "yhat"]],
        on="ds",
        how="inner"
    )

    metrics = evaluate_model(
        test_forecast["y"].values,
        test_forecast["yhat"].values
    )
    
    print("\n📊 Prophet Evaluation:")
    for k, v in metrics.items():
        print(f"  {k}: {v:.4f}")
    
    return model, forecast, metrics

def evaluate_model(y_true, y_pred):
    """
    Standard forecasting evaluation metrics.
    Returns a dict of all major metrics.
    """
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mape = np.mean(np.abs((y_true - y_pred) / np.maximum(np.abs(y_true), 1e-8))) * 100
    r2 = r2_score(y_true, y_pred)
    
    return {"MAE": mae, "RMSE": rmse, "MAPE": mape, "R2": r2}

#--------------------------------------------#
#--------------------------------------------#

def save_models(xgb_model, prophet_model, feature_cols, output_dir="models"):
    """Save trained models for use by the API."""
    os.makedirs(output_dir, exist_ok=True)
    
    # Keep XGBoost and Features as pkl
    joblib.dump(xgb_model, f"{output_dir}/xgb_model.pkl")
    joblib.dump(feature_cols, f"{output_dir}/feature_cols.pkl")
    
    # ⚠️ FIX: Save Prophet model natively using JSON serialization
    with open(f"{output_dir}/prophet_model.json", "w") as fout:
        json.dump(model_to_json(prophet_model), fout)
    
    print(f"\n✅ Models saved to {output_dir}/")

def get_feature_importance(model, feature_cols):
    importance_df = pd.DataFrame({
        "feature": feature_cols,
        "importance": model.feature_importances_
    }).sort_values("importance", ascending=False)

    return importance_df

if __name__ == "__main__":

    from src.preprocessing import create_master_dataset
    from src.features import build_feature_matrix

    # Create dataset
    df = create_master_dataset()

    # Build features
    featured_df = build_feature_matrix(df)

    target_col = "global_semiconductor_revenue_bn"

    feature_cols = [
        col for col in featured_df.columns
        if col != target_col
    ]

    # Train XGBoost
    xgb_model, metrics, _ = train_xgboost_model(
        featured_df,
        target_col,
        feature_cols
    )

    # Train Prophet
    prophet_model, forecast, prophet_metrics = train_prophet_model(
        featured_df,
        target_col
    )

    # Save models
    save_models(
        xgb_model,
        prophet_model,
        feature_cols
    )

    print("\n✅ Training pipeline completed successfully")