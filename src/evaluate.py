import os
os.makedirs("results", exist_ok=True)
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

def plot_actual_vs_predicted(y_true, y_pred, dates, title="Actual vs Predicted"):
    """
    The most important plot: shows how well the model tracks reality.
    """
    fig, axes = plt.subplots(2, 1, figsize=(14, 10))
    
    # Plot 1: Actual vs Predicted
    axes[0].plot(dates, y_true, label="Actual", color="#2196F3", linewidth=2)
    axes[0].plot(dates, y_pred, label="Predicted", color="#FF5722", 
                 linewidth=2, linestyle="--")
    axes[0].fill_between(dates, y_true, y_pred, alpha=0.2, color="gray")
    axes[0].set_title(title, fontsize=14, fontweight="bold")
    axes[0].legend()
    axes[0].set_ylabel("Revenue (USD Billions)")
    axes[0].grid(True, alpha=0.3)
    
    # Plot 2: Residuals
    residuals = np.array(y_true) - np.array(y_pred)
    axes[1].bar(dates, residuals, color=["#4CAF50" if r > 0 else "#F44336" 
                                          for r in residuals], alpha=0.7)
    axes[1].axhline(y=0, color="black", linewidth=1)
    axes[1].set_title("Prediction Residuals", fontsize=12)
    axes[1].set_ylabel("Residual (USD Billions)")
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig("results/actual_vs_predicted.png", dpi=150, bbox_inches="tight")
    plt.show()
    return fig

def plot_feature_importance(model, feature_names, top_n=15):
    """
    Feature importance plot — one of the most impressive things to show
    an interviewer. It demonstrates you understand what's driving predictions.
    """
    importances = model.feature_importances_
    top_n = min(top_n, len(feature_names))
    indices = np.argsort(importances)[::-1][:top_n]
    
    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.barh(
        range(top_n), 
        importances[indices][::-1],
        color="#1976D2",
        alpha=0.85
    )
    ax.set_yticks(range(top_n))
    ax.set_yticklabels([feature_names[i] for i in indices[::-1]])
    ax.set_xlabel("Feature Importance Score")
    ax.set_title("Top Features Driving Semiconductor Revenue Forecasts", 
                 fontsize=13, fontweight="bold")
    ax.grid(True, axis="x", alpha=0.3)
    
    plt.tight_layout()
    plt.savefig("results/feature_importance.png", dpi=150, bbox_inches="tight")
    plt.show()
    return fig

if __name__ == "__main__":

    from src.preprocessing import create_master_dataset
    from src.features import build_feature_matrix
    from src.train import train_xgboost_model

    df = create_master_dataset()

    featured_df = build_feature_matrix(df)

    target_col = "global_semiconductor_revenue_bn"

    feature_cols = [
        col for col in featured_df.columns
        if col != target_col
    ]

    model, metrics, test_data = train_xgboost_model(
        featured_df,
        target_col,
        feature_cols
    )

    X_test, y_test, y_pred = test_data

    plot_actual_vs_predicted(
        y_test,
        y_pred,
        X_test.index,
        # metrics=metrics
    )

    plot_feature_importance(
        model,
        feature_cols
    )

    print("\n✅ Evaluation plots generated successfully")