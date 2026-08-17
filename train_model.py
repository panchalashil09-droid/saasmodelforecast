"""
Offline training script for the SaaS Revenue Forecasting model.

Run this once (or whenever you want to retrain on fresh data) to produce all the
artifacts the Streamlit app needs:

    python train_model.py

It reuses the exact same feature engineering / model-selection logic defined in
utils/forecasting.py, which mirrors saas_revenue_forecasting.ipynb.
"""

import json
import warnings
import joblib
import pandas as pd

from utils.forecasting import train_and_evaluate, DRIVER_COLS

warnings.filterwarnings("ignore")

DATA_PATH = "simulated_saas_subscription_revenue_data.csv"
MODEL_PATH = "saas_revenue_forecasting_model.pkl"
FEATURE_COLS_PATH = "feature_columns.pkl"
ARTIFACTS_PATH = "model_artifacts.pkl"


def main():
    print("Loading dataset...")
    df = pd.read_csv(DATA_PATH)
    df["date"] = pd.to_datetime(df["date"])
    print(f"  {df.shape[0]} rows, {df.shape[1]} columns, "
          f"{df['segment'].nunique()} segments, "
          f"{df['date'].min().date()} to {df['date'].max().date()}")

    print("\nTraining candidate models and selecting the best one "
          "(chronological split, TimeSeriesSplit CV, majority-vote selection)...")
    results = train_and_evaluate(df, tune=True)

    print(f"\nRecommended model: {results['recommended_model_name']}")
    print("Final comparison table (chronological test set):")
    print(results["final_comparison"])

    print("\nSaving artifacts...")
    joblib.dump(results["final_model"], MODEL_PATH)
    joblib.dump(results["feature_cols"], FEATURE_COLS_PATH)

    # Bundle everything else the app needs into one artifacts file
    artifacts = {
        "segment_dummy_cols": results["segment_dummy_cols"],
        "recommended_model_name": results["recommended_model_name"],
        "final_comparison": results["final_comparison"],
        "cv_summary": results["cv_summary"],
        "baseline_results": results["baseline_results"],
        "final_metrics_best": results["final_metrics_best"],
        "residual_std": results["residual_std"],
        "importances": results["importances"],
        "segment_metrics_df": results["segment_metrics_df"],
        "test_predictions_df": results["test_predictions_df"],
        "cutoff_date": results["cutoff_date"],
        "xgb_available": results["xgb_available"],
        "lgb_available": results["lgb_available"],
        "trained_at": pd.Timestamp.now().isoformat(),
    }
    joblib.dump(artifacts, ARTIFACTS_PATH)

    results["final_comparison"].to_csv("model_evaluation_results.csv")

    print(f"  Saved: {MODEL_PATH}")
    print(f"  Saved: {FEATURE_COLS_PATH}")
    print(f"  Saved: {ARTIFACTS_PATH}")
    print(f"  Saved: model_evaluation_results.csv")
    print("\nDone. You can now run: streamlit run app.py")


if __name__ == "__main__":
    main()
