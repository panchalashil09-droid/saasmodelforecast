# 🚀 SaaS Revenue Intelligence & Forecasting

An AI-powered, gamified business intelligence and forecasting dashboard for SaaS
subscription revenue — built on top of the exact methodology developed in
`saas_revenue_forecasting.ipynb`.

## 1. Project Overview

This Streamlit application turns a time-series revenue forecasting notebook into a
production-style dashboard. It lets you:

- Explore historical revenue, subscriber, churn, and monetization trends across three
  business segments.
- Generate real, model-backed **next-month and multi-month revenue forecasts** with
  prediction intervals.
- Review model performance (accuracy, comparison across candidate algorithms, feature
  importance).
- See gamified summaries (Revenue Health Score, segment leaderboard, revenue
  milestones) that make the numbers easier to communicate to a non-technical audience.

## 2. Dataset

`simulated_saas_subscription_revenue_data.csv` — 141 monthly rows (47 months × 3
segments), January 2021 – November 2024.

Segments: `AI_Assistant`, `Streaming`, `Telecom_Hosting`.

Target variable: `target_next_month_revenue_usd` (next month's revenue for that
segment).

**This dataset is fully simulated/synthetic**, created for learning and
demonstration purposes — it does not represent a real company.

## 3. Forecasting Methodology

The app reuses the exact pipeline from the notebook (see `utils/forecasting.py`):

1. **Feature engineering** — revenue lags (1/2/3/6/12 months), rolling mean/std
   (3/6/12 months), month-over-month and quarter-over-quarter growth rates,
   one-hot encoded segments, and calendar features (year/month/quarter). Every
   feature for month *T* only uses information available at or before month *T*
   (no leakage).
2. **Chronological train/test split** — never random, since this is a genuine
   forecasting problem.
3. **Candidate models** — Linear Regression, Random Forest, Gradient Boosting,
   XGBoost, and LightGBM (the last two are used only if installed).
4. **Validation** — `TimeSeriesSplit` expanding-window cross-validation, plus a
   held-out chronological test set.
5. **Model selection** — the model that wins on the most metrics among RMSE / MAE /
   MAPE (not R² alone, which can look artificially strong on a small, trending test
   set).
6. **Recursive multi-step forecasting** — each predicted month feeds the lag/rolling
   features for the next step; exogenous (non-revenue) drivers are held at their most
   recent known values (a standard base-case assumption).
7. **Prediction intervals** — residual-based (test-set residual standard deviation,
   widened by `sqrt(horizon)` per step ahead) — labeled as estimates, not formal
   statistical confidence intervals.

## 4. Model Used

The app automatically selects and deploys whichever candidate model wins the
majority-vote selection rule above when you (re)train it — on this dataset that is
typically **XGBoost**, but the code makes no hard assumption; check the
**🤖 Model Intelligence** tab in the app for the currently deployed model and its
metrics.

## 5. Installation

```bash
pip install -r requirements.txt
```

## 6. Running Locally

```bash
streamlit run app.py
```

The first time you run the app, if no trained model is found it will show an
**⚙️ Train Forecasting Model** button. Click it once — training takes well under a
minute on this dataset — and the model artifacts will be saved to disk for all
future runs.

You can also train ahead of time from the command line:

```bash
python train_model.py
```

## 7. File Structure

```text
saas-revenue-forecasting/
│
├── app.py                                   # Main Streamlit application
├── train_model.py                           # Offline training script
├── simulated_saas_subscription_revenue_data.csv
├── saas_revenue_forecasting_model.pkl        # Trained model (created by train_model.py)
├── feature_columns.pkl                       # Ordered list of model input features
├── model_artifacts.pkl                       # Metrics, importances, CV results, etc.
├── model_evaluation_results.csv
├── requirements.txt
├── README.md
│
└── utils/
    ├── data_loader.py       # CSV loading, filtering, KPI aggregation (cached)
    ├── forecasting.py       # Feature engineering, training, recursive forecasting
    ├── visualizations.py    # Plotly chart builders
    └── styles.py            # Custom CSS for the dashboard theme
```

## 8. Streamlit Deployment

To deploy on [Streamlit Community Cloud](https://streamlit.io/cloud) or a similar
platform:

1. Push this folder to a GitHub repository.
2. Point the deployment at `app.py`.
3. Make sure `simulated_saas_subscription_revenue_data.csv` (and the pre-trained
   `.pkl` artifacts, if you want to skip the first-run training step) are committed
   alongside the code.
4. Set the Python version/requirements from `requirements.txt`.

## 9. How to Replace the Dataset

Replace `simulated_saas_subscription_revenue_data.csv` with a new CSV that has the
same schema (same column names, one row per month per segment, at least 12 months of
history per segment), then delete the existing `.pkl` artifacts and either:

- Click **⚙️ Train Forecasting Model** in the app, or
- Run `python train_model.py` from the command line.

## 10. How to Retrain the Model

```bash
python train_model.py
```

This re-runs the full pipeline (feature engineering → candidate models →
cross-validation → light hyperparameter tuning → model selection → refit on all
data) and overwrites the saved artifacts. Retrain whenever new monthly actuals
become available, since the lag/rolling features depend on having the most recent
history.

---

Built with **Python • Streamlit • Scikit-learn • XGBoost • LightGBM • Plotly**.
