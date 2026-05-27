"""Train the concert revenue predictor.

Run from the project root:
    python train_model.py

Produces:
    revenue_predictor.json          XGBoost model
    revenue_predictor_features.json Feature list (order matters)
    cv_stats.json                   CV RMSE / MAE / R^2 (used for confidence bands)
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupKFold
from xgboost import XGBRegressor

DATA_PATH = "test_sample_cleaned_v2.csv"
MODEL_PATH = "revenue_predictor.json"
FEATURES_PATH = "revenue_predictor_features.json"
CV_STATS_PATH = "cv_stats.json"

TARGET = "avg_gross_usd"

LEAKAGE = [
    "avg_tickets_sold",
    "avg_capacity_sold",
    "ticket_price_min",
    "ticket_price_max",
    "number_of_shows",
]

NON_FEATURES = [
    "eventid", "event_date",
    "headliner", "support", "venue", "city", "state", "country", "market",
    "company_type", "promoter", "genre",
    "gt_date_range", "last_album_date",
    "census_market_name", "market_clean", "acs_market",
    "day_of_week", "month",
]

XGB_PARAMS = dict(
    n_estimators=400,
    max_depth=4,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_alpha=0.5,
    reg_lambda=3.0,
    min_child_weight=5,
    random_state=42,
    n_jobs=-1,
)

N_FOLDS = 5


def build_xy(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series, np.ndarray]:
    groups = df["headliner"].fillna("UNKNOWN").to_numpy()

    drop_cols = [c for c in (LEAKAGE + NON_FEATURES) if c in df.columns]
    X = df.drop(columns=drop_cols + [TARGET], errors="ignore")
    y = df[TARGET]

    bool_cols = X.select_dtypes(include="bool").columns
    X[bool_cols] = X[bool_cols].astype(int)

    non_numeric = X.select_dtypes(exclude="number").columns.tolist()
    if non_numeric:
        print(f"Dropping non-numeric residual columns: {non_numeric}")
        X = X.drop(columns=non_numeric)

    return X, y, groups


def cross_validate(X: pd.DataFrame, y: pd.Series, groups: np.ndarray) -> pd.DataFrame:
    gkf = GroupKFold(n_splits=N_FOLDS)
    rows = []
    for fold, (tr, te) in enumerate(gkf.split(X, y, groups), start=1):
        m = XGBRegressor(**XGB_PARAMS)
        m.fit(X.iloc[tr], y.iloc[tr])
        pred = m.predict(X.iloc[te])
        rows.append({
            "fold": fold,
            "train_n": len(tr),
            "test_n": len(te),
            "MAE": mean_absolute_error(y.iloc[te], pred),
            "RMSE": float(np.sqrt(mean_squared_error(y.iloc[te], pred))),
            "R2": r2_score(y.iloc[te], pred),
        })
    return pd.DataFrame(rows)


def main() -> None:
    print(f"Loading {DATA_PATH}")
    df = pd.read_csv(DATA_PATH)
    print(f"  {len(df)} rows, {df.shape[1]} columns, {df['headliner'].nunique()} unique headliners")

    X, y, groups = build_xy(df)
    print(f"Feature count: {X.shape[1]}")

    print(f"\nRunning {N_FOLDS}-fold GroupKFold CV (by headliner)...")
    cv_df = cross_validate(X, y, groups)
    print(cv_df.to_string(index=False))

    cv_summary = {
        "n_folds": N_FOLDS,
        "mean_MAE": float(cv_df["MAE"].mean()),
        "std_MAE": float(cv_df["MAE"].std()),
        "mean_RMSE": float(cv_df["RMSE"].mean()),
        "std_RMSE": float(cv_df["RMSE"].std()),
        "mean_R2": float(cv_df["R2"].mean()),
        "std_R2": float(cv_df["R2"].std()),
        "per_fold": cv_df.to_dict(orient="records"),
    }
    print(
        f"\nCV mean (std):  MAE={cv_summary['mean_MAE']:.4f} ({cv_summary['std_MAE']:.4f})"
        f"  RMSE={cv_summary['mean_RMSE']:.4f} ({cv_summary['std_RMSE']:.4f})"
        f"  R2={cv_summary['mean_R2']:.4f} ({cv_summary['std_R2']:.4f})"
    )

    print("\nFitting final model on all data...")
    final_model = XGBRegressor(**XGB_PARAMS)
    final_model.fit(X, y)

    final_model.save_model(MODEL_PATH)
    Path(FEATURES_PATH).write_text(json.dumps(list(X.columns), indent=2))
    Path(CV_STATS_PATH).write_text(json.dumps(cv_summary, indent=2))
    print(f"Saved: {MODEL_PATH}")
    print(f"Saved: {FEATURES_PATH}")
    print(f"Saved: {CV_STATS_PATH}")


if __name__ == "__main__":
    main()
