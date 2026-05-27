# Concert Revenue Predictor

Predicts gross revenue (`avg_gross_usd`) for a configured concert using artist, venue, ticket price, date, and market. Includes a Streamlit dashboard for interactive use.

This is a **demo prototype** for an internal conversation, not production software. It is a *prediction* tool, not a *recommendation* tool — see the "About this tool" section in the dashboard for the full honesty disclosure.

---

## Quick start

```bash
# 1. Create venv (Python 3.12)
python -m venv .venv
.venv\Scripts\activate          # Windows PowerShell
# source .venv/bin/activate     # macOS/Linux

# 2. Install deps
pip install streamlit xgboost pandas scikit-learn matplotlib numpy plotly

# 3. (Optional) retrain the model — outputs are already checked in
python train_model.py

# 4. Run the dashboard
streamlit run streamlit_app.py
```

The dashboard opens at <http://localhost:8501>.

---

## Files

| File                                | Purpose                                                                              |
|-------------------------------------|--------------------------------------------------------------------------------------|
| `streamlit_app.py`                  | Dashboard entry point (dark-themed Streamlit app)                                    |
| `model_utils.py`                    | Bundle loader, feature constructor, prediction, peer-revenue helpers                 |
| `train_model.py`                    | Standalone training script (GroupKFold CV by headliner)                              |
| `revenue_predictor.json`            | Saved XGBoost model                                                                  |
| `revenue_predictor_features.json`   | Feature list in the order the model expects                                          |
| `cv_stats.json`                     | Cross-validation metrics (powers the confidence band)                                |
| `test_sample_cleaned_v2.csv`        | Training data (1,808 events × 103 columns, z-scored)                                 |
| `scaler_stats_apr8.json`            | Z-score mean/std per column — used to convert predictions back to dollars            |
| `.streamlit/config.toml`            | Theme config (dark + cyan accent)                                                    |

---

## How prediction works

1. **User supplies six inputs**: artist, genre, market, capacity (seats), ticket price ($), event date.
2. **The feature constructor** in `model_utils.build_feature_row()` builds a full 78-feature row:
   - Starts from training-data medians for everything.
   - Overrides one-hots for genre, market, headliner.
   - Sets `ticket_price_avg` and `avg_event_capacity` from user inputs (z-scored via `scaler_stats_apr8.json`).
   - Computes `month_sin/cos`, `day_of_week_sin/cos`, `year`, `year_offset`, `lockdown` from the event date.
   - **Artist signals** (Google Trends, Wikipedia views, historical concerts, days-since-last-album, past-year tickets):
     - If the artist is in training → take the median of their historical events.
     - If not → take the median across all events in the chosen genre, and track which features were imputed.
3. **XGBoost predicts** a z-scored revenue.
4. **Confidence range** is `prediction ± CV_RMSE` (z-scored), with both ends converted to dollars and the low end floored at $0.
5. **Peer comparison** filters training events to the same genre and a similar venue-capacity tier (small / mid / large), then shows the prediction inside that distribution.

---

## Model details

- **Algorithm**: XGBoost regressor, 400 trees, depth 4, learning rate 0.05, mild L1 + L2.
- **Cross-validation**: `GroupKFold(n_splits=5)` grouped by headliner — no artist appears in both train and test. This is essential because 5 artists account for ~25% of the dataset.
- **CV performance** (from `cv_stats.json`):
  - R² ≈ 0.90 (±0.05 across folds)
  - MAE ≈ 0.08 (z-scored units)
  - One fold is much worse than the others — this is real signal that some artist clusters are harder to predict, not a bug.
- **Confidence interval**: ±1σ from CV RMSE, converted to dollars (~±$227K).

---

## What this is **not**

The dashboard's "About this tool" expander spells this out, but in short:

- Not a recommendation engine. It tells you what events like this *typically earned*, not what this event would earn at a different price.
- No causal modeling. The data has endogeneity (popular shows get higher prices), so the model can't be inverted to find "the right price."
- Limited training data (~1,800 events, heavy Country / Latin / Pop-Rock bias). Unusual artists, markets, or venues are less reliable.

---

## Re-training

If you update `test_sample_cleaned_v2.csv`, regenerate the artifacts with:

```bash
python train_model.py
```

This re-runs CV, prints the per-fold scores, and overwrites `revenue_predictor.json`, `revenue_predictor_features.json`, and `cv_stats.json`. The dashboard picks up the new files automatically on next launch.
