"""Feature constructor and prediction utilities for the revenue dashboard.

The model expects ~78 z-scored features in a fixed order. The user only supplies six
inputs (artist, genre, capacity, price, date, market). This module fills in the rest:
artist-specific signals from training history when the artist is known, genre-peer
medians when they aren't.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
from xgboost import XGBRegressor

DATA_PATH = "test_sample_cleaned_v2.csv"
MODEL_PATH = "revenue_predictor.json"
FEATURES_PATH = "revenue_predictor_features.json"
SCALER_PATH = "scaler_stats_apr8.json"
CV_STATS_PATH = "cv_stats.json"

TARGET = "avg_gross_usd"

# Features pulled from artist's historical events when available
ARTIST_FEATURES = [
    "gt_avg_13w",
    "gt_max_13w",
    "gt_std_13w",
    "gt_momentum_13w",
    "wiki_avg_views_30d",
    "historical_concerts",
    "days_since_last_album",
    "past_year_avg_tickets",
    "album_release_last_12m",
]

# Genre key -> one-hot column name
GENRES = {
    "Pop / Rock":        "genre_cleaned_pop_rock",
    "Country":           "genre_cleaned_country",
    "Latin":             "genre_cleaned_latin",
    "Dance / Electronic": "genre_cleaned_dance_electronic",
    "Rap / HipHop":      "genre_cleaned_rap_hiphop",
    "Other":             "genre_cleaned_other",
}

# Market key -> one-hot column name. Market also drives the population/income
# columns through a lookup on the training rows.
MARKETS = {
    "New York":         "market_cleaned_new_york",
    "Los Angeles":      "market_cleaned_los_angeles",
    "Chicago":          "market_cleaned_chicago",
    "Washington DC":    "market_cleaned_washington_dc_hagerstown",
    "Boston":           "market_cleaned_boston_manchester",
    "Other":            "market_cleaned_other",
}

# Headliner one-hots that exist in the model (everything else maps to "other")
HEADLINER_ONEHOTS = {
    "Aventura":        "headliner_cleaned_aventura",
    "Christian Nodal": "headliner_cleaned_christian_nodal",
    "David Foster":    "headliner_cleaned_david_foster",
    "Lord Huron":      "headliner_cleaned_lord_huron",
    "Randall King":    "headliner_cleaned_randall_king",
}


@dataclass
class Bundle:
    """Everything the dashboard needs at runtime, loaded once."""
    model: XGBRegressor
    features: list[str]
    df: pd.DataFrame                # full training table (for lookups + peer comparison)
    scaler: dict                    # column -> {'mean': float, 'std': float}
    cv_stats: dict
    medians: pd.Series              # median feature row (z-scored)
    headliners: list[str]           # sorted list for autocomplete
    artist_lookup: dict             # headliner -> median of their training rows
    genre_lookup: dict              # genre col -> median of training rows in that genre

    @property
    def cv_rmse_z(self) -> float:
        return self.cv_stats["mean_RMSE"]


@dataclass
class PredictionResult:
    pred_z: float
    pred_dollars: float
    lo_dollars: float
    hi_dollars: float
    imputed_features: list[str] = field(default_factory=list)
    artist_known: bool = False
    n_peer_events: int = 0


def to_real(z: float, col: str, scaler: dict) -> float:
    """Inverse z-score for a given column."""
    s = scaler.get(col)
    if s is None:
        return z
    return z * s["std"] + s["mean"]


def to_scaled(real: float, col: str, scaler: dict) -> float:
    s = scaler.get(col)
    if s is None:
        return real
    return (real - s["mean"]) / s["std"]


def fmt_dollars(value: float) -> str:
    """$1,234,567 style formatting that handles negatives gracefully."""
    if value >= 0:
        return f"${value:,.0f}"
    return f"-${abs(value):,.0f}"


def load_bundle(root: str | Path = ".") -> Bundle:
    """Load everything the dashboard needs. Cache this with @st.cache_resource."""
    root = Path(root)

    model = XGBRegressor()
    model.load_model(str(root / MODEL_PATH))

    features = json.loads((root / FEATURES_PATH).read_text())
    scaler = json.loads((root / SCALER_PATH).read_text())
    cv_stats = json.loads((root / CV_STATS_PATH).read_text())

    df = pd.read_csv(root / DATA_PATH)

    # Reproduce the training X to compute medians and peer lookups
    bool_cols = df.select_dtypes(include="bool").columns
    df[bool_cols] = df[bool_cols].astype(int)

    X_template = df[features].copy()
    medians = X_template.median(numeric_only=True)

    # Artist lookup: median row per known headliner
    artist_lookup: dict[str, pd.Series] = {}
    for name, group in df.groupby("headliner"):
        artist_lookup[name] = group[features].median(numeric_only=True)

    # Genre lookup: median row per genre one-hot column (peer fallback)
    genre_lookup: dict[str, pd.Series] = {}
    for genre_col in GENRES.values():
        if genre_col not in df.columns:
            continue
        peers = df[df[genre_col] == 1]
        if len(peers) > 0:
            genre_lookup[genre_col] = peers[features].median(numeric_only=True)

    headliners = sorted(df["headliner"].dropna().unique().tolist())

    return Bundle(
        model=model,
        features=features,
        df=df,
        scaler=scaler,
        cv_stats=cv_stats,
        medians=medians,
        headliners=headliners,
        artist_lookup=artist_lookup,
        genre_lookup=genre_lookup,
    )


def date_features(d: date) -> dict[str, float]:
    """month_sin/cos and day_of_week_sin/cos for a given date."""
    month = d.month
    dow = d.weekday()  # Monday=0
    return {
        "month_sin":       float(np.sin(2 * np.pi * month / 12)),
        "month_cos":       float(np.cos(2 * np.pi * month / 12)),
        "day_of_week_sin": float(np.sin(2 * np.pi * dow / 7)),
        "day_of_week_cos": float(np.cos(2 * np.pi * dow / 7)),
        "year":            float(d.year),
        "year_offset":     float(d.year - 2020),
    }


def build_feature_row(
    bundle: Bundle,
    artist: str,
    genre: str,
    capacity: float,
    price: float,
    event_date: date,
    market: str,
) -> tuple[pd.Series, list[str], bool]:
    """Construct a single-row feature vector for the user's concert inputs.

    Returns (row, imputed_features, artist_known).
    """
    features = bundle.features
    row = bundle.medians.copy()
    imputed: list[str] = []

    # Genre one-hot
    genre_col = GENRES.get(genre, GENRES["Other"])
    for col in GENRES.values():
        if col in row.index:
            row[col] = 1 if col == genre_col else 0

    # Market one-hot (and pull market_population/median_income from a row in that market)
    mkt_col = MARKETS.get(market, MARKETS["Other"])
    for col in MARKETS.values():
        if col in row.index:
            row[col] = 1 if col == mkt_col else 0

    mkt_rows = bundle.df[bundle.df[mkt_col] == 1]
    if len(mkt_rows) > 0:
        for col in ("market_population", "population", "median_income"):
            if col in row.index:
                row[col] = mkt_rows[col].median()

    # Headliner one-hot (set the matching column if recognized, else mark "other")
    if artist in HEADLINER_ONEHOTS:
        target_col = HEADLINER_ONEHOTS[artist]
        for col in HEADLINER_ONEHOTS.values():
            if col in row.index:
                row[col] = 1 if col == target_col else 0
        if "headliner_cleaned_other" in row.index:
            row["headliner_cleaned_other"] = 0
    else:
        for col in HEADLINER_ONEHOTS.values():
            if col in row.index:
                row[col] = 0
        if "headliner_cleaned_other" in row.index:
            row["headliner_cleaned_other"] = 1

    # Controllable inputs (user-supplied dollars/seats, convert to z-score)
    row["ticket_price_avg"] = to_scaled(price, "ticket_price_avg", bundle.scaler)
    row["avg_event_capacity"] = to_scaled(capacity, "avg_event_capacity", bundle.scaler)

    # Date features
    for col, val in date_features(event_date).items():
        if col in row.index:
            row[col] = val
    # Lockdown only applies to 2020-2021
    if "lockdown" in row.index:
        row["lockdown"] = 1 if event_date.year in (2020, 2021) else 0

    # Artist-specific lookup or genre-peer imputation
    artist_known = artist in bundle.artist_lookup
    if artist_known:
        artist_row = bundle.artist_lookup[artist]
        for feat in ARTIST_FEATURES:
            if feat in row.index and feat in artist_row.index and not pd.isna(artist_row[feat]):
                row[feat] = artist_row[feat]
    else:
        genre_row = bundle.genre_lookup.get(genre_col)
        for feat in ARTIST_FEATURES:
            if feat in row.index:
                if genre_row is not None and feat in genre_row.index and not pd.isna(genre_row[feat]):
                    row[feat] = genre_row[feat]
                imputed.append(feat)

    # Return aligned to feature order
    return row[features], imputed, artist_known


def predict(
    bundle: Bundle,
    artist: str,
    genre: str,
    capacity: float,
    price: float,
    event_date: date,
    market: str,
) -> PredictionResult:
    row, imputed, artist_known = build_feature_row(
        bundle, artist, genre, capacity, price, event_date, market
    )

    X_input = pd.DataFrame([row.values], columns=bundle.features)
    pred_z = float(bundle.model.predict(X_input)[0])

    rmse_z = bundle.cv_rmse_z
    pred_dollars = to_real(pred_z, TARGET, bundle.scaler)
    lo_dollars = to_real(pred_z - rmse_z, TARGET, bundle.scaler)
    hi_dollars = to_real(pred_z + rmse_z, TARGET, bundle.scaler)

    # Floor the low end at $0 — negative dollar revenue is not meaningful for display
    lo_dollars = max(0.0, lo_dollars)

    n_peers = int((bundle.df[GENRES.get(genre, GENRES["Other"])] == 1).sum()) \
        if GENRES.get(genre, GENRES["Other"]) in bundle.df.columns else 0

    return PredictionResult(
        pred_z=pred_z,
        pred_dollars=pred_dollars,
        lo_dollars=lo_dollars,
        hi_dollars=hi_dollars,
        imputed_features=imputed,
        artist_known=artist_known,
        n_peer_events=n_peers,
    )


def peer_revenues(bundle: Bundle, genre: str, capacity: float) -> pd.DataFrame:
    """Return actual revenues (in $) for events similar to the user's inputs.

    Filters to the genre, then narrows by venue-capacity tier (small / mid / large)
    so the peer comparison is informative rather than a wall of unrelated events.
    """
    genre_col = GENRES.get(genre, GENRES["Other"])
    if genre_col not in bundle.df.columns:
        return pd.DataFrame()

    peers = bundle.df[bundle.df[genre_col] == 1].copy()
    if peers.empty:
        return pd.DataFrame()

    # Tier by user's capacity vs training distribution
    cap_real = peers["avg_event_capacity"].apply(
        lambda z: to_real(z, "avg_event_capacity", bundle.scaler)
    )
    if capacity < 1500:
        mask = cap_real < 2500
    elif capacity < 5000:
        mask = (cap_real >= 1500) & (cap_real < 8000)
    else:
        mask = cap_real >= 4000

    tier = peers[mask]
    if len(tier) < 20:
        tier = peers  # fall back to full genre if tier is too narrow

    out = pd.DataFrame({
        "revenue_dollars": tier["avg_gross_usd"].apply(
            lambda z: to_real(z, TARGET, bundle.scaler)
        ),
        "headliner": tier["headliner"].values,
        "capacity":  cap_real[tier.index].values,
    })
    return out.reset_index(drop=True)
