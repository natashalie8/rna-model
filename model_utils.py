"""Feature constructor and prediction utilities for the revenue dashboard.

The model expects ~78 z-scored features in a fixed order. The user only supplies six
inputs (artist, genre, capacity, price, date, market). This module fills in the rest:
artist-specific signals from training history when the artist is known, genre-peer
medians when they aren't.

New in v2: elasticity-based revenue curve (porting Ethan's within-artist OLS approach)
and in-sample performance diagnostics for the Model Performance tab.
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
ELASTICITIES_PATH = "genre_elasticities_v4.json"

TARGET = "avg_gross_usd"

# Features pulled from artist's historical events when available
ARTIST_FEATURES = [
    "gt_avg_13w", "gt_max_13w", "gt_std_13w", "gt_momentum_13w",
    "wiki_avg_views_30d", "historical_concerts", "days_since_last_album",
    "past_year_avg_tickets", "album_release_last_12m",
]

# Genre key -> one-hot column name
GENRES = {
    "Pop / Rock":         "genre_cleaned_pop_rock",
    "Country":            "genre_cleaned_country",
    "Latin":              "genre_cleaned_latin",
    "Dance / Electronic": "genre_cleaned_dance_electronic",
    "Rap / HipHop":       "genre_cleaned_rap_hiphop",
    "Other":              "genre_cleaned_other",
}

# Market key -> one-hot column name
MARKETS = {
    "New York":      "market_cleaned_new_york",
    "Los Angeles":   "market_cleaned_los_angeles",
    "Chicago":       "market_cleaned_chicago",
    "Washington DC": "market_cleaned_washington_dc_hagerstown",
    "Boston":        "market_cleaned_boston_manchester",
    "Other":         "market_cleaned_other",
}

# Headliner one-hots that exist in the model (everything else maps to "other")
HEADLINER_ONEHOTS = {
    "Aventura":        "headliner_cleaned_aventura",
    "Christian Nodal": "headliner_cleaned_christian_nodal",
    "David Foster":    "headliner_cleaned_david_foster",
    "Lord Huron":      "headliner_cleaned_lord_huron",
    "Randall King":    "headliner_cleaned_randall_king",
}

# Genre display name -> elasticities JSON key (Ethan's fitted names)
_GENRE_ELASTICITY_KEY = {
    "Pop / Rock":         "Pop / Rock",
    "Country":            "Country",
    "Latin":              "Latin",
    "Dance / Electronic": "Dance / Electronic",
    "Rap / HipHop":       "Rap / HipHop",
    "Other":              "Other",
}


def simplify_genre(g) -> str:
    """Map raw Pollstar genre string to one of the 6 display categories."""
    if pd.isna(g):
        return "Other"
    for k in ["Latin", "Pop / Rock", "Country", "Asian Pop"]:
        if k in str(g):
            return k
    if "Rap" in str(g) or "HipHop" in str(g):
        return "Rap / HipHop"
    if "Dance" in str(g) or "Electronic" in str(g):
        return "Dance / Electronic"
    return "Other"


# ---------------------------------------------------------------------------
# Core data structures
# ---------------------------------------------------------------------------

@dataclass
class Bundle:
    """Everything the dashboard needs at runtime, loaded once."""
    model: XGBRegressor
    features: list[str]
    df: pd.DataFrame
    scaler: dict
    cv_stats: dict
    medians: pd.Series
    headliners: list[str]
    artist_lookup: dict
    genre_lookup: dict
    genre_elasticities: dict

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


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def to_real(z: float, col: str, scaler: dict) -> float:
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
    if value >= 0:
        return f"${value:,.0f}"
    return f"-${abs(value):,.0f}"


def get_genre_elasticity(bundle: Bundle, genre: str) -> float | None:
    """Return the data-fitted within-artist price elasticity for this genre.

    NOTE: Most fitted betas are slightly positive due to selection bias in the
    training data (promoters charge more for popular shows). Use as reference
    only; the UI should let users override with an economically plausible value.
    """
    key = _GENRE_ELASTICITY_KEY.get(genre)
    by_genre = bundle.genre_elasticities.get("by_genre", {})
    if key and key in by_genre:
        return float(by_genre[key]["beta"])
    return None


# ---------------------------------------------------------------------------
# Bundle loader
# ---------------------------------------------------------------------------

def load_bundle(root: str | Path = ".") -> Bundle:
    """Load everything the dashboard needs. Cache with @st.cache_resource."""
    root = Path(root)

    model = XGBRegressor()
    model.load_model(str(root / MODEL_PATH))

    features = json.loads((root / FEATURES_PATH).read_text())
    scaler = json.loads((root / SCALER_PATH).read_text())
    cv_stats = json.loads((root / CV_STATS_PATH).read_text())
    genre_elasticities = json.loads((root / ELASTICITIES_PATH).read_text())

    df = pd.read_csv(root / DATA_PATH)
    bool_cols = df.select_dtypes(include="bool").columns
    df[bool_cols] = df[bool_cols].astype(int)

    X_template = df[features].copy()
    medians = X_template.median(numeric_only=True)

    artist_lookup: dict[str, pd.Series] = {}
    for name, group in df.groupby("headliner"):
        artist_lookup[name] = group[features].median(numeric_only=True)

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
        genre_elasticities=genre_elasticities,
    )


# ---------------------------------------------------------------------------
# Feature construction
# ---------------------------------------------------------------------------

def date_features(d: date) -> dict[str, float]:
    month = d.month
    dow = d.weekday()
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
    """Construct a 78-feature row from user inputs. Returns (row, imputed, artist_known)."""
    features = bundle.features
    row = bundle.medians.copy()
    imputed: list[str] = []

    # Genre one-hot
    genre_col = GENRES.get(genre, GENRES["Other"])
    for col in GENRES.values():
        if col in row.index:
            row[col] = 1 if col == genre_col else 0

    # Market one-hot + census features
    mkt_col = MARKETS.get(market, MARKETS["Other"])
    for col in MARKETS.values():
        if col in row.index:
            row[col] = 1 if col == mkt_col else 0
    mkt_rows = bundle.df[bundle.df[mkt_col] == 1]
    if len(mkt_rows) > 0:
        for col in ("market_population", "population", "median_income"):
            if col in row.index:
                row[col] = mkt_rows[col].median()

    # Headliner one-hot
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

    # Controllable inputs
    row["ticket_price_avg"] = to_scaled(price, "ticket_price_avg", bundle.scaler)
    row["avg_event_capacity"] = to_scaled(capacity, "avg_event_capacity", bundle.scaler)

    # Date features
    for col, val in date_features(event_date).items():
        if col in row.index:
            row[col] = val
    if "lockdown" in row.index:
        row["lockdown"] = 1 if event_date.year in (2020, 2021) else 0

    # Artist lookup or genre-peer imputation
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

    return row[features], imputed, artist_known


# ---------------------------------------------------------------------------
# Prediction
# ---------------------------------------------------------------------------

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
    lo_dollars = max(0.0, to_real(pred_z - rmse_z, TARGET, bundle.scaler))
    hi_dollars = to_real(pred_z + rmse_z, TARGET, bundle.scaler)

    genre_col = GENRES.get(genre, GENRES["Other"])
    n_peers = int((bundle.df[genre_col] == 1).sum()) \
        if genre_col in bundle.df.columns else 0

    return PredictionResult(
        pred_z=pred_z,
        pred_dollars=pred_dollars,
        lo_dollars=lo_dollars,
        hi_dollars=hi_dollars,
        imputed_features=imputed,
        artist_known=artist_known,
        n_peer_events=n_peers,
    )


# ---------------------------------------------------------------------------
# Peer revenue comparison
# ---------------------------------------------------------------------------

def peer_revenues(bundle: Bundle, genre: str, capacity: float) -> pd.DataFrame:
    """Actual revenues (in $) for similar-genre + similar-capacity events."""
    genre_col = GENRES.get(genre, GENRES["Other"])
    if genre_col not in bundle.df.columns:
        return pd.DataFrame()

    peers = bundle.df[bundle.df[genre_col] == 1].copy()
    if peers.empty:
        return pd.DataFrame()

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
        tier = peers

    return pd.DataFrame({
        "revenue_dollars": tier["avg_gross_usd"].apply(
            lambda z: to_real(z, TARGET, bundle.scaler)
        ),
        "headliner": tier["headliner"].values,
        "capacity":  cap_real[tier.index].values,
    }).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Elasticity-based revenue curve
# ---------------------------------------------------------------------------

def revenue_curve_with_elasticity(
    bundle: Bundle,
    artist: str,
    genre: str,
    capacity: float,
    event_date: date,
    market: str,
    elasticity: float,
    n_points: int = 60,
) -> tuple[pd.DataFrame, float]:
    """Price → revenue curve using XGBoost at a reference price + elasticity adjustment.

    Method (mirrors Ethan's approach adapted to our direct-revenue model):
    1. Anchor at the reference price (artist historical median, or genre median).
    2. Predict revenue at that reference price using our model.
    3. Back out implied tickets: base_tickets = revenue / ref_price.
    4. Sweep candidate prices [ref*0.3 .. ref*3.0]; apply elasticity multiplier to demand:
           adjusted_tickets = base_tickets * max(1 + elasticity * pct_change, 0.05)
       Capped at venue capacity.
    5. adjusted_revenue = candidate_price * adjusted_tickets.

    When elasticity=0 revenue grows linearly with price (no demand response) — this
    reveals the endogeneity assumption baked into the raw model. Negative elasticity
    creates a concave curve with a well-defined optimal price.

    Returns (curve_df, ref_price_dollars).
    """
    # Reference price: artist historical median or genre median
    if artist in bundle.artist_lookup:
        ref_z = float(bundle.artist_lookup[artist]["ticket_price_avg"])
    else:
        genre_col = GENRES.get(genre, GENRES["Other"])
        genre_rows = bundle.df[bundle.df.get(genre_col, pd.Series(dtype=float)) == 1]["ticket_price_avg"] \
            if genre_col in bundle.df.columns else pd.Series(dtype=float)
        ref_z = float(genre_rows.median()) if len(genre_rows) > 0 else 0.0

    ref_price = float(np.clip(to_real(ref_z, "ticket_price_avg", bundle.scaler), 10.0, 800.0))

    # Predict revenue at reference price
    row_ref, _, _ = build_feature_row(
        bundle, artist, genre, capacity, ref_price, event_date, market
    )
    X_ref = pd.DataFrame([row_ref.values], columns=bundle.features)
    base_rev_z = float(bundle.model.predict(X_ref)[0])
    base_rev = max(to_real(base_rev_z, TARGET, bundle.scaler), 0.0)

    # Back out implied tickets at the reference price
    base_tickets = base_rev / ref_price if ref_price > 0 else 0.0

    # Sweep candidate prices
    multipliers = np.linspace(0.3, 3.0, n_points)
    candidate_prices = ref_price * multipliers

    rows = []
    for p in candidate_prices:
        pct_change = (p - ref_price) / ref_price
        demand_mult = max(1.0 + elasticity * pct_change, 0.05)
        tickets = min(base_tickets * demand_mult, capacity)
        fill_pct = (tickets / capacity * 100) if capacity > 0 else 0.0
        revenue = p * tickets
        rows.append({
            "price": p,
            "tickets": tickets,
            "fill_pct": fill_pct,
            "revenue": revenue,
        })

    return pd.DataFrame(rows), ref_price


# ---------------------------------------------------------------------------
# In-sample performance (for Model Performance tab)
# ---------------------------------------------------------------------------

def compute_in_sample_predictions(bundle: Bundle) -> pd.DataFrame:
    """Predict revenue for every training event using its actual features.

    This is in-sample (optimistic vs true held-out performance). Use CV stats
    for real out-of-sample estimates. Useful for residual analysis and
    identifying where the model struggles.
    """
    df = bundle.df
    X = df[bundle.features].copy()
    bool_cols = X.select_dtypes(include="bool").columns
    X[bool_cols] = X[bool_cols].astype(int)
    non_num = X.select_dtypes(exclude="number").columns.tolist()
    if non_num:
        X = X.drop(columns=non_num)

    pred_z = bundle.model.predict(X)
    scaler_rev = bundle.scaler[TARGET]
    pred_dollars = pred_z * scaler_rev["std"] + scaler_rev["mean"]
    actual_dollars = df[TARGET].values * scaler_rev["std"] + scaler_rev["mean"]

    error = pred_dollars - actual_dollars
    pct_error = error / np.clip(np.abs(actual_dollars), 1, None) * 100

    return pd.DataFrame({
        "headliner":    df["headliner"].values,
        "genre":        df["genre"].apply(simplify_genre).values,
        "actual_rev":   actual_dollars,
        "pred_rev":     pred_dollars,
        "error":        error,
        "pct_error":    pct_error,
        "abs_error":    np.abs(error),
    })
