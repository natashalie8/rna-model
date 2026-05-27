"""Concert Revenue Predictor dashboard.

Run with:
    streamlit run streamlit_app.py
"""

from __future__ import annotations

from datetime import date, timedelta

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

import model_utils as mu

st.set_page_config(
    page_title="Concert Revenue Predictor",
    page_icon="🎫",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Theme + custom CSS
# ---------------------------------------------------------------------------
ACCENT = "#22d3ee"        # cyan
ACCENT_SOFT = "#0e7490"   # darker cyan
SUCCESS = "#4ade80"
WARN = "#fbbf24"
MUTED = "#94a3b8"
SURFACE = "#111827"
SURFACE_2 = "#1f2937"
BG = "#0b1220"

st.markdown(
    f"""
    <style>
        :root {{
            --accent: {ACCENT};
            --accent-soft: {ACCENT_SOFT};
            --surface: {SURFACE};
            --surface-2: {SURFACE_2};
            --muted: {MUTED};
        }}

        /* Page background */
        .stApp {{
            background: radial-gradient(1200px 600px at 10% -10%, rgba(34,211,238,0.08), transparent 60%),
                        radial-gradient(900px 500px at 100% 110%, rgba(34,211,238,0.05), transparent 60%),
                        {BG};
        }}

        /* Top header band */
        .app-header {{
            display: flex;
            align-items: center;
            justify-content: space-between;
            padding: 1.25rem 0 0.75rem 0;
            border-bottom: 1px solid rgba(148, 163, 184, 0.12);
            margin-bottom: 1.5rem;
        }}
        .app-title {{
            font-size: 1.5rem;
            font-weight: 700;
            letter-spacing: -0.02em;
            color: #f1f5f9;
            margin: 0;
        }}
        .app-subtitle {{
            color: var(--muted);
            font-size: 0.875rem;
            margin-top: 0.15rem;
        }}
        .app-badge {{
            background: rgba(34, 211, 238, 0.12);
            color: var(--accent);
            padding: 0.3rem 0.7rem;
            border-radius: 999px;
            font-size: 0.75rem;
            font-weight: 600;
            border: 1px solid rgba(34, 211, 238, 0.3);
            letter-spacing: 0.04em;
        }}

        /* Headline metric card */
        .metric-card {{
            background: linear-gradient(135deg, rgba(34,211,238,0.10), rgba(34,211,238,0.02));
            border: 1px solid rgba(34, 211, 238, 0.25);
            border-radius: 16px;
            padding: 2rem 2rem 1.75rem 2rem;
            margin-bottom: 1.25rem;
        }}
        .metric-label {{
            color: var(--muted);
            font-size: 0.8rem;
            text-transform: uppercase;
            letter-spacing: 0.1em;
            margin-bottom: 0.5rem;
        }}
        .metric-value {{
            font-size: 3.25rem;
            font-weight: 700;
            letter-spacing: -0.03em;
            color: #f8fafc;
            line-height: 1;
        }}
        .metric-range {{
            color: var(--muted);
            font-size: 1rem;
            margin-top: 0.75rem;
        }}
        .metric-range strong {{
            color: #e5e7eb;
        }}

        /* Secondary cards */
        .info-card {{
            background: var(--surface);
            border: 1px solid rgba(148, 163, 184, 0.08);
            border-radius: 12px;
            padding: 1.1rem 1.25rem;
            margin-bottom: 1rem;
        }}
        .info-card h4 {{
            margin: 0 0 0.45rem 0;
            font-size: 0.7rem;
            color: var(--muted);
            text-transform: uppercase;
            letter-spacing: 0.08em;
            font-weight: 600;
            white-space: nowrap;
        }}
        .info-card p {{
            margin: 0;
            color: #e5e7eb;
            font-size: 0.95rem;
            line-height: 1.45;
        }}
        .info-card.warn {{
            border-color: rgba(251, 191, 36, 0.35);
            background: rgba(251, 191, 36, 0.05);
        }}
        .info-card.warn p {{
            color: #fde68a;
        }}

        /* Sidebar styling */
        section[data-testid="stSidebar"] {{
            background: {SURFACE};
            border-right: 1px solid rgba(148, 163, 184, 0.08);
        }}
        section[data-testid="stSidebar"] .stMarkdown h2,
        section[data-testid="stSidebar"] .stMarkdown h3 {{
            color: #f1f5f9;
            font-weight: 600;
            letter-spacing: -0.01em;
        }}

        /* Inputs */
        .stTextInput input, .stNumberInput input, .stDateInput input {{
            background: {SURFACE_2} !important;
            border-radius: 8px !important;
            border: 1px solid rgba(148, 163, 184, 0.15) !important;
            color: #e5e7eb !important;
        }}
        .stSelectbox div[data-baseweb="select"] > div {{
            background: {SURFACE_2} !important;
            border-radius: 8px !important;
            border: 1px solid rgba(148, 163, 184, 0.15) !important;
        }}

        /* Primary button */
        .stButton > button {{
            width: 100%;
            background: linear-gradient(180deg, {ACCENT}, {ACCENT_SOFT});
            color: #0b1220 !important;
            border: 0;
            font-weight: 700;
            letter-spacing: 0.02em;
            padding: 0.65rem 1rem;
            border-radius: 10px;
            transition: transform 0.06s ease, box-shadow 0.2s ease;
        }}
        .stButton > button:hover {{
            transform: translateY(-1px);
            box-shadow: 0 8px 24px rgba(34, 211, 238, 0.25);
        }}

        /* Expander */
        .streamlit-expanderHeader {{
            background: var(--surface) !important;
            border-radius: 10px !important;
            border: 1px solid rgba(148, 163, 184, 0.08) !important;
        }}

        /* Hide default footer / hamburger noise for demo polish */
        footer {{ visibility: hidden; }}
        #MainMenu {{ visibility: hidden; }}
    </style>
    """,
    unsafe_allow_html=True,
)


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------
@st.cache_resource(show_spinner="Loading model and training data...")
def get_bundle():
    return mu.load_bundle(".")


bundle = get_bundle()


# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------
st.markdown(
    """
    <div class="app-header">
        <div>
            <div class="app-title">Concert Revenue Predictor</div>
            <div class="app-subtitle">Predict gross revenue for a configured concert, with honest uncertainty.</div>
        </div>
        <div class="app-badge">DEMO &nbsp;•&nbsp; ARound Entertainment Group</div>
    </div>
    """,
    unsafe_allow_html=True,
)


# ---------------------------------------------------------------------------
# Sidebar inputs
# ---------------------------------------------------------------------------
with st.sidebar:
    st.markdown("### Concert details")
    st.caption("Configure the inputs and click Predict.")

    artist_options = ["(custom — type below)"] + bundle.headliners
    artist_choice = st.selectbox(
        "Artist",
        options=artist_options,
        index=artist_options.index("Lord Huron") if "Lord Huron" in artist_options else 0,
        help="Pick from known artists in the training data, or choose '(custom)' and type your own.",
    )
    if artist_choice == "(custom — type below)":
        artist = st.text_input("Custom artist name", value="New Headliner")
    else:
        artist = artist_choice

    genre = st.selectbox(
        "Genre",
        options=list(mu.GENRES.keys()),
        index=0,
    )

    market = st.selectbox(
        "Market",
        options=list(mu.MARKETS.keys()),
        index=0,
    )

    capacity = st.number_input(
        "Venue capacity",
        min_value=100, max_value=80000,
        value=3000, step=100,
        help="Number of seats / capacity of the venue.",
    )

    price = st.number_input(
        "Avg ticket price ($)",
        min_value=5.0, max_value=2000.0,
        value=65.0, step=5.0,
        help="Average ticket price across all tiers.",
    )

    default_date = (date.today() + timedelta(days=180))
    event_date = st.date_input(
        "Event date",
        value=default_date,
        min_value=date(2020, 1, 1),
        max_value=date(2030, 12, 31),
    )

    st.markdown("")
    predict_clicked = st.button("Predict revenue", type="primary")


# ---------------------------------------------------------------------------
# Main panel
# ---------------------------------------------------------------------------
if not predict_clicked and "last_result" not in st.session_state:
    # First-run welcome
    st.markdown(
        """
        <div class="info-card">
            <h4>Welcome</h4>
            <p>Set the concert details in the sidebar and press <strong>Predict revenue</strong> to generate a prediction.
            Every prediction comes with a confidence range and a comparison against historical events with similar genre and venue size.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
else:
    if predict_clicked:
        result = mu.predict(
            bundle,
            artist=artist.strip() or "Unknown",
            genre=genre,
            capacity=float(capacity),
            price=float(price),
            event_date=event_date,
            market=market,
        )
        st.session_state["last_result"] = result
        st.session_state["last_inputs"] = {
            "artist": artist, "genre": genre, "market": market,
            "capacity": capacity, "price": price, "event_date": event_date,
        }

    result: mu.PredictionResult = st.session_state["last_result"]
    inputs = st.session_state["last_inputs"]

    # ---- Headline card -----------------------------------------------------
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-label">Predicted gross revenue</div>
            <div class="metric-value">{mu.fmt_dollars(result.pred_dollars)}</div>
            <div class="metric-range">
                Confidence range: <strong>{mu.fmt_dollars(result.lo_dollars)}</strong> — <strong>{mu.fmt_dollars(result.hi_dollars)}</strong>
                <span style="color: var(--muted); margin-left: 0.5rem;">(±1σ from cross-validation error)</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # ---- Three info cards: artist match / market / peer count --------------
    c1, c2, c3 = st.columns(3)
    with c1:
        if result.artist_known:
            st.markdown(
                f"""
                <div class="info-card">
                    <h4>Artist match</h4>
                    <p><strong>{inputs['artist']}</strong> found in training data — using their historical signals.</p>
                </div>
                """,
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                f"""
                <div class="info-card warn">
                    <h4>Artist not in dataset</h4>
                    <p><strong>{inputs['artist']}</strong> isn't in training. Filling in {len(result.imputed_features)} signals from <em>{inputs['genre']}</em> peers — prediction is less certain.</p>
                </div>
                """,
                unsafe_allow_html=True,
            )
    with c2:
        st.markdown(
            f"""
            <div class="info-card">
                <h4>Configuration</h4>
                <p><strong>{inputs['capacity']:,}</strong> seats &nbsp;·&nbsp; <strong>${inputs['price']:.0f}</strong> avg ticket<br>
                {inputs['market']} &nbsp;·&nbsp; {inputs['event_date'].strftime('%b %Y')}</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with c3:
        st.markdown(
            f"""
            <div class="info-card">
                <h4>Peer events compared</h4>
                <p><strong>{result.n_peer_events}</strong> historical events in <em>{inputs['genre']}</em> available for comparison.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # ---- Peer comparison chart --------------------------------------------
    peers = mu.peer_revenues(bundle, inputs["genre"], float(inputs["capacity"]))
    if not peers.empty:
        # Clip extreme outliers for display so the histogram is readable
        clip_hi = float(peers["revenue_dollars"].quantile(0.97))
        peers_display = peers[peers["revenue_dollars"] <= clip_hi]

        fig = go.Figure()
        fig.add_trace(
            go.Histogram(
                x=peers_display["revenue_dollars"],
                nbinsx=30,
                marker=dict(color=ACCENT_SOFT, line=dict(width=0)),
                opacity=0.85,
                name="Peer events",
                hovertemplate="Revenue: $%{x:,.0f}<br>Count: %{y}<extra></extra>",
            )
        )
        # Vertical line for the prediction
        fig.add_vline(
            x=result.pred_dollars,
            line=dict(color=ACCENT, width=3),
            annotation_text=f"  Prediction · {mu.fmt_dollars(result.pred_dollars)}",
            annotation_position="top right",
            annotation_font=dict(color=ACCENT, size=13),
        )
        # Confidence range as a shaded band
        fig.add_vrect(
            x0=max(0, result.lo_dollars),
            x1=min(clip_hi, result.hi_dollars),
            fillcolor=ACCENT,
            opacity=0.08,
            line_width=0,
        )

        # Quantile reference lines
        med = float(peers["revenue_dollars"].median())
        fig.add_vline(
            x=med,
            line=dict(color=MUTED, width=1, dash="dot"),
            annotation_text=f"  Peer median · {mu.fmt_dollars(med)}",
            annotation_position="bottom right",
            annotation_font=dict(color=MUTED, size=11),
        )

        fig.update_layout(
            title=dict(
                text=f"How this prediction compares to {len(peers)} similar concerts",
                font=dict(size=15, color="#e5e7eb"),
                x=0, xanchor="left",
            ),
            height=380,
            margin=dict(l=20, r=20, t=70, b=40),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#cbd5e1", family="sans-serif"),
            xaxis=dict(
                title="Actual revenue ($)",
                tickformat="$,.0f",
                gridcolor="rgba(148,163,184,0.08)",
                zeroline=False,
            ),
            yaxis=dict(
                title="Number of events",
                gridcolor="rgba(148,163,184,0.08)",
                zeroline=False,
            ),
            showlegend=False,
            bargap=0.05,
        )
        st.plotly_chart(fig, use_container_width=True)

        # Where the prediction lands in the peer distribution
        pct = float((peers["revenue_dollars"] <= result.pred_dollars).mean() * 100)
        pct_int = int(round(pct))
        # Ordinal suffix
        if 10 <= pct_int % 100 <= 20:
            suffix = "th"
        else:
            suffix = {1: "st", 2: "nd", 3: "rd"}.get(pct_int % 10, "th")
        if pct < 33:
            verdict = "<strong>low</strong> for this kind of event"
            color = WARN
        elif pct < 67:
            verdict = "<strong>typical</strong> for this kind of event"
            color = MUTED
        else:
            verdict = "<strong>high</strong> for this kind of event"
            color = SUCCESS
        st.markdown(
            f"""
            <div class="info-card">
                <h4>Verdict</h4>
                <p>The prediction sits at the <strong>{pct_int}{suffix} percentile</strong> of peer events —
                <span style="color: {color}">{verdict}</span>.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # ---- Imputed features detail ------------------------------------------
    if result.imputed_features:
        with st.expander(f"⚠ Imputed features ({len(result.imputed_features)})", expanded=False):
            st.markdown(
                "These features weren't available for this artist, so genre-median values were used. "
                "Predictions for unknown artists are less reliable than for artists in the training set."
            )
            st.code("\n".join(f"• {f}" for f in result.imputed_features))

    # ---- About expander ----------------------------------------------------
    with st.expander("About this tool — what it does, what it doesn't"):
        st.markdown(
            f"""
**What it does.** Predicts gross revenue for a configured concert based on ~1,800 historical events from
Pollstar data. A gradient-boosted regression model maps artist, venue capacity, ticket price, date, and
market into a predicted revenue figure. Cross-validated with GroupKFold by headliner (no artist
appears in both train and test), giving an honest CV R² of about 0.90.

**Confidence range.** The ±1σ band is the cross-validation RMSE
(~{mu.fmt_dollars(bundle.cv_rmse_z * bundle.scaler['avg_gross_usd']['std'])}). It reflects typical model
error, not a true predictive interval — truly unusual events can fall outside it.

**What this tool is not.**
1. **Not a recommendation engine.** It tells you *what events like this typically earned*, not *what
   this specific event would earn at a different price*. The data contains endogeneity: promoters set
   higher prices for popular shows, so high prices and high revenue appear together. The model
   reflects that correlation, which makes it unsuitable for prescriptive pricing decisions.
2. **No causal modeling.** Predictions describe historical patterns. Changing one input doesn't tell
   you what would actually happen in the real world.
3. **Limited training data.** ~1,800 events with heavy concentration of Country, Latin, and Pop/Rock
   acts. Predictions for unusual artists, markets, or venue types are less reliable.

**Treat predictions as informative starting points, not prescriptive answers.**
            """
        )

# ---------------------------------------------------------------------------
# Footer model-stats line
# ---------------------------------------------------------------------------
cv = bundle.cv_stats
st.markdown(
    f"""
    <div style="text-align: center; color: {MUTED}; font-size: 0.75rem; margin-top: 2rem; padding-top: 1rem; border-top: 1px solid rgba(148,163,184,0.08);">
        Model: XGBoost · GroupKFold CV (k=5, by headliner) ·
        R² {cv['mean_R2']:.3f} ±{cv['std_R2']:.3f} ·
        MAE {cv['mean_MAE']:.3f} ±{cv['std_MAE']:.3f} (z-scored units) ·
        Trained on 1,808 events
    </div>
    """,
    unsafe_allow_html=True,
)
