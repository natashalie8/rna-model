"""Concert Revenue Predictor dashboard — v2.

New in v2:
  - Elasticity-based price sensitivity analysis (fixes the monotonic revenue issue)
  - Revenue curve with optimal price recommendation
  - Elasticity sweep table
  - Model Performance tab: in-sample diagnostics, R² by genre, feature importances,
    residual analysis, worst predictions
"""

from __future__ import annotations

from datetime import date, timedelta

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import streamlit as st

import model_utils as mu

st.set_page_config(
    page_title="Concert Revenue Predictor",
    page_icon="🎫",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Theme
# ---------------------------------------------------------------------------
ACCENT   = "#22d3ee"
ACCENT_S = "#0e7490"
SUCCESS  = "#4ade80"
WARN     = "#fbbf24"
MUTED    = "#94a3b8"
SURFACE  = "#111827"
SURFACE2 = "#1f2937"
BG       = "#0b1220"

st.markdown(f"""
<style>
    .stApp {{
        background: radial-gradient(1200px 600px at 10% -10%, rgba(34,211,238,0.08), transparent 60%),
                    radial-gradient(900px 500px at 100% 110%, rgba(34,211,238,0.05), transparent 60%),
                    {BG};
    }}
    .app-header {{
        display: flex; align-items: center; justify-content: space-between;
        padding: 1.25rem 0 0.75rem 0;
        border-bottom: 1px solid rgba(148,163,184,0.12);
        margin-bottom: 1.25rem;
    }}
    .app-title {{ font-size:1.5rem; font-weight:700; letter-spacing:-0.02em; color:#f1f5f9; margin:0; }}
    .app-subtitle {{ color:{MUTED}; font-size:0.875rem; margin-top:0.15rem; }}
    .app-badge {{
        background: rgba(34,211,238,0.12); color:{ACCENT};
        padding:0.3rem 0.7rem; border-radius:999px; font-size:0.75rem;
        font-weight:600; border:1px solid rgba(34,211,238,0.3); letter-spacing:0.04em;
    }}
    .metric-card {{
        background: linear-gradient(135deg, rgba(34,211,238,0.10), rgba(34,211,238,0.02));
        border: 1px solid rgba(34,211,238,0.25); border-radius:16px;
        padding: 2rem 2rem 1.75rem 2rem; margin-bottom:1.25rem;
    }}
    .metric-label {{ color:{MUTED}; font-size:0.8rem; text-transform:uppercase; letter-spacing:0.1em; margin-bottom:0.5rem; }}
    .metric-value {{ font-size:3.25rem; font-weight:700; letter-spacing:-0.03em; color:#f8fafc; line-height:1; }}
    .metric-range {{ color:{MUTED}; font-size:1rem; margin-top:0.75rem; }}
    .metric-range strong {{ color:#e5e7eb; }}
    .rec-card {{
        background: linear-gradient(135deg, rgba(74,222,128,0.10), rgba(74,222,128,0.02));
        border: 1px solid rgba(74,222,128,0.3); border-radius:16px;
        padding: 1.5rem 2rem; margin-bottom:1.25rem;
    }}
    .rec-label {{ color:{SUCCESS}; font-size:0.75rem; text-transform:uppercase; letter-spacing:0.1em; margin-bottom:0.4rem; font-weight:600; }}
    .rec-value {{ font-size:2.5rem; font-weight:700; letter-spacing:-0.03em; color:#f8fafc; line-height:1; }}
    .rec-sub {{ color:{MUTED}; font-size:0.9rem; margin-top:0.5rem; }}
    .info-card {{
        background:{SURFACE}; border:1px solid rgba(148,163,184,0.08);
        border-radius:12px; padding:1.1rem 1.25rem; margin-bottom:1rem;
    }}
    .info-card h4 {{
        margin:0 0 0.45rem 0; font-size:0.7rem; color:{MUTED};
        text-transform:uppercase; letter-spacing:0.08em; font-weight:600; white-space:nowrap;
    }}
    .info-card p {{ margin:0; color:#e5e7eb; font-size:0.95rem; line-height:1.45; }}
    .info-card.warn {{ border-color:rgba(251,191,36,0.35); background:rgba(251,191,36,0.05); }}
    .info-card.warn p {{ color:#fde68a; }}
    .section-title {{
        font-size:1rem; font-weight:600; color:#e5e7eb;
        margin:1.5rem 0 0.75rem 0; letter-spacing:-0.01em;
    }}
    section[data-testid="stSidebar"] {{
        background:{SURFACE}; border-right:1px solid rgba(148,163,184,0.08);
    }}
    .stButton > button {{
        width:100%;
        background: linear-gradient(180deg,{ACCENT},{ACCENT_S});
        color:#0b1220 !important; border:0; font-weight:700;
        letter-spacing:0.02em; padding:0.65rem 1rem; border-radius:10px;
        transition:transform 0.06s ease, box-shadow 0.2s ease;
    }}
    .stButton > button:hover {{
        transform:translateY(-1px);
        box-shadow:0 8px 24px rgba(34,211,238,0.25);
    }}
    footer {{ visibility:hidden; }}
    #MainMenu {{ visibility:hidden; }}
</style>
""", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Load bundle (cached — keyed on model file mtime so retraining auto-refreshes)
# ---------------------------------------------------------------------------
import os as _os
_model_mtime = _os.path.getmtime("revenue_predictor.json") if _os.path.exists("revenue_predictor.json") else 0

@st.cache_resource(show_spinner="Loading model and training data…")
def get_bundle(_mtime=_model_mtime):
    return mu.load_bundle(".")

@st.cache_data
def get_in_sample_preds(_mtime=_model_mtime):
    b = get_bundle(_mtime)
    return mu.compute_in_sample_predictions(b)

bundle = get_bundle(_model_mtime)

# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------
st.markdown("""
<div class="app-header">
    <div>
        <div class="app-title">Concert Revenue Predictor</div>
        <div class="app-subtitle">Predict gross revenue · Price sensitivity · Model diagnostics</div>
    </div>
    <div class="app-badge">DEMO &nbsp;•&nbsp; ARound Entertainment Group</div>
</div>
""", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
with st.sidebar:
    st.markdown("### Concert details")
    st.caption("Configure the inputs and click Predict.")

    artist_options = ["(custom — type below)"] + bundle.headliners
    artist_choice = st.selectbox(
        "Artist",
        options=artist_options,
        index=artist_options.index("Lord Huron") if "Lord Huron" in artist_options else 0,
        help="Pick from known artists in the training data, or type your own.",
    )
    if artist_choice == "(custom — type below)":
        artist = st.text_input("Custom artist name", value="New Headliner")
    else:
        artist = artist_choice

    genre = st.selectbox("Genre", options=list(mu.GENRES.keys()), index=0)
    market = st.selectbox("Market", options=list(mu.MARKETS.keys()), index=0)
    capacity = st.number_input("Venue capacity", min_value=100, max_value=80000, value=3000, step=100)
    price = st.number_input("Avg ticket price ($)", min_value=5.0, max_value=2000.0, value=65.0, step=5.0)
    default_date = date.today() + timedelta(days=180)
    event_date = st.date_input("Event date", value=default_date,
                               min_value=date(2020,1,1), max_value=date(2030,12,31))

    st.markdown("")
    predict_clicked = st.button("Predict revenue", type="primary")

    # -- Elasticity -----------------------------------------------------------
    st.markdown("---")
    st.markdown("### Price sensitivity")

    fitted_beta = mu.get_genre_elasticity(bundle, genre)
    fitted_note = f"Data-fitted ({genre}): **{fitted_beta:+.3f}**" if fitted_beta is not None else ""
    if fitted_note:
        st.caption(
            fitted_note + "  \n⚠ Fitted betas reflect selection bias (promoters charge "
            "more for popular shows). Use as reference only — adjust the slider for a "
            "realistic demand response."
        )

    elasticity = st.slider(
        "Price elasticity of demand",
        min_value=-2.0, max_value=0.5, value=-0.5, step=0.05,
        help="How much demand changes with price. "
             "Literature range for live events: −0.3 to −0.7. "
             "0 = no demand response (shows raw model endogeneity). "
             "Negative = higher price → lower attendance.",
    )
    if elasticity == 0.0:
        st.warning("Elasticity = 0: revenue grows linearly with price. "
                   "Slide to ~−0.5 for a realistic demand curve.", icon="⚠")


# ---------------------------------------------------------------------------
# Tabs
# ---------------------------------------------------------------------------
tab_predict, tab_perf = st.tabs(["🎫  Predict Revenue", "📊  Model Performance"])


# ===========================================================================
# TAB 1 — Predict
# ===========================================================================
with tab_predict:

    if not predict_clicked and "last_result" not in st.session_state:
        st.markdown("""
        <div class="info-card">
            <h4>Welcome</h4>
            <p>Set the concert details in the sidebar and press <strong>Predict revenue</strong>.
            Every prediction includes a confidence range, a peer comparison, and a
            price-sensitivity curve that shows the revenue-maximising ticket price.</p>
        </div>
        """, unsafe_allow_html=True)

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
            curve_df, ref_price = mu.revenue_curve_with_elasticity(
                bundle,
                artist=artist.strip() or "Unknown",
                genre=genre,
                capacity=float(capacity),
                event_date=event_date,
                market=market,
                elasticity=elasticity,
            )
            st.session_state.update({
                "last_result": result,
                "last_curve": curve_df,
                "last_ref_price": ref_price,
                "last_inputs": {
                    "artist": artist, "genre": genre, "market": market,
                    "capacity": capacity, "price": price,
                    "event_date": event_date, "elasticity": elasticity,
                },
            })

        result: mu.PredictionResult = st.session_state["last_result"]
        curve_df: pd.DataFrame      = st.session_state["last_curve"]
        ref_price: float            = st.session_state["last_ref_price"]
        inputs                      = st.session_state["last_inputs"]

        # ---- Headline metric -------------------------------------------------
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Predicted gross revenue (model estimate)</div>
            <div class="metric-value">{mu.fmt_dollars(result.pred_dollars)}</div>
            <div class="metric-range">
                Confidence range: <strong>{mu.fmt_dollars(result.lo_dollars)}</strong>
                — <strong>{mu.fmt_dollars(result.hi_dollars)}</strong>
                <span style="color:{MUTED}; margin-left:0.5rem;">(±1σ cross-validation error)</span>
            </div>
        </div>
        """, unsafe_allow_html=True)

        # ---- Three info cards ------------------------------------------------
        c1, c2, c3 = st.columns(3)
        with c1:
            if result.artist_known:
                st.markdown(f"""
                <div class="info-card">
                    <h4>Artist match</h4>
                    <p><strong>{inputs['artist']}</strong> found in training data —
                    using historical signals.</p>
                </div>""", unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="info-card warn">
                    <h4>Artist not in dataset</h4>
                    <p><strong>{inputs['artist']}</strong> isn't in training.
                    {len(result.imputed_features)} signals imputed from
                    <em>{inputs['genre']}</em> peers.</p>
                </div>""", unsafe_allow_html=True)
        with c2:
            st.markdown(f"""
            <div class="info-card">
                <h4>Configuration</h4>
                <p><strong>{inputs['capacity']:,}</strong> seats &nbsp;·&nbsp;
                <strong>${inputs['price']:.0f}</strong> avg ticket<br>
                {inputs['market']} &nbsp;·&nbsp;
                {inputs['event_date'].strftime('%b %Y')}</p>
            </div>""", unsafe_allow_html=True)
        with c3:
            st.markdown(f"""
            <div class="info-card">
                <h4>Peer events compared</h4>
                <p><strong>{result.n_peer_events}</strong> historical events in
                <em>{inputs['genre']}</em>.</p>
            </div>""", unsafe_allow_html=True)

        # ---- Peer comparison chart -------------------------------------------
        peers = mu.peer_revenues(bundle, inputs["genre"], float(inputs["capacity"]))
        if not peers.empty:
            clip_hi = float(peers["revenue_dollars"].quantile(0.97))
            peers_d = peers[peers["revenue_dollars"] <= clip_hi]

            fig_peer = go.Figure()
            fig_peer.add_trace(go.Histogram(
                x=peers_d["revenue_dollars"], nbinsx=30,
                marker=dict(color=ACCENT_S, line=dict(width=0)),
                opacity=0.85, name="Peer events",
                hovertemplate="Revenue: $%{x:,.0f}<br>Count: %{y}<extra></extra>",
            ))
            fig_peer.add_vline(x=result.pred_dollars, line=dict(color=ACCENT, width=3),
                               annotation_text=f"  Prediction · {mu.fmt_dollars(result.pred_dollars)}",
                               annotation_position="top right",
                               annotation_font=dict(color=ACCENT, size=13))
            fig_peer.add_vrect(x0=max(0, result.lo_dollars),
                               x1=min(clip_hi, result.hi_dollars),
                               fillcolor=ACCENT, opacity=0.08, line_width=0)
            med = float(peers["revenue_dollars"].median())
            fig_peer.add_vline(x=med, line=dict(color=MUTED, width=1, dash="dot"),
                               annotation_text=f"  Peer median · {mu.fmt_dollars(med)}",
                               annotation_position="bottom right",
                               annotation_font=dict(color=MUTED, size=11))
            fig_peer.update_layout(
                title=dict(text=f"How this prediction compares to {len(peers)} similar concerts",
                           font=dict(size=15, color="#e5e7eb"), x=0, xanchor="left"),
                height=340, margin=dict(l=20,r=20,t=60,b=40),
                paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                font=dict(color="#cbd5e1"),
                xaxis=dict(title="Actual revenue ($)", tickformat="$,.0f",
                           gridcolor="rgba(148,163,184,0.08)", zeroline=False),
                yaxis=dict(title="Number of events",
                           gridcolor="rgba(148,163,184,0.08)", zeroline=False),
                showlegend=False, bargap=0.05,
            )
            st.plotly_chart(fig_peer, use_container_width=True)

            pct_val = float((peers["revenue_dollars"] <= result.pred_dollars).mean() * 100)
            pct_int = int(round(pct_val))
            suffix = "th" if 10 <= pct_int % 100 <= 20 else {1:"st",2:"nd",3:"rd"}.get(pct_int % 10, "th")
            if pct_val < 33:
                verdict_txt, verdict_col = "<strong>low</strong> for this kind of event", WARN
            elif pct_val < 67:
                verdict_txt, verdict_col = "<strong>typical</strong> for this kind of event", MUTED
            else:
                verdict_txt, verdict_col = "<strong>high</strong> for this kind of event", SUCCESS
            st.markdown(f"""
            <div class="info-card">
                <h4>Verdict</h4>
                <p>The prediction sits at the <strong>{pct_int}{suffix} percentile</strong>
                of peer events — <span style="color:{verdict_col}">{verdict_txt}</span>.</p>
            </div>""", unsafe_allow_html=True)

        # ====================================================================
        # PRICE SENSITIVITY SECTION
        # ====================================================================
        st.markdown(f'<div class="section-title">Price sensitivity analysis</div>', unsafe_allow_html=True)
        st.caption(
            f"Anchored at the reference price **{mu.fmt_dollars(ref_price)}** "
            f"(artist/genre historical median). The model predicts revenue there; "
            f"elasticity **{inputs['elasticity']:+.2f}** adjusts demand as price moves away."
        )

        if not curve_df.empty:
            best_idx = int(curve_df["revenue"].idxmax())
            best_price = float(curve_df.loc[best_idx, "price"])
            best_rev   = float(curve_df.loc[best_idx, "revenue"])
            best_fill  = float(curve_df.loc[best_idx, "fill_pct"])

            # Recommended price card
            col_rec, col_gap = st.columns([2, 1])
            with col_rec:
                st.markdown(f"""
                <div class="rec-card">
                    <div class="rec-label">Recommended price (elasticity-adjusted)</div>
                    <div class="rec-value">{mu.fmt_dollars(best_price)}</div>
                    <div class="rec-sub">
                        Projected revenue: <strong>{mu.fmt_dollars(best_rev)}</strong>
                        &nbsp;·&nbsp; Projected fill: <strong>{best_fill:.0f}%</strong>
                        &nbsp;·&nbsp; Capacity: <strong>{int(inputs['capacity']):,} seats</strong>
                    </div>
                </div>""", unsafe_allow_html=True)

            # Revenue curve chart
            fig_curve = go.Figure()

            # Fill between 0 and curve
            fig_curve.add_trace(go.Scatter(
                x=curve_df["price"], y=curve_df["revenue"],
                fill="tozeroy", fillcolor="rgba(34,211,238,0.06)",
                line=dict(color=ACCENT, width=2.5),
                name="Revenue", mode="lines",
                hovertemplate="Price: $%{x:,.0f}<br>Revenue: $%{y:,.0f}<extra></extra>",
            ))

            # Recommended price line
            fig_curve.add_vline(
                x=best_price, line=dict(color=SUCCESS, width=2, dash="dash"),
                annotation_text=f"  Recommended · {mu.fmt_dollars(best_price)}",
                annotation_position="top right",
                annotation_font=dict(color=SUCCESS, size=12),
            )
            # Reference price line
            fig_curve.add_vline(
                x=ref_price, line=dict(color=MUTED, width=1.5, dash="dot"),
                annotation_text=f"  Reference · {mu.fmt_dollars(ref_price)}",
                annotation_position="bottom right",
                annotation_font=dict(color=MUTED, size=11),
            )
            # User's chosen price
            user_price = float(inputs["price"])
            user_rev = float(np.interp(user_price, curve_df["price"], curve_df["revenue"]))
            fig_curve.add_vline(
                x=user_price, line=dict(color=WARN, width=1.5, dash="dash"),
                annotation_text=f"  Your price · {mu.fmt_dollars(user_price)}",
                annotation_position="top left",
                annotation_font=dict(color=WARN, size=11),
            )

            fig_curve.update_layout(
                title=dict(
                    text=f"Predicted revenue vs ticket price  (elasticity = {inputs['elasticity']:+.2f})",
                    font=dict(size=14, color="#e5e7eb"), x=0, xanchor="left",
                ),
                height=360, margin=dict(l=20,r=20,t=60,b=40),
                paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                font=dict(color="#cbd5e1"),
                xaxis=dict(title="Ticket price ($)", tickformat="$,.0f",
                           gridcolor="rgba(148,163,184,0.08)", zeroline=False),
                yaxis=dict(title="Projected revenue ($)", tickformat="$,.0f",
                           gridcolor="rgba(148,163,184,0.08)", zeroline=False),
                showlegend=False,
            )
            st.plotly_chart(fig_curve, use_container_width=True)

            # Price gap note
            gap_pct = (best_price - user_price) / user_price * 100 if user_price > 0 else 0
            if abs(gap_pct) > 15:
                direction = "lower" if gap_pct > 0 else "higher"
                st.markdown(f"""
                <div class="info-card warn">
                    <h4>Pricing gap</h4>
                    <p>Your chosen price ({mu.fmt_dollars(user_price)}) is
                    <strong>{abs(gap_pct):.0f}% {direction}</strong> than the
                    elasticity-adjusted optimum ({mu.fmt_dollars(best_price)}).
                    At the recommended price, projected revenue is
                    <strong>{mu.fmt_dollars(best_rev)}</strong>.</p>
                </div>""", unsafe_allow_html=True)

            # Elasticity sweep table
            with st.expander("How does the recommendation change with elasticity?"):
                st.caption("Revenue-maximising price across different elasticity assumptions.")
                sweep_rows = []
                for e_val in [0.0, -0.2, -0.3, -0.5, -0.7, -1.0, -1.5, -2.0]:
                    c_df, rp = mu.revenue_curve_with_elasticity(
                        bundle, artist=inputs["artist"], genre=inputs["genre"],
                        capacity=float(inputs["capacity"]), event_date=inputs["event_date"],
                        market=inputs["market"], elasticity=e_val,
                    )
                    bi = int(c_df["revenue"].idxmax())
                    sweep_rows.append({
                        "Elasticity": f"{e_val:+.1f}",
                        "Rec. price":   mu.fmt_dollars(c_df.loc[bi, "price"]),
                        "Rec. revenue": mu.fmt_dollars(c_df.loc[bi, "revenue"]),
                        "Rec. fill %":  f"{c_df.loc[bi, 'fill_pct']:.0f}%",
                    })
                st.dataframe(pd.DataFrame(sweep_rows), hide_index=True, use_container_width=True)

        # ---- Imputed features -----------------------------------------------
        if result.imputed_features:
            with st.expander(f"⚠ Imputed features ({len(result.imputed_features)})", expanded=False):
                st.markdown(
                    "These features weren't available for this artist — genre-median values were used. "
                    "Predictions for unknown artists are less reliable."
                )
                st.code("\n".join(f"• {f}" for f in result.imputed_features))

        # ---- About ----------------------------------------------------------
        with st.expander("About this tool — what it does, what it doesn't"):
            st.markdown(f"""
**What it does.** Predicts gross revenue for a configured concert based on ~1,800 historical events
from Pollstar data. A gradient-boosted regression model maps artist, venue capacity, ticket price,
date, and market into a predicted revenue figure. Cross-validated with GroupKFold by headliner
(no artist appears in both train and test), giving CV R² ≈ 0.90.

**Price sensitivity.** The revenue curve uses a within-artist elasticity adjustment: the model
predicts at a reference price, infers implied ticket demand, then applies the elasticity multiplier
as price moves away. This creates a concave curve with a well-defined optimum — addressing the
raw model's tendency to always predict higher revenue at higher prices.

**Elasticity betas.** Fitted via within-artist demeaned OLS (log fill rate on log price) following
Ethan Chuang's `fit_elasticity_v4.py`. Most fitted betas are slightly positive due to selection
bias. The dashboard lets you override with an economically plausible value.

**What this tool is not.**
1. **Not a causal model.** It captures historical correlations, not what would happen if you
   changed prices for a specific event.
2. **Not a recommendation engine for unknown artists.** Without historical signals, artist-specific
   features are imputed from genre medians.
3. **Limited training data.** ~1,800 events, heavy Country / Latin / Pop-Rock concentration.
   Unusual artists, markets, or venues are less reliable.

*Treat predictions as informative starting points, not prescriptive answers.*
            """)


# ===========================================================================
# TAB 2 — Model Performance
# ===========================================================================
with tab_perf:
    st.markdown(
        '<div class="section-title">In-sample model diagnostics</div>',
        unsafe_allow_html=True,
    )
    st.caption(
        "Predictions on the **training set** — optimistic vs true out-of-sample. "
        "For held-out estimates use the CV stats in the footer. "
        "Useful for understanding where the model struggles."
    )

    preds = get_in_sample_preds(_model_mtime)

    # ---- Summary metrics ---------------------------------------------------
    r2  = r2_score(preds["actual_rev"], preds["pred_rev"])
    mae = mean_absolute_error(preds["actual_rev"], preds["pred_rev"])
    rmse = float(np.sqrt(mean_squared_error(preds["actual_rev"], preds["pred_rev"])))
    mape = float(np.mean(np.abs(preds["pct_error"]))
                 if "pct_error" in preds.columns else 0)
    cv = bundle.cv_stats

    m1, m2, m3, m4, m5, m6 = st.columns(6)
    m1.metric("In-sample R²",    f"{r2:.3f}")
    m2.metric("In-sample MAE",   mu.fmt_dollars(mae))
    m3.metric("In-sample RMSE",  mu.fmt_dollars(rmse))
    m4.metric("CV R² (honest)",  f"{cv['mean_R2']:.3f} ±{cv['std_R2']:.3f}")
    m5.metric("CV MAE (honest)", f"{cv['mean_MAE']:.3f} ±{cv['std_MAE']:.3f} z")
    m6.metric("In-sample MAPE",  f"{mape:.0f}%")

    st.markdown("---")

    # ---- Pred vs actual scatter --------------------------------------------
    col_l, col_r = st.columns(2)
    with col_l:
        ok = (preds["actual_rev"] > 0) & (preds["pred_rev"] > 0)
        lo_ax = float(np.log10(max(preds.loc[ok, "actual_rev"].min(), 1)))
        hi_ax = float(np.log10(preds.loc[ok, "actual_rev"].max()))
        diag  = [10 ** lo_ax, 10 ** hi_ax]

        fig_sc = go.Figure()
        fig_sc.add_trace(go.Scatter(
            x=preds.loc[ok, "actual_rev"], y=preds.loc[ok, "pred_rev"],
            mode="markers",
            marker=dict(color=ACCENT_S, size=5, opacity=0.5, line=dict(width=0)),
            hovertemplate="Actual: $%{x:,.0f}<br>Predicted: $%{y:,.0f}<extra></extra>",
            name="Events",
        ))
        fig_sc.add_trace(go.Scatter(
            x=diag, y=diag, mode="lines",
            line=dict(color="rgba(248,250,252,0.3)", dash="dash", width=1.5),
            name="Perfect", hoverinfo="skip",
        ))
        fig_sc.update_layout(
            title=dict(text="Predicted vs Actual revenue (log scale)",
                       font=dict(size=13, color="#e5e7eb"), x=0, xanchor="left"),
            height=380, margin=dict(l=20,r=20,t=50,b=40),
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#cbd5e1"),
            xaxis=dict(title="Actual revenue ($)", type="log", tickformat="$,.0f",
                       gridcolor="rgba(148,163,184,0.08)", zeroline=False),
            yaxis=dict(title="Predicted revenue ($)", type="log", tickformat="$,.0f",
                       gridcolor="rgba(148,163,184,0.08)", zeroline=False),
            showlegend=False,
        )
        st.plotly_chart(fig_sc, use_container_width=True)

    with col_r:
        residuals = preds["error"]
        clip_lo = float(residuals.quantile(0.01))
        clip_hi_r = float(residuals.quantile(0.99))
        res_clipped = residuals.clip(clip_lo, clip_hi_r)

        fig_res = go.Figure()
        fig_res.add_trace(go.Histogram(
            x=res_clipped, nbinsx=40,
            marker=dict(color=ACCENT_S, line=dict(width=0)),
            opacity=0.85,
            hovertemplate="Error: $%{x:,.0f}<br>Count: %{y}<extra></extra>",
        ))
        fig_res.add_vline(x=0, line=dict(color="rgba(248,250,252,0.4)", width=1.5, dash="dash"))
        fig_res.add_vline(
            x=float(residuals.mean()),
            line=dict(color=WARN, width=1.5),
            annotation_text=f"  mean {mu.fmt_dollars(residuals.mean())}",
            annotation_font=dict(color=WARN, size=11),
        )
        fig_res.update_layout(
            title=dict(text="Residual distribution (clipped 1–99th pctile)",
                       font=dict(size=13, color="#e5e7eb"), x=0, xanchor="left"),
            height=380, margin=dict(l=20,r=20,t=50,b=40),
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#cbd5e1"),
            xaxis=dict(title="Predicted − Actual ($)", tickformat="$,.0f",
                       gridcolor="rgba(148,163,184,0.08)", zeroline=False),
            yaxis=dict(title="# events", gridcolor="rgba(148,163,184,0.08)", zeroline=False),
            showlegend=False,
        )
        st.plotly_chart(fig_res, use_container_width=True)

    # ---- R² by genre -------------------------------------------------------
    st.markdown("---")
    st.markdown('<div class="section-title">Performance by genre</div>', unsafe_allow_html=True)

    genre_rows = []
    for g, sub in preds.groupby("genre"):
        if len(sub) < 10:
            continue
        genre_rows.append({
            "Genre":      g,
            "Events":     len(sub),
            "R²":         round(r2_score(sub["actual_rev"], sub["pred_rev"]), 3),
            "MAE":        mu.fmt_dollars(mean_absolute_error(sub["actual_rev"], sub["pred_rev"])),
            "Mean actual": mu.fmt_dollars(sub["actual_rev"].mean()),
            "Mean pred":   mu.fmt_dollars(sub["pred_rev"].mean()),
        })
    gp_df = pd.DataFrame(genre_rows).sort_values("Events", ascending=False)

    gcol_l, gcol_r = st.columns(2)
    with gcol_l:
        fig_g = go.Figure(go.Bar(
            x=gp_df["R²"], y=gp_df["Genre"], orientation="h",
            marker=dict(color=ACCENT_S, line=dict(width=0)),
            hovertemplate="%{y}: R²=%{x:.3f}<extra></extra>",
        ))
        fig_g.add_vline(x=r2, line=dict(color=WARN, width=1.5, dash="dot"),
                        annotation_text=f"  Overall {r2:.3f}",
                        annotation_font=dict(color=WARN, size=11))
        fig_g.update_layout(
            title=dict(text="R² by genre", font=dict(size=13, color="#e5e7eb"), x=0, xanchor="left"),
            height=280, margin=dict(l=20,r=20,t=50,b=20),
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#cbd5e1"),
            xaxis=dict(range=[0,1], gridcolor="rgba(148,163,184,0.08)", zeroline=False),
            yaxis=dict(gridcolor="rgba(0,0,0,0)"),
        )
        st.plotly_chart(fig_g, use_container_width=True)
    with gcol_r:
        st.dataframe(gp_df, hide_index=True, use_container_width=True)

    # ---- Feature importances -----------------------------------------------
    st.markdown("---")
    st.markdown('<div class="section-title">Top 20 feature importances</div>', unsafe_allow_html=True)

    imp = pd.Series(bundle.model.feature_importances_, index=bundle.features).sort_values()
    top20 = imp.tail(20)

    fig_imp = go.Figure(go.Bar(
        x=top20.values, y=top20.index, orientation="h",
        marker=dict(
            color=top20.values,
            colorscale=[[0, ACCENT_S], [1, ACCENT]],
            line=dict(width=0),
        ),
        hovertemplate="%{y}: %{x:.4f}<extra></extra>",
    ))
    fig_imp.update_layout(
        height=500, margin=dict(l=20,r=20,t=20,b=20),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#cbd5e1"),
        xaxis=dict(title="Importance (gain)", gridcolor="rgba(148,163,184,0.08)", zeroline=False),
        yaxis=dict(gridcolor="rgba(0,0,0,0)"),
    )
    st.plotly_chart(fig_imp, use_container_width=True)

    # ---- Worst predictions -------------------------------------------------
    st.markdown("---")
    st.markdown('<div class="section-title">Where the model misses most (top 10 errors)</div>',
                unsafe_allow_html=True)
    st.caption("In-sample errors. High absolute errors often correspond to mega-events or "
               "artists with few training examples.")

    worst = (
        preds.nlargest(10, "abs_error")[
            ["headliner", "genre", "actual_rev", "pred_rev", "error", "pct_error"]
        ]
        .copy()
    )
    worst["actual_rev"]  = worst["actual_rev"].apply(mu.fmt_dollars)
    worst["pred_rev"]    = worst["pred_rev"].apply(mu.fmt_dollars)
    worst["error"]       = worst["error"].apply(mu.fmt_dollars)
    worst["pct_error"]   = worst["pct_error"].apply(lambda x: f"{x:+.0f}%")
    worst.columns        = ["Artist", "Genre", "Actual", "Predicted", "Error ($)", "Error (%)"]
    st.dataframe(worst, hide_index=True, use_container_width=True)


# ---------------------------------------------------------------------------
# Footer
# ---------------------------------------------------------------------------
cv = bundle.cv_stats
st.markdown(f"""
<div style="text-align:center; color:{MUTED}; font-size:0.75rem;
     margin-top:2rem; padding-top:1rem; border-top:1px solid rgba(148,163,184,0.08);">
    Model: XGBoost · GroupKFold CV (k=5, by headliner) ·
    CV R² {cv['mean_R2']:.3f} ±{cv['std_R2']:.3f} ·
    CV MAE {cv['mean_MAE']:.3f} ±{cv['std_MAE']:.3f} (z-scored) ·
    Trained on 1,808 events
</div>
""", unsafe_allow_html=True)
