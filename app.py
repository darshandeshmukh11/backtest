#!/usr/bin/env python3
"""
NSE swing-trading decision support system — Streamlit UI.

Run:
  cd test/backtest && streamlit run app.py
"""

from __future__ import annotations

from datetime import datetime, timedelta

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

from backtest_engine import run_backtrader
from config import DSSConfig
from data import (
    IST,
    fetch_financials_table,
    fetch_fundamentals,
    fetch_ohlcv,
    fetch_ohlcv_live,
    ist_now,
    market_session_status,
    resolve_yahoo_ticker,
)
from indicators import add_indicators
from research import build_analyst_view
from signals import (
    analyze_indicator_adherence,
    evaluate_live_signal,
    generate_signals,
    simulate_swing_tranche,
)
from symbols import (
    default_symbol_index,
    load_nse_equity_symbols,
    normalize_symbol,
    symbol_picker_options,
    universe_for_index,
)
from zones import add_zones, latest_zones

DEPLOY_VERSION = "2026-06-08-realtime"

st.set_page_config(
    page_title="NSE Swing DSS",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)


def _inject_metric_css() -> None:
    """Prevent st.metric values from truncating with ellipsis on long labels."""
    st.markdown(
        """
        <style>
        [data-testid="stMetricValue"],
        [data-testid="stMetricValue"] > div {
            white-space: normal !important;
            overflow: visible !important;
            text-overflow: unset !important;
            word-break: break-word;
            line-height: 1.3;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


@st.cache_data(ttl=3600, show_spinner=False)
def _load_nse_universe() -> list[str]:
    return load_nse_equity_symbols()


@st.cache_data(ttl=120, show_spinner="Loading market data…")
def load_pipeline(
    cfg: DSSConfig,
    refresh_key: int = 0,
) -> tuple[pd.DataFrame, object, object, object, object, object | None]:
    if cfg.use_realtime:
        ohlcv, quote = fetch_ohlcv_live(cfg.symbol, cfg.years)
    else:
        ohlcv = fetch_ohlcv(cfg.symbol, cfg.years)
        quote = None
    fundamentals = fetch_fundamentals(cfg.symbol)
    enriched = add_indicators(ohlcv, cfg)
    enriched = add_zones(enriched, cfg)
    enriched = generate_signals(enriched, cfg)
    adherence = analyze_indicator_adherence(enriched)
    swing_bt = simulate_swing_tranche(enriched, cfg)
    bt_summary = run_backtrader(enriched, cfg)
    return enriched, fundamentals, adherence, swing_bt, bt_summary, quote


def trades_dataframe(swing_bt) -> pd.DataFrame:
    rows = []
    for t in swing_bt.trades:
        rows.append(
            {
                "Side": t.side,
                "Date": pd.Timestamp(t.date).strftime("%Y-%m-%d"),
                "Price": round(t.price, 2),
                "Qty": t.qty,
                "Reason": t.reason,
                "P&L ₹": round(t.pnl, 2) if t.pnl is not None else None,
                "Return %": round(t.pnl_pct, 2) if t.pnl_pct is not None else None,
            }
        )
    return pd.DataFrame(rows)


def build_price_chart(df: pd.DataFrame, trades, cfg: DSSConfig) -> go.Figure:
    fig = make_subplots(
        rows=4,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.025,
        row_heights=[0.46, 0.18, 0.18, 0.18],
        subplot_titles=(
            f"Price · VWAP {cfg.vwap_period}d · ATR {cfg.atr_period}d bands · zones · signals",
            "Volume",
            "RSI",
            f"ATR ({cfg.atr_period})",
        ),
    )

    close = df["Close"].astype(float)
    atr = df["ATR"].astype(float)

    fig.add_trace(
        go.Candlestick(
            x=df.index,
            open=df["Open"],
            high=df["High"],
            low=df["Low"],
            close=df["Close"],
            name="OHLC",
            increasing_line_color="#22c55e",
            decreasing_line_color="#ef4444",
        ),
        row=1,
        col=1,
    )
    for col, name, color, width, dash in [
        ("EMA20", "EMA 20", "#f59e0b", 1.5, None),
        ("EMA50", "EMA 50", "#3b82f6", 1.5, None),
        ("VWAP", f"VWAP {cfg.vwap_period}d", "#c084fc", 2.2, "solid"),
    ]:
        line_style = dict(color=color, width=width)
        if dash:
            line_style["dash"] = dash
        fig.add_trace(
            go.Scatter(
                x=df.index,
                y=df[col],
                mode="lines",
                name=name,
                line=line_style,
            ),
            row=1,
            col=1,
        )

    # ATR envelope around close (±1 ATR) — stop/target context on price panel
    atr_upper = close + atr
    atr_lower = close - atr
    fig.add_trace(
        go.Scatter(
            x=df.index,
            y=atr_upper,
            mode="lines",
            name=f"Close + ATR({cfg.atr_period})",
            line=dict(color="#22d3ee", width=1, dash="dot"),
            opacity=0.85,
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=df.index,
            y=atr_lower,
            mode="lines",
            name=f"Close − ATR({cfg.atr_period})",
            line=dict(color="#22d3ee", width=1, dash="dot"),
            fill="tonexty",
            fillcolor="rgba(34,211,238,0.08)",
            opacity=0.85,
        ),
        row=1,
        col=1,
    )

    # Buy / sell zone bands (latest 60 sessions for clarity)
    tail = df.tail(120)
    fig.add_trace(
        go.Scatter(
            x=tail.index.tolist() + tail.index.tolist()[::-1],
            y=tail["BUY_ZONE_HIGH"].tolist() + tail["BUY_ZONE_LOW"].tolist()[::-1],
            fill="toself",
            fillcolor="rgba(34,197,94,0.12)",
            line=dict(color="rgba(34,197,94,0.2)"),
            name="Buy zone",
            hoverinfo="skip",
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=tail.index.tolist() + tail.index.tolist()[::-1],
            y=tail["SELL_ZONE_HIGH"].tolist() + tail["SELL_ZONE_LOW"].tolist()[::-1],
            fill="toself",
            fillcolor="rgba(239,68,68,0.10)",
            line=dict(color="rgba(239,68,68,0.2)"),
            name="Sell zone",
            hoverinfo="skip",
        ),
        row=1,
        col=1,
    )

    buys = df[df["BUY_SIGNAL"]]
    sells = df[df["SELL_SIGNAL"]]
    if not buys.empty:
        fig.add_trace(
            go.Scatter(
                x=buys.index,
                y=buys["Low"] * 0.992,
                mode="markers",
                name="Buy signal",
                marker=dict(symbol="triangle-up", size=10, color="#22c55e"),
            ),
            row=1,
            col=1,
        )
    if not sells.empty:
        fig.add_trace(
            go.Scatter(
                x=sells.index,
                y=sells["High"] * 1.008,
                mode="markers",
                name="Sell signal",
                marker=dict(symbol="triangle-down", size=10, color="#ef4444"),
            ),
            row=1,
            col=1,
        )

    for t in trades:
        if t.side == "BUY":
            fig.add_trace(
                go.Scatter(
                    x=[t.date],
                    y=[t.price],
                    mode="markers+text",
                    text=["B"],
                    textposition="bottom center",
                    name="Swing buy",
                    marker=dict(symbol="circle", size=12, color="#16a34a", line=dict(width=2, color="white")),
                    showlegend=False,
                ),
                row=1,
                col=1,
            )
        elif t.side == "SELL":
            fig.add_trace(
                go.Scatter(
                    x=[t.date],
                    y=[t.price],
                    mode="markers+text",
                    text=["S"],
                    textposition="top center",
                    name="Swing sell",
                    marker=dict(symbol="circle", size=12, color="#dc2626", line=dict(width=2, color="white")),
                    showlegend=False,
                ),
                row=1,
                col=1,
            )

    colors = ["#22c55e" if c >= o else "#ef4444" for c, o in zip(df["Close"], df["Open"])]
    fig.add_trace(
        go.Bar(x=df.index, y=df["Volume"], name="Volume", marker_color=colors, opacity=0.55),
        row=2,
        col=1,
    )
    fig.add_trace(
        go.Scatter(x=df.index, y=df["VOL_MA"], mode="lines", name="Vol MA20", line=dict(color="#94a3b8")),
        row=2,
        col=1,
    )

    fig.add_trace(
        go.Scatter(x=df.index, y=df["RSI"], mode="lines", name="RSI", line=dict(color="#eab308")),
        row=3,
        col=1,
    )
    fig.add_hline(y=cfg.rsi_buy_min, line_dash="dot", line_color="#22c55e", row=3, col=1)
    fig.add_hline(y=cfg.rsi_sell_max, line_dash="dot", line_color="#ef4444", row=3, col=1)
    fig.add_hline(y=50, line_dash="dash", line_color="#64748b", row=3, col=1)

    fig.add_trace(
        go.Scatter(
            x=df.index,
            y=df["ATR"],
            mode="lines",
            name=f"ATR {cfg.atr_period}",
            line=dict(color="#22d3ee", width=2),
            fill="tozeroy",
            fillcolor="rgba(34,211,238,0.12)",
        ),
        row=4,
        col=1,
    )
    last_atr = float(atr.iloc[-1]) if not atr.empty and pd.notna(atr.iloc[-1]) else None
    if last_atr is not None:
        fig.add_annotation(
            xref="x4 domain",
            yref="y4",
            x=1.01,
            y=last_atr,
            text=f"₹{last_atr:.2f}",
            showarrow=False,
            font=dict(color="#22d3ee", size=11),
            xanchor="left",
        )

    fig.update_layout(
        height=920,
        xaxis_rangeslider_visible=False,
        template="plotly_dark",
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
        margin=dict(l=40, r=40, t=60, b=40),
    )
    fig.update_yaxes(title_text="Price ₹", row=1, col=1)
    fig.update_yaxes(title_text="Volume", row=2, col=1)
    fig.update_yaxes(title_text="RSI", row=3, col=1, range=[20, 85])
    fig.update_yaxes(title_text="ATR ₹", row=4, col=1)
    return fig


def main() -> None:
    _inject_metric_css()
    st.title("NSE Swing Trading — Decision Support")
    st.caption(
        "Equity research + technical zones for any **NSE** stock (Yahoo `SYMBOL.NS`). "
        "Uses MA, Volume, RSI, VWAP, ATR · [Backtrader](https://www.backtrader.com/) validation."
    )

    nse_universe = _load_nse_universe()

    with st.sidebar:
        if st.button(
            "Refresh data",
            type="primary",
            use_container_width=True,
            help="Fetch latest OHLCV and fundamentals from Yahoo (clears cache).",
        ):
            load_pipeline.clear()
            _load_nse_universe.clear()
            st.session_state["live_refresh_key"] = st.session_state.get("live_refresh_key", 0) + 1
            st.session_state["live_last_tick"] = ist_now()
            st.session_state["last_data_refresh"] = datetime.now(IST).strftime("%Y-%m-%d %H:%M:%S %Z")
            st.rerun()

        if refreshed := st.session_state.get("last_data_refresh"):
            st.caption(f"Last refresh: {refreshed}")

        st.divider()
        st.header("Stock")
        index_filter = st.radio(
            "Universe",
            ["All NSE", "Nifty 50", "Nifty 100"],
            horizontal=True,
        )
        active_universe = universe_for_index(index_filter, nse_universe)
        search = st.text_input(
            "Search symbol",
            placeholder="RELIANCE, JINDALSTEL, TATAST…",
        )
        picker_options = symbol_picker_options(search, active_universe)
        if search.strip() and not picker_options:
            st.warning(f"No match for «{search.strip().upper()}» in {index_filter}.")
            picker_options = active_universe[:1] if active_universe else ["RELIANCE"]

        if search.strip():
            sym_label = f"Symbol ({len(picker_options)} matches)"
        elif index_filter == "All NSE":
            sym_label = f"Symbol ({len(active_universe):,} NSE equities)"
        else:
            sym_label = f"Symbol — {index_filter} ({len(active_universe)})"

        symbol = st.selectbox(
            sym_label,
            options=picker_options,
            index=default_symbol_index(picker_options, "JINDALSTEL"),
        )
        symbol = normalize_symbol(symbol)
        yahoo = resolve_yahoo_ticker(symbol)
        st.caption(f"Yahoo: **{yahoo}**")
        st.caption(f"Build: `{DEPLOY_VERSION}`")

        st.divider()
        st.subheader("Realtime")
        use_realtime = st.checkbox(
            "Realtime (live LTP)",
            value=True,
            help="Merge Yahoo LTP into today's bar for live RSI, zones, and signals.",
        )
        auto_refresh = st.checkbox(
            "Auto-refresh LTP",
            value=use_realtime,
            disabled=not use_realtime,
        )
        refresh_sec = st.slider(
            "Refresh interval (sec)",
            30,
            300,
            90,
            15,
            disabled=not (use_realtime and auto_refresh),
        )

        st.divider()
        st.header("Parameters")
        cfg = DSSConfig(
            symbol=symbol,
            core_holding_qty=st.number_input("Total holding (shares)", 100, 50000, 4460, step=10),
            swing_pct=st.slider("Swing tranche (% of holding)", 5, 40, 10),
            years=st.selectbox("History (years)", [1, 2, 3], index=2),
            profit_target_pct=st.number_input("Swing profit target (%)", 0.5, 10.0, 2.5, 0.25),
            stop_atr_mult=st.number_input("Stop (× ATR)", 0.5, 3.0, 1.5, 0.1),
            rsi_buy_min=st.number_input("RSI buy floor", 45.0, 60.0, 52.0),
            backtest_cash=st.number_input("Backtrader capital (₹)", 50_000, 5_000_000, 500_000, 50_000),
            use_realtime=use_realtime,
        )
        st.divider()
        st.markdown(
            f"**{symbol}**  \n"
            f"**Core:** {cfg.core_qty:,} shares  \n"
            f"**Swing:** {cfg.swing_qty:,} shares  \n"
            f"**Target lot restore:** {cfg.core_holding_qty:,}"
        )

    refresh_key = st.session_state.get("live_refresh_key", 0)
    if cfg.use_realtime and auto_refresh:
        last_tick = st.session_state.get("live_last_tick")
        if last_tick is None or ist_now() - last_tick >= timedelta(seconds=refresh_sec):
            st.session_state["live_refresh_key"] = refresh_key + 1
            st.session_state["live_last_tick"] = ist_now()
            st.rerun()

    try:
        df, fundamentals, adherence, swing_bt, bt_summary, quote = load_pipeline(
            cfg, refresh_key if cfg.use_realtime else 0
        )
    except Exception as exc:
        st.error(f"Failed to load data: {exc}")
        st.stop()

    zones = latest_zones(df)
    last_price = float(df["Close"].iloc[-1])
    live_state = evaluate_live_signal(df, cfg) if cfg.use_realtime else None
    session = market_session_status()

    if cfg.use_realtime:
        if session["is_open"]:
            st.success(
                f"**NSE open** · {session['as_of']} · Live LTP drives price, RSI, and zones. "
                f"Volume filter uses the **last completed session** until today's volume is in."
            )
        else:
            st.info(
                f"**NSE {session['phase']}** · {session['as_of']} · Live LTP still updates the last bar."
            )

    if live_state:
        st.subheader("Live signal")
        st.markdown(f"**Action:** {live_state.action}")
        sig_cols = st.columns(5)
        sig_cols[0].metric("LTP", f"₹{live_state.live_ltp:,.2f}", f"{live_state.change_pct:+.2f}%")
        sig_cols[1].metric("RSI", f"{live_state.rsi:.1f}")
        sig_cols[2].metric("Trend", "Bullish" if live_state.bullish else "Weak")
        sig_cols[3].metric(
            "Signal today",
            "BUY" if live_state.buy_signal_today else ("SELL" if live_state.sell_signal_today else "—"),
        )
        sig_cols[4].metric("Updated", live_state.as_of.split(" ")[1])
        if live_state.buy_ready:
            st.success("**Swing BUY** — trend + buy zone / dip entry conditions are met on live LTP.")
        elif live_state.sell_alert:
            st.warning(f"**Swing SELL / TRIM** — {live_state.action}")
        else:
            st.caption(" · ".join(live_state.reasons))
        if quote:
            st.caption(
                f"Yahoo LTP ₹{quote.price:,.2f} · Day H/L ₹{quote.day_high:,.2f} / ₹{quote.day_low:,.2f}"
                + (f" · Vol {quote.volume:,.0f}" if quote.volume > 0 else "")
            )

    m1, m2, m3, m4, m5, m6 = st.columns([1.4, 1, 1, 1, 1, 1])
    m1.metric("Symbol", symbol)
    price_label = "LTP" if cfg.use_realtime else "Last close"
    m2.metric(price_label, f"₹{last_price:,.2f}")
    m3.metric("RSI", f"{float(df['RSI'].iloc[-1]):.1f}")
    m4.metric("Swing P&L (sim)", f"₹{swing_bt.total_pnl:+,.0f}")
    m5.metric("Win rate", f"{swing_bt.win_rate:.0f}%")
    m6.metric("BT return", f"{bt_summary.total_return_pct:+.1f}%")

    tab_chart, tab_research, tab_trades, tab_funda, tab_bt = st.tabs(
        ["Chart & zones", "Analyst view", "Swing trades", "Fundamentals", "Backtrader"]
    )

    with tab_chart:
        st.plotly_chart(build_price_chart(df, swing_bt.trades, cfg), use_container_width=True)
        c1, c2 = st.columns(2)
        with c1:
            st.subheader("Buy zone (dip accumulation)")
            st.success(
                f"₹{zones['buy_low']:,.2f} – ₹{zones['buy_high']:,.2f}  \n"
                f"Support ₹{zones['support']:,.2f} · ATR ₹{zones['atr']:,.2f}"
            )
        with c2:
            st.subheader("Sell zone (profit booking)")
            st.error(
                f"₹{zones['sell_low']:,.2f} – ₹{zones['sell_high']:,.2f}  \n"
                f"Resistance ₹{zones['resistance']:,.2f}"
            )
        st.info(
            "Purple **VWAP** line = rolling volume-weighted average. Cyan **ATR bands** = close ± 1 ATR "
            "(typical swing stop width). Bottom panel = raw ATR (₹). Green/red bands = buy/sell zones. "
            "Triangles = raw signals; circles B/S = simulated swing-tranche entries/exits."
        )

    with tab_research:
        narrative = build_analyst_view(
            cfg.symbol, df, fundamentals, adherence, swing_bt, cfg
        )
        st.markdown("\n".join(narrative))
        st.subheader(f"Indicator adherence ({cfg.years}Y · {symbol})")
        ad_df = pd.DataFrame(
            [
                {
                    "Indicator": a.name,
                    "On up weeks %": a.hit_rate_on_up_weeks,
                    "On down weeks %": a.hit_rate_on_down_weeks,
                    "Avg 5d fwd % when true": a.avg_forward_5d_pct_when_true,
                }
                for a in adherence
            ]
        )
        st.dataframe(ad_df, use_container_width=True, hide_index=True)

    with tab_trades:
        tdf = trades_dataframe(swing_bt)
        if tdf.empty:
            st.warning("No swing trades in this window.")
        else:
            st.dataframe(tdf, use_container_width=True, hide_index=True)
            st.caption(
                f"Simulated rotation of **{cfg.swing_qty}** shares while keeping "
                f"**{cfg.core_qty}** as core. Rebuy after dips to restore **{cfg.core_holding_qty}**."
            )
        if swing_bt.equity_curve is not None:
            st.subheader("Portfolio mark-to-market (core + swing)")
            st.line_chart(swing_bt.equity_curve)

    with tab_funda:
        st.subheader(fundamentals.company_name)
        f1, f2, f3 = st.columns(3)
        f1.metric("P/E (trail)", f"{fundamentals.trailing_pe:.1f}" if fundamentals.trailing_pe else "—")
        f2.metric("P/B", f"{fundamentals.price_to_book:.2f}" if fundamentals.price_to_book else "—")
        f3.metric("Mkt cap (Cr)", f"{fundamentals.market_cap_cr:,.0f}" if fundamentals.market_cap_cr else "—")
        annual, quarterly = fetch_financials_table(cfg.symbol)
        if not annual.empty:
            st.markdown("**Annual income statement (₹)**")
            st.dataframe(annual / 1e7, use_container_width=True)
        if not quarterly.empty:
            st.markdown("**Quarterly income statement (₹ Cr)**")
            st.dataframe(quarterly / 1e7, use_container_width=True)
        if annual.empty and quarterly.empty:
            st.info("Fundamental tables unavailable from Yahoo for this ticker.")

    with tab_bt:
        st.markdown(
            "Independent validation via [Backtrader](https://www.backtrader.com/) "
            "(same EMA/RSI/volume stack on a capital pool)."
        )
        b1, b2, b3, b4 = st.columns(4)
        b1.metric("Start ₹", f"{bt_summary.start_value:,.0f}")
        b2.metric("End ₹", f"{bt_summary.end_value:,.0f}")
        b3.metric("Max DD", f"{bt_summary.max_drawdown_pct:.1f}%")
        b4.metric("Sharpe", f"{bt_summary.sharpe:.2f}" if bt_summary.sharpe else "—")
        st.write(
            f"Closed trades: **{bt_summary.closed_trades}** · Won: **{bt_summary.won_trades}**"
        )
        for note in bt_summary.notes:
            st.caption(note)


if __name__ == "__main__":
    main()
