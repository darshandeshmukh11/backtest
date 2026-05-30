#!/usr/bin/env python3
"""
NSE swing-trading decision support system — Streamlit UI.

Run:
  cd test/backtest && streamlit run app.py
"""

from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

from jindalstel_dss.backtest_engine import run_backtrader
from jindalstel_dss.config import DSSConfig
from jindalstel_dss.data import fetch_financials_table, fetch_fundamentals, fetch_ohlcv, resolve_yahoo_ticker
from jindalstel_dss.indicators import add_indicators
from jindalstel_dss.research import build_analyst_view
from jindalstel_dss.signals import analyze_indicator_adherence, generate_signals, simulate_swing_tranche
from jindalstel_dss.symbols import (
    default_symbol_index,
    load_nse_equity_symbols,
    normalize_symbol,
    symbol_picker_options,
    universe_for_index,
)
from jindalstel_dss.zones import add_zones, latest_zones

st.set_page_config(
    page_title="NSE Swing DSS",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)


@st.cache_data(ttl=3600, show_spinner=False)
def _load_nse_universe() -> list[str]:
    return load_nse_equity_symbols()


@st.cache_data(ttl=3600, show_spinner="Loading market data…")
def load_pipeline(cfg: DSSConfig) -> tuple[pd.DataFrame, object, object, object, object]:
    ohlcv = fetch_ohlcv(cfg.symbol, cfg.years)
    fundamentals = fetch_fundamentals(cfg.symbol)
    enriched = add_indicators(ohlcv, cfg)
    enriched = add_zones(enriched, cfg)
    enriched = generate_signals(enriched, cfg)
    adherence = analyze_indicator_adherence(enriched)
    swing_bt = simulate_swing_tranche(enriched, cfg)
    bt_summary = run_backtrader(enriched, cfg)
    return enriched, fundamentals, adherence, swing_bt, bt_summary


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
        rows=3,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.03,
        row_heights=[0.55, 0.22, 0.23],
        subplot_titles=("Price · zones · signals", "Volume", "RSI"),
    )

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
    for col, name, color in [
        ("EMA20", "EMA 20", "#f59e0b"),
        ("EMA50", "EMA 50", "#3b82f6"),
        ("VWAP", f"VWAP {cfg.vwap_period}d", "#a855f7"),
    ]:
        fig.add_trace(
            go.Scatter(x=df.index, y=df[col], mode="lines", name=name, line=dict(color=color, width=1.5)),
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

    fig.update_layout(
        height=820,
        xaxis_rangeslider_visible=False,
        template="plotly_dark",
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
        margin=dict(l=40, r=40, t=60, b=40),
    )
    fig.update_yaxes(title_text="Price ₹", row=1, col=1)
    fig.update_yaxes(title_text="Volume", row=2, col=1)
    fig.update_yaxes(title_text="RSI", row=3, col=1, range=[20, 85])
    return fig


def main() -> None:
    st.title("NSE Swing Trading — Decision Support")
    st.caption(
        "Equity research + technical zones for any **NSE** stock (Yahoo `SYMBOL.NS`). "
        "Uses MA, Volume, RSI, VWAP, ATR · [Backtrader](https://www.backtrader.com/) validation."
    )

    nse_universe = _load_nse_universe()

    with st.sidebar:
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
        )
        st.divider()
        st.markdown(
            f"**{symbol}**  \n"
            f"**Core:** {cfg.core_qty:,} shares  \n"
            f"**Swing:** {cfg.swing_qty:,} shares  \n"
            f"**Target lot restore:** {cfg.core_holding_qty:,}"
        )

    try:
        df, fundamentals, adherence, swing_bt, bt_summary = load_pipeline(cfg)
    except Exception as exc:
        st.error(f"Failed to load data: {exc}")
        st.stop()

    zones = latest_zones(df)
    last_price = float(df["Close"].iloc[-1])

    m1, m2, m3, m4, m5, m6 = st.columns(6)
    m1.metric("Symbol", symbol)
    m2.metric("Last close", f"₹{last_price:,.2f}")
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
            "Green band = buy zone (support / EMA20 / VWAP cluster). "
            "Red band = sell zone (resistance trim). Triangles = raw signals; "
            "circles B/S = simulated swing-tranche entries/exits on your configured lot."
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
