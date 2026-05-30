from __future__ import annotations

import pandas as pd

from config import DSSConfig
from data import FundamentalSnapshot
from signals import BacktestResult, IndicatorAdherence


def build_analyst_view(
    symbol: str,
    df: pd.DataFrame,
    fundamentals: FundamentalSnapshot,
    adherence: list[IndicatorAdherence],
    swing_result: BacktestResult,
    cfg: DSSConfig,
) -> list[str]:
    """Senior-analyst style narrative for the DSS."""
    if df.empty:
        return ["Insufficient price history."]

    last = df.iloc[-1]
    price = float(last["Close"])
    ema20 = float(last["EMA20"])
    ema50 = float(last["EMA50"])
    rsi = float(last["RSI"])
    vol_ratio = float(last["VOL_RATIO"])

    lines: list[str] = [
        f"**{fundamentals.company_name} ({symbol})** — swing decision support as of {fundamentals.as_of}.",
        "",
        "### Business & fundamentals (Yahoo Finance)",
        f"- Sector: {fundamentals.sector} · Industry: {fundamentals.industry}",
    ]
    if fundamentals.market_cap_cr:
        lines.append(f"- Market cap: ~₹{fundamentals.market_cap_cr:,.0f} Cr")
    if fundamentals.trailing_pe:
        lines.append(f"- Trailing P/E: {fundamentals.trailing_pe:.1f}")
    if fundamentals.debt_to_equity is not None:
        lines.append(f"- Debt/Equity: {fundamentals.debt_to_equity:.1f}")
    if fundamentals.revenue_growth_pct is not None:
        lines.append(f"- Revenue growth: {fundamentals.revenue_growth_pct:+.1f}%")
    for note in fundamentals.notes:
        lines.append(f"- {note}")

    trend = "uptrend" if price > ema20 > ema50 else "mixed / corrective" if price > ema50 else "weak / below 50 EMA"
    lines.extend(
        [
            "",
            "### Technical posture (latest session)",
            f"- Close **₹{price:,.2f}** · EMA20 ₹{ema20:,.2f} · EMA50 ₹{ema50:,.2f} · RSI {rsi:.1f}",
            f"- Volume vs 20d avg: **{vol_ratio:.2f}×** · VWAP({cfg.vwap_period}d): ₹{float(last['VWAP']):,.2f}",
            f"- Structure: **{trend}** on daily timeframe.",
            "",
            "### Holding model",
            f"- Core delivery lot: **{cfg.core_qty:,}** shares (buy-and-hold base).",
            f"- Swing tranche: **{cfg.swing_qty:,}** shares ({cfg.swing_pct:.0f}% of {cfg.core_holding_qty:,}) "
            f"to rotate for short swings; rebuy dips to restore full **{cfg.core_holding_qty:,}** when price falls.",
            f"- Profit target on swing leg: **+{cfg.profit_target_pct:.1f}%** · Stop: **{cfg.stop_atr_mult:.1f}× ATR**.",
            "",
            f"### What {symbol} has adhered to ({cfg.years}Y sample)",
        ]
    )
    for ad in adherence[:6]:
        lines.append(
            f"- **{ad.name}**: present on {ad.hit_rate_on_up_weeks:.0f}% of up-5d weeks vs "
            f"{ad.hit_rate_on_down_weeks:.0f}% of down weeks · {ad.description}"
        )

    closed = [t for t in swing_result.trades if t.side == "SELL"]
    lines.extend(
        [
            "",
            "### Simulated swing-tranche backtest (historical signals)",
            f"- Buy signals: {swing_result.signals_buy} · Sell signals: {swing_result.signals_sell}",
            f"- Completed swing legs: **{len(closed)}** · Win rate **{swing_result.win_rate:.0f}%**",
            f"- Total swing P&L (tranche only): **₹{swing_result.total_pnl:+,.0f}** · Avg leg **{swing_result.avg_pnl_pct:+.2f}%**",
            "",
            "_Disclaimer: Research support only — not investment advice. Past simulated signals ≠ future results._",
        ]
    )
    return lines
