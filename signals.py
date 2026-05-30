from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from config import DSSConfig


@dataclass
class TradeRecord:
    side: str
    date: pd.Timestamp
    price: float
    qty: int
    reason: str
    pnl: float | None = None
    pnl_pct: float | None = None


@dataclass
class BacktestResult:
    trades: list[TradeRecord] = field(default_factory=list)
    equity_curve: pd.Series | None = None
    total_pnl: float = 0.0
    win_rate: float = 0.0
    avg_pnl_pct: float = 0.0
    signals_buy: int = 0
    signals_sell: int = 0


def _bullish_state(row: pd.Series, cfg: DSSConfig) -> bool:
    return (
        float(row["Close"]) > float(row["EMA20"]) > float(row["EMA50"])
        and cfg.rsi_buy_min <= float(row["RSI"]) <= cfg.rsi_sell_max
        and float(row["VOL_RATIO"]) >= 0.95
        and float(row["Close"]) >= float(row["VWAP"]) * 0.99
    )


def _dip_entry(row: pd.Series) -> bool:
    close = float(row["Close"])
    in_zone = bool(row["IN_BUY_ZONE"]) or close <= float(row["BUY_ZONE_HIGH"])
    near_ema = abs(close - float(row["EMA20"])) / float(row["EMA20"]) <= 0.025
    return in_zone or near_ema


def generate_signals(df: pd.DataFrame, cfg: DSSConfig) -> pd.DataFrame:
    """Edge-triggered buy/sell markers for chart + swing simulation."""
    out = df.copy()
    warmup = max(cfg.ema_slow, cfg.vol_ma_period, cfg.vwap_period, 60)

    states: list[bool] = []
    for i in range(len(out)):
        if i < warmup:
            states.append(False)
            continue
        states.append(_bullish_state(out.iloc[i], cfg))

    out["_BULL"] = states
    out["_BULL_PREV"] = out["_BULL"].shift(1).fillna(False).astype(bool)

    buy_sig = out["_BULL"] & ~out["_BULL_PREV"] & out.apply(_dip_entry, axis=1)
    sell_sig = (
        (out["_BULL_PREV"] & ~out["_BULL"])
        | (out["RSI"] >= cfg.rsi_sell_max)
        | (out["IN_SELL_ZONE"] & (out["RSI"] >= cfg.rsi_buy_min))
    )
    buy_sig.iloc[:warmup] = False
    sell_sig.iloc[:warmup] = False

    out["BUY_SIGNAL"] = buy_sig.astype(bool)
    out["SELL_SIGNAL"] = sell_sig.astype(bool)
    out.drop(columns=["_BULL", "_BULL_PREV"], inplace=True)
    return out


def simulate_swing_tranche(
    df: pd.DataFrame,
    cfg: DSSConfig,
    swing_qty: int | None = None,
) -> BacktestResult:
    """
    Core holding stays at core_qty; swing tranche rotates on signals.
    Buy swing tranche on BUY_SIGNAL; sell on SELL_SIGNAL or profit target.
    """
    qty = swing_qty or cfg.swing_qty
    core = cfg.core_qty
    cash_proceeds = 0.0
    in_swing = False
    entry_price = 0.0
    entry_date: pd.Timestamp | None = None
    trades: list[TradeRecord] = []
    equity: list[float] = []

    for i in range(len(df)):
        row = df.iloc[i]
        price = float(row["Close"])
        date = pd.Timestamp(df.index[i])

        if not in_swing and row["BUY_SIGNAL"]:
            entry_price = price
            entry_date = date
            in_swing = True
            cash_proceeds = 0.0
            trades.append(
                TradeRecord("BUY", date, price, qty, "Swing entry — trend + buy zone + volume")
            )
        elif in_swing and entry_date is not None:
            target = entry_price * (1 + cfg.profit_target_pct / 100)
            stop = entry_price - cfg.stop_atr_mult * float(row["ATR"])
            hit_target = float(row["High"]) >= target
            hit_stop = float(row["Low"]) <= stop
            exit_reason = None
            exit_price = price

            if hit_target:
                exit_reason = f"Target +{cfg.profit_target_pct:.1f}%"
                exit_price = target
            elif hit_stop:
                exit_reason = f"Stop (ATR×{cfg.stop_atr_mult:.1f})"
                exit_price = stop
            elif row["SELL_SIGNAL"]:
                exit_reason = "Sell signal — trend weak / sell zone"
                exit_price = price

            if exit_reason:
                pnl = (exit_price - entry_price) * qty
                pnl_pct = (exit_price / entry_price - 1) * 100
                cash_proceeds = exit_price * qty
                trades.append(
                    TradeRecord("SELL", date, exit_price, qty, exit_reason, pnl, pnl_pct)
                )
                in_swing = False
                entry_price = 0.0
                entry_date = None

        portfolio_value = core * price + (qty * price if in_swing else cash_proceeds)
        equity.append(portfolio_value)

    closed = [t for t in trades if t.side == "SELL"]
    wins = [t for t in closed if (t.pnl or 0) > 0]
    total_pnl = sum(t.pnl or 0 for t in closed)
    avg_pct = float(np.mean([t.pnl_pct or 0 for t in closed])) if closed else 0.0

    return BacktestResult(
        trades=trades,
        equity_curve=pd.Series(equity, index=df.index, name="Portfolio"),
        total_pnl=total_pnl,
        win_rate=len(wins) / len(closed) * 100 if closed else 0.0,
        avg_pnl_pct=avg_pct,
        signals_buy=int(df["BUY_SIGNAL"].sum()),
        signals_sell=int(df["SELL_SIGNAL"].sum()),
    )


@dataclass
class IndicatorAdherence:
    name: str
    hit_rate_on_up_weeks: float
    hit_rate_on_down_weeks: float
    avg_forward_5d_pct_when_true: float
    description: str


def analyze_indicator_adherence(df: pd.DataFrame) -> list[IndicatorAdherence]:
    """Which technical conditions the stock tended to follow over the sample."""
    if len(df) < 80:
        return []

    fwd5 = df["Close"].astype(float).pct_change(5).shift(-5) * 100
    up_week = fwd5 > 0
    down_week = fwd5 <= 0

    checks = [
        ("Close > EMA20 > EMA50", (df["Close"] > df["EMA20"]) & (df["EMA20"] > df["EMA50"])),
        ("RSI > 55", df["RSI"] > 55),
        ("Volume > 20d avg", df["VOL_RATIO"] > 1.0),
        ("Close above VWAP", df["Close"] >= df["VWAP"]),
        ("In buy zone", df["IN_BUY_ZONE"]),
        ("In sell zone", df["IN_SELL_ZONE"]),
    ]

    results: list[IndicatorAdherence] = []
    for name, mask in checks:
        m = mask.fillna(False)
        up_hit = m[up_week].mean() * 100 if up_week.any() else 0.0
        dn_hit = m[down_week].mean() * 100 if down_week.any() else 0.0
        avg_fwd = float(fwd5[m].mean()) if m.any() else 0.0
        results.append(
            IndicatorAdherence(
                name=name,
                hit_rate_on_up_weeks=round(up_hit, 1),
                hit_rate_on_down_weeks=round(dn_hit, 1),
                avg_forward_5d_pct_when_true=round(avg_fwd, 2),
                description=f"5-day forward return when true: {avg_fwd:+.2f}%",
            )
        )
    return results
