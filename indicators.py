from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

_ROOT = Path(__file__).resolve().parents[1]
_EOD_SWING = _ROOT / "eod-swing"
if str(_EOD_SWING) not in sys.path:
    sys.path.insert(0, str(_EOD_SWING))

from eod_swing_lib import compute_ema, compute_rsi, infer_support_resistance  # noqa: E402

from config import DSSConfig


def typical_price(df: pd.DataFrame) -> pd.Series:
    return (df["High"].astype(float) + df["Low"].astype(float) + df["Close"].astype(float)) / 3.0


def rolling_vwap(df: pd.DataFrame, period: int) -> pd.Series:
    tp = typical_price(df)
    vol = df["Volume"].astype(float)
    pv = tp * vol
    return pv.rolling(period).sum() / vol.rolling(period).sum()


def compute_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    high = df["High"].astype(float)
    low = df["Low"].astype(float)
    close = df["Close"].astype(float)
    tr = pd.concat(
        [high - low, (high - close.shift(1)).abs(), (low - close.shift(1)).abs()],
        axis=1,
    ).max(axis=1)
    return tr.rolling(period).mean()


def add_indicators(df: pd.DataFrame, cfg: DSSConfig) -> pd.DataFrame:
    out = df.copy()
    close = out["Close"].astype(float)
    out["EMA20"] = compute_ema(close, cfg.ema_fast)
    out["EMA50"] = compute_ema(close, cfg.ema_slow)
    out["EMA200"] = compute_ema(close, 200)
    out["RSI"] = compute_rsi(close, cfg.rsi_period)
    out["VOL_MA"] = out["Volume"].astype(float).rolling(cfg.vol_ma_period).mean()
    out["VWAP"] = rolling_vwap(out, cfg.vwap_period)
    out["ATR"] = compute_atr(out, cfg.atr_period)
    out["VOL_RATIO"] = out["Volume"].astype(float) / out["VOL_MA"]

    supports: list[float] = []
    resistances: list[float] = []
    for i in range(len(out)):
        if i < 60:
            supports.append(np.nan)
            resistances.append(np.nan)
            continue
        window = out.iloc[: i + 1]
        price = float(close.iloc[i])
        sup, res, _ = infer_support_resistance(window, price, cfg.sr_lookback)
        supports.append(sup)
        resistances.append(res)

    out["SUPPORT"] = supports
    out["RESISTANCE"] = resistances
    return out
