from __future__ import annotations

import numpy as np
import pandas as pd

from config import DSSConfig


def add_zones(df: pd.DataFrame, cfg: DSSConfig) -> pd.DataFrame:
    """
    Buy zone: around swing support / EMA20 / VWAP (dip accumulation).
    Sell zone: around swing resistance / R1 proxy (profit booking).
    """
    out = df.copy()
    atr = out["ATR"].astype(float)
    close = out["Close"].astype(float)
    ema20 = out["EMA20"].astype(float)
    vwap = out["VWAP"].astype(float)
    support = out["SUPPORT"].astype(float)
    resistance = out["RESISTANCE"].astype(float)

    band = atr * cfg.zone_atr_mult
    band = band.fillna(close * 0.012)

    dip_anchor = pd.concat([support, ema20, vwap], axis=1).min(axis=1)
    out["BUY_ZONE_LOW"] = (dip_anchor - band).round(2)
    out["BUY_ZONE_HIGH"] = (dip_anchor + band * 0.5).round(2)

    trim_anchor = pd.concat([resistance, close * 1.02], axis=1).max(axis=1)
    out["SELL_ZONE_LOW"] = (trim_anchor - band * 0.5).round(2)
    out["SELL_ZONE_HIGH"] = (trim_anchor + band).round(2)

    out["IN_BUY_ZONE"] = (close >= out["BUY_ZONE_LOW"]) & (close <= out["BUY_ZONE_HIGH"])
    out["IN_SELL_ZONE"] = (close >= out["SELL_ZONE_LOW"]) & (close <= out["SELL_ZONE_HIGH"])
    return out


def latest_zones(df: pd.DataFrame) -> dict[str, float]:
    if df.empty:
        return {}
    row = df.iloc[-1]
    return {
        "buy_low": float(row["BUY_ZONE_LOW"]),
        "buy_high": float(row["BUY_ZONE_HIGH"]),
        "sell_low": float(row["SELL_ZONE_LOW"]),
        "sell_high": float(row["SELL_ZONE_HIGH"]),
        "support": float(row["SUPPORT"]) if not np.isnan(row["SUPPORT"]) else float(row["Close"]),
        "resistance": float(row["RESISTANCE"]) if not np.isnan(row["RESISTANCE"]) else float(row["Close"]),
        "atr": float(row["ATR"]) if not np.isnan(row["ATR"]) else float(row["Close"]) * 0.015,
    }
