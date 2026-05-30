from __future__ import annotations

import sys
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Optional
from zoneinfo import ZoneInfo

import pandas as pd
import yfinance as yf

_ROOT = Path(__file__).resolve().parents[1]
_EOD_SWING = _ROOT / "eod-swing"
if str(_EOD_SWING) not in sys.path:
    sys.path.insert(0, str(_EOD_SWING))

from eod_swing_lib import download_daily_single, to_yahoo_nse  # noqa: E402

IST = ZoneInfo("Asia/Kolkata")


@dataclass
class FundamentalSnapshot:
    as_of: str
    company_name: str = ""
    sector: str = ""
    industry: str = ""
    market_cap_cr: Optional[float] = None
    trailing_pe: Optional[float] = None
    forward_pe: Optional[float] = None
    price_to_book: Optional[float] = None
    profit_margin_pct: Optional[float] = None
    revenue_growth_pct: Optional[float] = None
    earnings_growth_pct: Optional[float] = None
    debt_to_equity: Optional[float] = None
    dividend_yield_pct: Optional[float] = None
    beta: Optional[float] = None
    fifty_two_week_high: Optional[float] = None
    fifty_two_week_low: Optional[float] = None
    notes: list[str] = field(default_factory=list)


def resolve_yahoo_ticker(symbol: str) -> str:
    sym = symbol.strip().upper()
    if sym.endswith(".NS") or sym.endswith(".BO"):
        return sym
    return to_yahoo_nse(sym)


def normalize_ohlcv_index(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    idx = pd.to_datetime(out.index)
    if getattr(idx, "tz", None) is not None:
        idx = idx.tz_convert(IST).tz_localize(None)
    out.index = idx.normalize()
    return out.sort_index()


def fetch_ohlcv(symbol: str, years: int = 3) -> pd.DataFrame:
    yahoo = resolve_yahoo_ticker(symbol)
    end = datetime.now()
    start = end - timedelta(days=int(365.25 * years) + 30)
    df = yf.download(
        yahoo,
        start=start.strftime("%Y-%m-%d"),
        end=end.strftime("%Y-%m-%d"),
        auto_adjust=True,
        progress=False,
    )
    if df.empty:
        df = download_daily_single(yahoo, period=f"{years}y")
    if df.empty:
        raise RuntimeError(f"No OHLCV data for {yahoo}")

    if getattr(df.columns, "nlevels", 1) > 1:
        df.columns = df.columns.get_level_values(0)
    df = df[["Open", "High", "Low", "Close", "Volume"]].copy()
    return normalize_ohlcv_index(df.dropna())


def _safe_float(val: Any) -> Optional[float]:
    try:
        if val is None or (isinstance(val, float) and pd.isna(val)):
            return None
        return float(val)
    except (TypeError, ValueError):
        return None


def fetch_fundamentals(symbol: str) -> FundamentalSnapshot:
    yahoo = resolve_yahoo_ticker(symbol)
    ticker = yf.Ticker(yahoo)
    info: dict[str, Any] = {}
    try:
        info = ticker.info or {}
    except Exception:
        info = {}

    mcap = _safe_float(info.get("marketCap"))
    notes: list[str] = []
    if mcap:
        notes.append(f"Market cap ≈ ₹{mcap / 1e7:,.0f} Cr (Yahoo Finance).")
    pe = _safe_float(info.get("trailingPE"))
    if pe and pe > 35:
        notes.append("Trailing P/E elevated — price may embed high growth expectations.")
    elif pe and pe < 12:
        notes.append("Trailing P/E modest — market may be pricing cyclical downturn or lower ROE.")
    rev_g = _safe_float(info.get("revenueGrowth"))
    if rev_g is not None:
        notes.append(f"Revenue growth (YoY): {rev_g * 100:+.1f}%.")

    return FundamentalSnapshot(
        as_of=datetime.now(IST).strftime("%Y-%m-%d"),
        company_name=str(info.get("longName") or info.get("shortName") or symbol),
        sector=str(info.get("sector") or "—"),
        industry=str(info.get("industry") or "—"),
        market_cap_cr=round(mcap / 1e7, 1) if mcap else None,
        trailing_pe=_safe_float(info.get("trailingPE")),
        forward_pe=_safe_float(info.get("forwardPE")),
        price_to_book=_safe_float(info.get("priceToBook")),
        profit_margin_pct=_safe_float(info.get("profitMargins")),
        revenue_growth_pct=rev_g * 100 if rev_g is not None else None,
        earnings_growth_pct=(
            _safe_float(info.get("earningsGrowth")) * 100
            if _safe_float(info.get("earningsGrowth")) is not None
            else None
        ),
        debt_to_equity=_safe_float(info.get("debtToEquity")),
        dividend_yield_pct=_safe_float(info.get("dividendYield")),
        beta=_safe_float(info.get("beta")),
        fifty_two_week_high=_safe_float(info.get("fiftyTwoWeekHigh")),
        fifty_two_week_low=_safe_float(info.get("fiftyTwoWeekLow")),
        notes=notes,
    )


def fetch_financials_table(symbol: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Annual and quarterly income statement snippets (Yahoo)."""
    yahoo = resolve_yahoo_ticker(symbol)
    ticker = yf.Ticker(yahoo)
    annual = pd.DataFrame()
    quarterly = pd.DataFrame()
    try:
        annual = ticker.financials
        if annual is not None and not annual.empty:
            annual = annual.T.sort_index().tail(4)
    except Exception:
        pass
    try:
        quarterly = ticker.quarterly_financials
        if quarterly is not None and not quarterly.empty:
            quarterly = quarterly.T.sort_index().tail(8)
    except Exception:
        pass
    return annual, quarterly
