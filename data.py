"""
Yahoo / NSE data + fundamentals for the swing backtest app.

Self-contained for Streamlit Cloud deploy (no parent-folder imports).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Optional
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import yfinance as yf

IST = ZoneInfo("Asia/Kolkata")
_OHLCV_COLS = ("Open", "High", "Low", "Close", "Volume")

NIFTY_50_FALLBACK: list[str] = [
    "ADANIENT", "ADANIPORTS", "APOLLOHOSP", "ASIANPAINT", "AXISBANK", "BAJAJ-AUTO",
    "BAJFINANCE", "BAJAJFINSV", "BEL", "BHARTIARTL", "CIPLA", "COALINDIA", "DRREDDY",
    "EICHERMOT", "ETERNAL", "GRASIM", "HCLTECH", "HDFCBANK", "HDFCLIFE", "HINDALCO",
    "HINDUNILVR", "HINDZINC", "ICICIBANK", "INDIGO", "INFY", "ITC", "JIOFIN", "JSWSTEEL",
    "KOTAKBANK", "LT", "M&M", "MARUTI", "NESTLEIND", "NTPC", "ONGC", "POWERGRID",
    "RELIANCE", "SBILIFE", "SBIN", "SHRIRAMFIN", "SUNPHARMA", "TATACONSUM", "TATAMOTORS",
    "TATASTEEL", "TCS", "TECHM", "TITAN", "TRENT", "ULTRACEMCO", "WIPRO",
]

NSE_SYMBOL_RENAMES: dict[str, str] = {"ZOMATO": "ETERNAL"}
YAHOO_TICKER_ALIASES: dict[str, str] = {"TATAMOTORS": "TMPV.NS", "ZOMATO": "ETERNAL.NS"}

NIFTY_100_EXTRA_FALLBACK: list[str] = [
    "ABB", "ADANIGREEN", "ADANIPOWER", "AMBUJACEM", "DMART", "GAIL", "HAL", "HAVELLS",
    "ICICIPRULI", "INDUSTOWER", "IOC", "IRFC", "JINDALSTEL", "LICI", "LODHA", "NAUKRI",
    "PIDILITIND", "PNB", "SIEMENS", "VEDL",
]


def normalize_nse_symbol(symbol: str) -> str:
    key = symbol.strip().upper()
    return NSE_SYMBOL_RENAMES.get(key, key)


def to_yahoo_nse(symbol: str) -> str:
    raw = symbol.strip().upper()
    key = normalize_nse_symbol(raw)
    if raw in YAHOO_TICKER_ALIASES:
        return YAHOO_TICKER_ALIASES[raw]
    return f"{key}.NS"


def _symbols_from_wikipedia() -> list[str]:
    tables = pd.read_html("https://en.wikipedia.org/wiki/NIFTY_50")
    for table in tables:
        cols = {str(c).lower(): c for c in table.columns}
        symbol_col = next((cols[k] for k in ("symbol", "ticker", "nse symbol") if k in cols), None)
        if symbol_col is None:
            continue
        symbols = (
            table[symbol_col].astype(str).str.strip().str.upper()
            .replace({"NAN": None, "": None}).dropna().tolist()
        )
        symbols = [s for s in symbols if s.isalnum() or "-" in s]
        if len(symbols) >= 45:
            return sorted({normalize_nse_symbol(s) for s in symbols})
    raise ValueError("Could not parse NIFTY 50 symbols from Wikipedia")


def _symbols_from_wikipedia_title(title: str, min_count: int = 45) -> list[str]:
    tables = pd.read_html(f"https://en.wikipedia.org/wiki/{title.replace(' ', '_')}")
    for table in tables:
        cols = {str(c).lower(): c for c in table.columns}
        symbol_col = next((cols[k] for k in ("symbol", "ticker", "nse symbol") if k in cols), None)
        if symbol_col is None:
            continue
        symbols = (
            table[symbol_col].astype(str).str.strip().str.upper()
            .replace({"NAN": None, "": None}).dropna().tolist()
        )
        symbols = [s for s in symbols if s.isalnum() or "-" in s]
        if len(symbols) >= min_count:
            return sorted({normalize_nse_symbol(s) for s in symbols})
    raise ValueError(f"Could not parse symbols from Wikipedia: {title}")


def get_nifty50_symbols(prefer_live: bool = True) -> list[str]:
    if prefer_live:
        try:
            return _symbols_from_wikipedia()
        except Exception:
            pass
    return sorted({normalize_nse_symbol(s) for s in NIFTY_50_FALLBACK})


def get_nifty100_symbols(prefer_live: bool = True) -> list[str]:
    if prefer_live:
        try:
            return _symbols_from_wikipedia_title("NIFTY 100", min_count=90)
        except Exception:
            pass
    return sorted(
        {normalize_nse_symbol(s) for s in NIFTY_50_FALLBACK}
        | {normalize_nse_symbol(s) for s in NIFTY_100_EXTRA_FALLBACK}
    )


def get_nifty50_and_100_universe(prefer_live: bool = True) -> tuple[list[str], set[str]]:
    n50 = get_nifty50_symbols(prefer_live=prefer_live)
    n100 = get_nifty100_symbols(prefer_live=prefer_live)
    n50_set = {normalize_nse_symbol(s) for s in n50}
    return sorted(n50_set | {normalize_nse_symbol(s) for s in n100}), n50_set


def _trim_ohlcv(frame: pd.DataFrame) -> pd.DataFrame:
    if isinstance(frame.columns, pd.MultiIndex):
        frame.columns = frame.columns.get_level_values(-1)
    frame.columns = [str(c) for c in frame.columns]
    keep = [c for c in _OHLCV_COLS if c in frame.columns]
    if not keep:
        return pd.DataFrame()
    return frame[keep].dropna(how="all")


def download_daily_single(yahoo_ticker: str, period: str) -> pd.DataFrame:
    df = yf.download(yahoo_ticker, period=period, interval="1d", auto_adjust=False, progress=False)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    return _trim_ohlcv(df).dropna()


def compute_ema(close: pd.Series, period: int) -> pd.Series:
    return close.astype(float).ewm(span=period, adjust=False).mean()


def compute_rsi(close: pd.Series, period: int = 14) -> pd.Series:
    delta = close.astype(float).diff()
    gain = delta.clip(lower=0)
    loss = (-delta).clip(lower=0)
    avg_gain = gain.ewm(alpha=1 / period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / period, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def compute_atr_scalar(df: pd.DataFrame, period: int = 14) -> float:
    high_low = df["High"] - df["Low"]
    high_close = (df["High"] - df["Close"].shift(1)).abs()
    low_close = (df["Low"] - df["Close"].shift(1)).abs()
    true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    atr = true_range.rolling(period).mean()
    val = float(atr.iloc[-1]) if len(atr) and not np.isnan(atr.iloc[-1]) else np.nan
    return val


def find_swing_points(df: pd.DataFrame, lookback: int = 3) -> tuple[list[float], list[float]]:
    highs = df["High"].to_numpy()
    lows = df["Low"].to_numpy()
    swing_highs: list[float] = []
    swing_lows: list[float] = []
    for i in range(lookback, len(df) - lookback):
        if highs[i] == np.max(highs[i - lookback : i + lookback + 1]):
            swing_highs.append(float(highs[i]))
        if lows[i] == np.min(lows[i - lookback : i + lookback + 1]):
            swing_lows.append(float(lows[i]))
    return swing_highs, swing_lows


def cluster_levels(levels: list[float], tolerance: float) -> list[float]:
    if not levels:
        return []
    sorted_levels = sorted(levels)
    clusters: list[list[float]] = [[sorted_levels[0]]]
    for level in sorted_levels[1:]:
        if abs(level - np.mean(clusters[-1])) <= tolerance:
            clusters[-1].append(level)
        else:
            clusters.append([level])
    return [round(float(np.mean(c)), 2) for c in clusters]


def previous_day_pivots(df: pd.DataFrame) -> list[float]:
    if len(df) < 2:
        return []
    prev = df.iloc[-2]
    pivot = (prev["High"] + prev["Low"] + prev["Close"]) / 3.0
    r1 = 2 * pivot - prev["Low"]
    s1 = 2 * pivot - prev["High"]
    r2 = pivot + (prev["High"] - prev["Low"])
    s2 = pivot - (prev["High"] - prev["Low"])
    return [round(x, 2) for x in [pivot, r1, s1, r2, s2]]


def infer_support_resistance(
    df: pd.DataFrame,
    current_price: float,
    lookback_days: int,
) -> tuple[float, float, float]:
    atr_value = compute_atr_scalar(df, 14)
    if np.isnan(atr_value):
        atr_value = current_price * 0.01
    tolerance = max(atr_value * 0.35, current_price * 0.004)
    recent = df.tail(lookback_days)
    swing_highs, swing_lows = find_swing_points(recent, lookback=3)
    pivots = previous_day_pivots(df)
    support_candidates = cluster_levels(swing_lows + [p for p in pivots if p < current_price], tolerance)
    resistance_candidates = cluster_levels(swing_highs + [p for p in pivots if p > current_price], tolerance)
    supports = sorted([x for x in support_candidates if x < current_price], reverse=True)
    resistances = sorted([x for x in resistance_candidates if x > current_price])
    nearest_support = supports[0] if supports else current_price * 0.95
    nearest_resistance = resistances[0] if resistances else current_price * 1.08
    return nearest_support, nearest_resistance, atr_value


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
