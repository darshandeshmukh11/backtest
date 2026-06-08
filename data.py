"""
Yahoo / NSE data + fundamentals for the swing backtest app.

Self-contained for Streamlit Cloud deploy (no parent-folder imports).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, time, timedelta
from typing import Any, Optional
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import yfinance as yf

IST = ZoneInfo("Asia/Kolkata")
NSE_MARKET_OPEN = time(9, 15)
NSE_MARKET_CLOSE = time(15, 30)
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


def ist_now() -> datetime:
    return datetime.now(IST)


def _today_ts() -> pd.Timestamp:
    return pd.Timestamp(ist_now().date())


def market_session_status() -> dict[str, str | bool]:
    """NSE cash session phase for UI banners."""
    now = ist_now()
    weekday = now.weekday()
    t = now.time()
    as_of = now.strftime("%a %d %b %Y, %H:%M IST")

    if weekday >= 5:
        return {"phase": "closed (weekend)", "is_open": False, "as_of": as_of}
    if t < NSE_MARKET_OPEN:
        return {"phase": "pre-open", "is_open": False, "as_of": as_of}
    if t <= NSE_MARKET_CLOSE:
        return {"phase": "open", "is_open": True, "as_of": as_of}
    return {"phase": "closed", "is_open": False, "as_of": as_of}


@dataclass
class LiveQuote:
    symbol: str
    yahoo: str
    price: float
    day_high: float
    day_low: float
    volume: float


def merge_realtime_session(
    df: pd.DataFrame,
    live_price: float,
    *,
    day_high: Optional[float] = None,
    day_low: Optional[float] = None,
    day_volume: Optional[float] = None,
) -> pd.DataFrame:
    """Patch or append today's daily bar with live LTP (and optional session H/L/volume)."""
    out = normalize_ohlcv_index(df)
    if out.empty:
        return out

    live_price = float(live_price)
    today_ts = _today_ts()
    hi = float(day_high) if day_high and day_high > 0 else live_price
    low = float(day_low) if day_low and day_low > 0 else live_price

    if out.index[-1] >= today_ts:
        o = float(out["Open"].iloc[-1])
        h = max(float(out["High"].iloc[-1]), hi, live_price)
        low_val = min(float(out["Low"].iloc[-1]), low, live_price)
        out.iloc[-1, out.columns.get_loc("High")] = h
        out.iloc[-1, out.columns.get_loc("Low")] = low_val
        out.iloc[-1, out.columns.get_loc("Close")] = live_price
        if day_volume is not None and day_volume > 0 and "Volume" in out.columns:
            out.iloc[-1, out.columns.get_loc("Volume")] = float(day_volume)
    else:
        prev_close = float(out["Close"].iloc[-1])
        vol = float(day_volume) if day_volume and day_volume > 0 else 0.0
        out.loc[today_ts] = {
            "Open": prev_close,
            "High": max(prev_close, hi, live_price),
            "Low": min(prev_close, low, live_price),
            "Close": live_price,
            "Volume": vol,
        }
    return out


def has_today_bar(df: pd.DataFrame) -> bool:
    if df.empty:
        return False
    return pd.Timestamp(df.index[-1]).normalize() >= _today_ts()


def _last_completed_bar_idx(df: pd.DataFrame) -> int:
    if len(df) < 2:
        return -1
    if has_today_bar(df):
        return -2
    return -1


def session_is_developing(df: pd.DataFrame, volume_ma_period: int = 20) -> bool:
    """True when today's bar should not drive volume filters yet."""
    if not has_today_bar(df) or len(df) < 2:
        return False

    now = ist_now()
    if now.weekday() < 5 and NSE_MARKET_OPEN <= now.time() <= NSE_MARKET_CLOSE:
        return True

    idx = len(df) - 1
    vol = float(df["Volume"].astype(float).iloc[idx])
    if vol <= 0:
        return True

    avg_series = df["Volume"].astype(float).rolling(volume_ma_period).mean()
    avg_vol = float(avg_series.iloc[idx - 1]) if idx >= 1 else 0.0
    if avg_vol > 0 and vol < avg_vol * 0.25:
        return True
    return False


def _pick_positive(*values: object) -> Optional[float]:
    for val in values:
        if val is None:
            continue
        try:
            num = float(val)
        except (TypeError, ValueError):
            continue
        if num > 0:
            return num
    return None


def fetch_live_quote(nse_symbol: str) -> Optional[LiveQuote]:
    """Latest LTP for an NSE symbol via Yahoo Finance."""
    yahoo = to_yahoo_nse(nse_symbol)
    try:
        ticker = yf.Ticker(yahoo)
        price: Optional[float] = None
        day_high = 0.0
        day_low = 0.0
        volume = 0.0

        try:
            fi = ticker.fast_info
            price = _pick_positive(
                getattr(fi, "last_price", None),
                getattr(fi, "lastPrice", None),
            )
            day_high = float(getattr(fi, "day_high", 0) or getattr(fi, "dayHigh", 0) or price or 0)
            day_low = float(getattr(fi, "day_low", 0) or getattr(fi, "dayLow", 0) or price or 0)
            volume = float(getattr(fi, "last_volume", 0) or getattr(fi, "lastVolume", 0) or 0)
        except Exception:
            pass

        if price is None:
            info = ticker.info or {}
            price = _pick_positive(
                info.get("regularMarketPrice"),
                info.get("currentPrice"),
                info.get("previousClose"),
            )
            day_high = float(info.get("dayHigh") or info.get("regularMarketDayHigh") or price or 0)
            day_low = float(info.get("dayLow") or info.get("regularMarketDayLow") or price or 0)
            volume = float(info.get("volume") or info.get("regularMarketVolume") or 0)

        if price is None:
            hist = ticker.history(period="5d", interval="1d")
            if hist is not None and not hist.empty:
                price = float(hist["Close"].iloc[-1])
                day_high = float(hist["High"].iloc[-1])
                day_low = float(hist["Low"].iloc[-1])
                volume = float(hist["Volume"].iloc[-1]) if "Volume" in hist.columns else 0.0

        if price is None or price <= 0:
            return None

        return LiveQuote(
            symbol=nse_symbol,
            yahoo=yahoo,
            price=round(price, 2),
            day_high=round(day_high or price, 2),
            day_low=round(day_low or price, 2),
            volume=volume,
        )
    except Exception:
        return None


def fetch_ohlcv_live(symbol: str, years: int = 3) -> tuple[pd.DataFrame, Optional[LiveQuote]]:
    """Historical OHLCV with today's bar patched from live LTP."""
    df = fetch_ohlcv(symbol, years)
    quote = fetch_live_quote(symbol)
    if quote and quote.price > 0:
        df = merge_realtime_session(
            df,
            quote.price,
            day_high=quote.day_high,
            day_low=quote.day_low,
            day_volume=quote.volume if quote.volume > 0 else None,
        )
    return df, quote


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
