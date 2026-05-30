"""NSE symbol universe for the swing DSS stock picker."""

from __future__ import annotations

import json
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
_EOD_SWING = _ROOT / "eod-swing"
if str(_EOD_SWING) not in sys.path:
    sys.path.insert(0, str(_EOD_SWING))

from eod_swing_lib import get_nifty50_and_100_universe, get_nifty50_symbols  # noqa: E402

NSE_CACHE_PATH = _ROOT / "paper-trade" / "data" / "nse_equity_symbols.json"

POPULAR_NSE_SYMBOLS: list[str] = [
    "RELIANCE",
    "TCS",
    "INFY",
    "HDFCBANK",
    "ICICIBANK",
    "SBIN",
    "BHARTIARTL",
    "ITC",
    "KOTAKBANK",
    "LT",
    "AXISBANK",
    "TATASTEEL",
    "TATAMOTORS",
    "JINDALSTEL",
    "WIPRO",
    "MARUTI",
    "SUNPHARMA",
    "HINDUNILVR",
    "BAJFINANCE",
    "ADANIENT",
    "M&M",
]

FALLBACK_SYMBOLS: list[str] = sorted(set(POPULAR_NSE_SYMBOLS))


def normalize_symbol(symbol: str) -> str:
    return symbol.strip().upper().replace(".NS", "").replace(".BO", "")


def load_nse_equity_symbols() -> list[str]:
    """All NSE EQ symbols from shared monorepo cache, else NIFTY union fallback."""
    if NSE_CACHE_PATH.exists():
        try:
            payload = json.loads(NSE_CACHE_PATH.read_text(encoding="utf-8"))
            symbols = payload.get("symbols") or []
            if len(symbols) >= 100:
                return sorted({normalize_symbol(s) for s in symbols})
        except Exception:
            pass
    try:
        all_syms, _ = get_nifty50_and_100_universe(prefer_live=False)
        if all_syms:
            return sorted(set(all_syms) | set(FALLBACK_SYMBOLS))
    except Exception:
        pass
    return list(FALLBACK_SYMBOLS)


def universe_for_index(index_filter: str, nse_universe: list[str]) -> list[str]:
    if index_filter == "Nifty 50":
        return get_nifty50_symbols(prefer_live=False)
    if index_filter == "Nifty 100":
        all_syms, _ = get_nifty50_and_100_universe(prefer_live=False)
        return all_syms
    return nse_universe


def search_symbols(query: str, universe: list[str], limit: int = 200) -> list[str]:
    q = query.strip().upper()
    if not q:
        popular = [s for s in POPULAR_NSE_SYMBOLS if s in set(universe)]
        return popular[:limit]
    exact = [s for s in universe if s == q]
    prefix = [s for s in universe if s.startswith(q) and s != q]
    contains = [s for s in universe if q in s and not s.startswith(q)]
    out: list[str] = []
    for group in (exact, prefix, contains):
        for s in group:
            if s not in out:
                out.append(s)
            if len(out) >= limit:
                return out
    if q.replace("-", "").isalnum() and q not in out:
        out.insert(0, q)
    return out[:limit]


def symbol_picker_options(query: str, universe: list[str]) -> list[str]:
    q = query.strip().upper()
    if not q:
        popular = [s for s in POPULAR_NSE_SYMBOLS if s in set(universe)]
        if popular:
            rest = [s for s in universe if s not in popular]
            return popular + rest[: max(0, 100 - len(popular))]
        return universe[:100]
    return search_symbols(q, universe, limit=300)


def default_symbol_index(options: list[str], preferred: str = "JINDALSTEL") -> int:
    pref = normalize_symbol(preferred)
    if pref in options:
        return options.index(pref)
    return 0
