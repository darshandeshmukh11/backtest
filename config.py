from __future__ import annotations

from dataclasses import dataclass


@dataclass
class DSSConfig:
    symbol: str = "JINDALSTEL"
    years: int = 3
    core_holding_qty: int = 4460
    swing_pct: float = 10.0
    profit_target_pct: float = 2.5
    stop_atr_mult: float = 1.5
    ema_fast: int = 20
    ema_slow: int = 50
    rsi_period: int = 14
    rsi_buy_min: float = 52.0
    rsi_sell_max: float = 68.0
    vol_ma_period: int = 20
    vwap_period: int = 20
    atr_period: int = 14
    sr_lookback: int = 120
    zone_atr_mult: float = 0.35
    backtest_cash: float = 500_000.0
    commission: float = 0.001
    use_realtime: bool = False

    @property
    def swing_qty(self) -> int:
        return max(1, int(round(self.core_holding_qty * self.swing_pct / 100)))

    @property
    def core_qty(self) -> int:
        return max(0, self.core_holding_qty - self.swing_qty)

    @property
    def yahoo_ticker(self) -> str:
        from data import resolve_yahoo_ticker

        return resolve_yahoo_ticker(self.symbol)
