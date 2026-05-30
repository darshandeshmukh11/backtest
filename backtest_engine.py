from __future__ import annotations

from dataclasses import dataclass, field

import backtrader as bt
import backtrader.indicators as btind
import pandas as pd

from config import DSSConfig


@dataclass
class BacktraderSummary:
    start_value: float
    end_value: float
    total_return_pct: float
    max_drawdown_pct: float
    sharpe: float | None
    closed_trades: int
    won_trades: int
    notes: list[str] = field(default_factory=list)


class SwingTrancheStrategy(bt.Strategy):
    """
    Long-only swing on a fraction of capital — mirrors DSS signal stack:
    EMA20>EMA50, RSI>52, volume>MA, close near VWAP proxy (EMA20).
    Exit: EMA cross down, RSI>68, or profit target.
    """

    params = dict(
        ema_fast=20,
        ema_slow=50,
        rsi_period=14,
        rsi_min=52.0,
        rsi_exit=68.0,
        vol_ma=20,
        profit_pct=2.5,
        stop_atr_mult=1.5,
        swing_pct=95.0,
    )

    def __init__(self) -> None:
        self.ema_fast = btind.ExponentialMovingAverage(self.data.close, period=self.p.ema_fast)
        self.ema_slow = btind.ExponentialMovingAverage(self.data.close, period=self.p.ema_slow)
        self.rsi = btind.RelativeStrengthIndex(self.data.close, period=self.p.rsi_period)
        self.vol_ma = btind.SimpleMovingAverage(self.data.volume, period=self.p.vol_ma)
        self.atr = btind.ATR(self.data, period=14)
        self.entry_price: float | None = None

    def next(self) -> None:
        if not self.position:
            trend = self.data.close[0] > self.ema_fast[0] > self.ema_slow[0]
            momentum = self.rsi[0] >= self.p.rsi_min
            volume = self.data.volume[0] > self.vol_ma[0]
            if trend and momentum and volume:
                self.buy(size=self.broker.getcash() * self.p.swing_pct / 100 / self.data.close[0])
                self.entry_price = float(self.data.close[0])
        else:
            assert self.entry_price is not None
            target = self.entry_price * (1 + self.p.profit_pct / 100)
            stop = self.entry_price - self.p.stop_atr_mult * float(self.atr[0])
            exit_trend = self.ema_fast[0] < self.ema_slow[0]
            exit_rsi = self.rsi[0] >= self.p.rsi_exit
            hit_target = self.data.high[0] >= target
            hit_stop = self.data.low[0] <= stop
            if hit_target or hit_stop or exit_trend or exit_rsi:
                self.close()
                self.entry_price = None


def run_backtrader(
    df: pd.DataFrame,
    cfg: DSSConfig,
) -> BacktraderSummary:
    cerebro = bt.Cerebro()
    cerebro.addstrategy(
        SwingTrancheStrategy,
        ema_fast=cfg.ema_fast,
        ema_slow=cfg.ema_slow,
        rsi_period=cfg.rsi_period,
        rsi_min=cfg.rsi_buy_min,
        rsi_exit=cfg.rsi_sell_max,
        vol_ma=cfg.vol_ma_period,
        profit_pct=cfg.profit_target_pct,
        stop_atr_mult=cfg.stop_atr_mult,
    )
    feed_df = df[["Open", "High", "Low", "Close", "Volume"]].copy()
    cerebro.adddata(bt.feeds.PandasData(dataname=feed_df))
    cerebro.broker.setcash(cfg.backtest_cash)
    cerebro.broker.setcommission(commission=cfg.commission)
    cerebro.addsizer(bt.sizers.PercentSizer, percents=95)

    cerebro.addanalyzer(bt.analyzers.Returns, _name="returns")
    cerebro.addanalyzer(bt.analyzers.DrawDown, _name="drawdown")
    cerebro.addanalyzer(bt.analyzers.SharpeRatio, _name="sharpe", riskfreerate=0.0)
    cerebro.addanalyzer(bt.analyzers.TradeAnalyzer, _name="trades")

    start = cfg.backtest_cash
    results = cerebro.run()
    strat = results[0]
    end = cerebro.broker.getvalue()
    dd = strat.analyzers.drawdown.get_analysis()
    sharpe_a = strat.analyzers.sharpe.get_analysis()
    trades_a = strat.analyzers.trades.get_analysis()

    closed = trades_a.get("total", {}).get("closed", 0) if trades_a else 0
    won = trades_a.get("won", {}).get("total", 0) if trades_a else 0
    sharpe_val = sharpe_a.get("sharperatio") if sharpe_a else None

    return BacktraderSummary(
        start_value=start,
        end_value=end,
        total_return_pct=(end / start - 1) * 100,
        max_drawdown_pct=float(dd.max.drawdown) if dd else 0.0,
        sharpe=float(sharpe_val) if sharpe_val is not None else None,
        closed_trades=int(closed),
        won_trades=int(won),
        notes=[
            "Backtrader validates the signal stack on a capital pool (not share-count model).",
            "See pandas swing-tranche simulation for share-count rotation logic.",
        ],
    )
