# NSE Swing Trading DSS

Decision support for swing-trading a **portion** of a delivery holding while keeping the core lot intact. Works for **any NSE equity** — pick from the sidebar dropdown (2,100+ symbols from shared cache, or Nifty 50 / Nifty 100 filter).

## Features

- **3-year OHLCV** + Yahoo Finance fundamentals
- **Indicators:** EMA 20/50/200, RSI, rolling VWAP, ATR, volume vs 20d average
- **Buy / sell zones** shaded on the chart (support–resistance + ATR bands)
- **Historical buy/sell signals** and simulated swing-tranche P&L
- **Indicator adherence** — which rules the stock tended to follow over the sample
- **[Backtrader](https://www.backtrader.com/)** capital backtest for independent validation
- **Streamlit** dark-theme UI

## Setup

```bash
cd /Users/admin/Desktop/Codebase/ri/test
source .venv/bin/activate
pip install -r backtest/requirements.txt
```

## Run

```bash
cd backtest
streamlit run app.py
```

## Holding model

| Layer | Default | Purpose |
|-------|---------|---------|
| Core | 4,014 shares (90%) | Never sold in simulation — delivery base |
| Swing | 446 shares (10%) | Rotated on signals; rebuy dips to restore full lot |

Adjust **Total holding** and **Swing tranche %** in the sidebar.

## Signal stack

**Buy (swing add):** Close > EMA20 > EMA50 · RSI ≥ 52 · Volume > 20d avg · Close ≥ VWAP · price in buy zone  

**Sell (swing trim):** EMA20 < EMA50 · RSI ≥ 68 · sell zone · +profit target or ATR stop

## Layout

| Path | Purpose |
|------|---------|
| `app.py` | Streamlit UI |
| `config.py` | Defaults (4460 shares, targets, periods) |
| `data.py` | Yahoo OHLCV + fundamentals |
| `indicators.py` | MA, RSI, VWAP, ATR, S/R |
| `zones.py` | Buy/sell zone bands |
| `signals.py` | Signals + swing-tranche simulation |
| `symbols.py` | NSE symbol picker |
| `backtest_engine.py` | Backtrader strategy |
| `research.py` | Analyst narrative |

Reuses `../eod-swing/eod_swing_lib.py` for Yahoo download and EMA/RSI/S/R helpers.

## Disclaimer

Research and education only — not investment advice.
