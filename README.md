# market-scanner-toolkit

A collection of Python scripts I wrote to screen markets, flag intraday setups,
and score news sentiment. These are personal research tools built over several
years: readable and hackable, but not production trading systems.

> **Disclaimer:** For educational and research purposes only. Nothing here is
> financial advice. Signals are heuristic, have not been validated out-of-sample,
> and can lose money. Use at your own risk.

## What's inside

| Folder | Script | What it does |
|---|---|---|
| `scanners/` | `premarket_screener.py` | Ranks tickers by premarket gap %, relative volume, and headline sentiment |
| | `rsi_vwap_5m.py` | Live 5-minute RSI + VWAP scanner with reversal warnings (no API key) |
| | `multi_timeframe_vwap.py` | VWAP/ATR bands + MACD across 1h/15m/5m with a majority-vote signal |
| | `swing_reversal.py` | 2-5 day outlook combining RSI/MACD reversals with news polarity |
| | `earnings_scanner.py` | EPS surprise, guidance headlines, and latest 1-day move |
| | `fundamental_scorer.py` | Scores growth, profitability, balance sheet, valuation and sentiment into a 0-100 fundamental bias |
| `sentiment/` | `sentiment_scanner.py` | Pulls finance RSS feeds, scores with VADER + FinBERT, stores daily bullish/bearish/reversal calls in SQLite |
| `options/` | `options_alert_bot.py` | Engulfing-candle + MACD alerts with low-premium contract selection (alerts only) |
| | `momentum_options_scanner.py` | Momentum scanner with option volume/open-interest filters (alerts only) |
| `backtest/` | `intraday_optimizer.py` | Grid-searches RSI / volume / target-multiple rules on 15-minute bars |

## Setup

```bash
git clone https://github.com/bankyb73/market-scanner-toolkit.git
cd market-scanner-toolkit
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env        # add your own API keys (only for the scripts you run)
```

The sentiment scanner needs extra packages (FinBERT downloads ~400 MB on first run):

```bash
pip install -r requirements-sentiment.txt
```

### Data sources and keys

| Source | Used by | Key variable |
|---|---|---|
| Yahoo Finance via `yfinance` | premarket, 5m RSI/VWAP, options scripts | none |
| [Twelve Data](https://twelvedata.com) | multi-timeframe VWAP, swing reversal, earnings, backtest | `TWELVE_DATA_API_KEY` |
| [Finnhub](https://finnhub.io) | earnings, fundamentals | `FINNHUB_API_KEY` |
| [NewsAPI](https://newsapi.org) | swing reversal | `NEWS_API_KEY` |

Free tiers are rate-limited (Twelve Data: 8 requests/minute), so the scripts
include sleeps and batching. Some Finnhub endpoints (price targets, news
sentiment) may require a paid plan; `fundamental_scorer.py` degrades gracefully
when one is unavailable.

## Usage

```bash
python scanners/premarket_screener.py
python scanners/rsi_vwap_5m.py                  # prompts for a ticker
python scanners/fundamental_scorer.py AAPL MSFT
python scanners/fundamental_scorer.py --keys AAPL   # list Finnhub metric fields
python sentiment/sentiment_scanner.py
python options/options_alert_bot.py
python backtest/intraday_optimizer.py
```

Edit the `TICKERS` / `WATCHLIST` list at the top of each script to suit you.

## Known limitations

- Indicators are simple and untuned. The backtester picks parameters by
  in-sample PnL, so treat its results as optimistic.
- Backtests simulate the underlying only and ignore commissions, slippage, and
  options pricing. If a bar touches both stop and target, the stop is assumed
  to hit first.
- Free data feeds can be delayed, incomplete, or change format without notice
  (RSS URLs and Yahoo responses especially).
- The fundamental score uses hand-picked thresholds; it is a screening
  heuristic, not a valuation.
- Alert bots notify only; they never place orders. The options scripts use the
  nearest expiry by default, which can be same-day.

## Security

API keys and email credentials are read from environment variables / `.env`
(git-ignored). Never commit real keys. If you fork this and accidentally push
one, rotate it immediately.

## License

MIT. See [LICENSE](LICENSE).
