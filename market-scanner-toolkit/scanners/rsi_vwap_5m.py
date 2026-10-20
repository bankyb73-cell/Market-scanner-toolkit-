# file: realtime_5m_rsi_vwap_scanner.py
# Usage: python realtime_5m_rsi_vwap_scanner.py
# Requires: pip install yfinance pandas numpy

import time, datetime as dt
from typing import Tuple
import numpy as np
import pandas as pd
import yfinance as yf

POLL_SECONDS = 60          # how often to refresh data
RSI_LENGTH   = 14
OVERSOLD     = 30
OVERBOUGHT   = 70
REV_LO       = 25          # "imminent reversal" lower threshold
REV_HI       = 75          # "imminent reversal" upper threshold
NEAR_VWAP    = 0.005       # 0.5% proximity to VWAP
EXT_VWAP     = 0.015       # 1.5% extension from VWAP for warnings

def rsi(series: pd.Series, length: int = 14) -> pd.Series:
    delta = series.diff()
    up = np.where(delta > 0, delta, 0.0)
    down = np.where(delta < 0, -delta, 0.0)
    roll_up = pd.Series(up, index=series.index).ewm(alpha=1/length, adjust=False).mean()
    roll_down = pd.Series(down, index=series.index).ewm(alpha=1/length, adjust=False).mean()
    rs = roll_up / (roll_down.replace(0, np.nan))
    rsi = 100 - (100 / (1 + rs))
    return rsi.fillna(50)

def intraday_vwap(df: pd.DataFrame) -> pd.Series:
    # VWAP resets each session; compute per trading day
    tp = (df['High'] + df['Low'] + df['Close']) / 3
    vol = df['Volume']
    day = df.index.tz_convert(None).date  # group by calendar date
    cum_pv = tp.mul(vol).groupby(day).cumsum()
    cum_v  = vol.groupby(day).cumsum().replace(0, np.nan)
    return cum_pv / cum_v

def build_5m(df_1m: pd.DataFrame) -> pd.DataFrame:
    ohlc = {
        'Open':  'first',
        'High':  'max',
        'Low':   'min',
        'Close': 'last',
        'Volume':'sum'
    }
    df5 = df_1m.resample('5min', label='right', closed='right').agg(ohlc).dropna()
    df5['VWAP'] = intraday_vwap(df5)
    df5['RSI'] = rsi(df5['Close'], RSI_LENGTH)
    return df5

def crosses(series: pd.Series, level: float) -> Tuple[bool, str]:
    """Returns (crossed, direction) where direction is 'up' or 'down'"""
    if len(series) < 2: return (False, '')
    prev, curr = series.iloc[-2], series.iloc[-1]
    if prev < level <= curr:  return (True, 'up')
    if prev > level >= curr:  return (True, 'down')
    return (False, '')

def pct_from_vwap(close: float, vwap: float) -> float:
    return 0.0 if vwap == 0 else (close - vwap) / vwap

def format_now(ts) -> str:
    # Display in local time for clarity
    return ts.tz_convert(None).strftime('%Y-%m-%d %H:%M')

def scan_once(ticker: str, last_bar_seen=None, last_signal_state: dict=None):
    if last_signal_state is None:
        last_signal_state = {}

    # pull last 2 trading days of 1m candles, then make 5m
    data_1m = yf.download(
        tickers=ticker,
        period='2d',
        interval='1m',
        auto_adjust=False,
        progress=False,
        multi_level_index=False,
    )
    if data_1m.empty:
        print("No data returned. Market closed or bad ticker?")
        return last_bar_seen, last_signal_state

    df5 = build_5m(data_1m)
    if df5.empty:
        print("Not enough data for 5m bars yet.")
        return last_bar_seen, last_signal_state

    # work on the latest completed 5m bar
    last_idx = df5.index[-1]
    if last_bar_seen is not None and last_idx == last_bar_seen:
        # nothing new yet
        return last_bar_seen, last_signal_state

    row = df5.iloc[-1]
    ts = last_idx
    close = float(row['Close'])
    vwap = float(row['VWAP'])
    rsi_now = float(row['RSI'])
    dist = pct_from_vwap(close, vwap)
    msg_time = format_now(ts)

    # determine crosses
    rsi_series = df5['RSI']
    crossed_50, dir_50 = crosses(rsi_series, 50)
    crossed_30, dir_30 = crosses(rsi_series, OVERSOLD)
    crossed_70, dir_70 = crosses(rsi_series, OVERBOUGHT)

    signals = []

    # BULLISH zones
    if (crossed_50 and dir_50 == 'up' and close > vwap) \
       or (crossed_30 and dir_30 == 'up' and abs(dist) <= NEAR_VWAP):
        signals.append(("BULLISH",
                        f"{ticker} {msg_time}  Close={close:.2f}  VWAP={vwap:.2f}  RSI={rsi_now:.1f}  ΔVWAP={dist*100:.2f}%"))

    # BEARISH zones
    if (crossed_50 and dir_50 == 'down' and close < vwap) \
       or (crossed_70 and dir_70 == 'down' and abs(dist) <= NEAR_VWAP):
        signals.append(("BEARISH",
                        f"{ticker} {msg_time}  Close={close:.2f}  VWAP={vwap:.2f}  RSI={rsi_now:.1f}  ΔVWAP={dist*100:.2f}%"))

    # REVERSAL WARNINGS (heuristics):
    # 1) Extended from VWAP + extreme RSI
    if (rsi_now >= REV_HI and dist > EXT_VWAP):
        signals.append(("REVERSAL ⚠ (downside risk)",
                        f"{ticker} {msg_time}  Overbought & >{EXT_VWAP*100:.1f}% above VWAP. Watch for fade / lower high. "
                        f"Close={close:.2f} VWAP={vwap:.2f} RSI={rsi_now:.1f} ΔVWAP={dist*100:.2f}%"))
    if (rsi_now <= REV_LO and dist < -EXT_VWAP):
        signals.append(("REVERSAL ⚠ (upside risk)",
                        f"{ticker} {msg_time}  Oversold & >{EXT_VWAP*100:.1f}% below VWAP. Watch for bounce / higher low. "
                        f"Close={close:.2f} VWAP={vwap:.2f} RSI={rsi_now:.1f} ΔVWAP={dist*100:.2f}%"))

    # 2) RSI rolling back through extremes (early momentum loss), confirm with VWAP side
    if crossed_70 and dir_70 == 'down' and close < vwap:
        signals.append(("REVERSAL ⚠ (momentum cooling)",
                        f"{ticker} {msg_time}  RSI fell back under 70 while below VWAP. Potential topping. "
                        f"Close={close:.2f} VWAP={vwap:.2f} RSI={rsi_now:.1f}"))
    if crossed_30 and dir_30 == 'up' and close > vwap:
        signals.append(("REVERSAL ⚠ (momentum turning up)",
                        f"{ticker} {msg_time}  RSI rose back above 30 while above VWAP. Potential bottoming. "
                        f"Close={close:.2f} VWAP={vwap:.2f} RSI={rsi_now:.1f}"))

    # de-duplicate per bar + signal type
    logname = f"signals_{ticker}_{dt.date.today().isoformat()}.log"
    for level, text in signals:
        key = (str(last_idx), level)
        if last_signal_state.get(key):  # already printed for this bar
            continue
        line = f"[{level}] {text}"
        print(line, flush=True)
        with open(logname, "a", encoding="utf-8") as f:
            f.write(line + "\n")
        last_signal_state[key] = True

    return last_idx, last_signal_state

def main():
    # Prompt the user for a ticker symbol
    ticker = input("Enter the ticker symbol you want to scan (e.g., AAPL, TSLA): ").upper().strip()
    if not ticker:
        print("No ticker entered. Exiting.")
        return

    print(f"Starting 5m RSI/VWAP scanner for {ticker}. Ctrl+C to stop.")
    last_bar_seen = None
    last_signal_state = {}
    try:
        while True:
            last_bar_seen, last_signal_state = scan_once(ticker, last_bar_seen, last_signal_state)
            time.sleep(POLL_SECONDS)
    except KeyboardInterrupt:
        print("\nStopped.")

if __name__ == "__main__":
    main()
