"""
momentum_options_scanner.py

Scans a watchlist on 5-minute bars for intraday momentum (>= +/-2% from the
open) confirmed by MACD and proximity to VWAP, filters on stock price and
average volume, then suggests the at-the-money contract that clears option
volume / open-interest minimums. You confirm entries at the prompt; confirmed
trades are tracked and flagged on a large move in the underlying.

It does NOT place orders. Research/education only; not financial advice.
"""
import yfinance as yf
import time
from datetime import datetime
import pytz

# === USER SETTINGS ===
TICKERS = ["AMD", "INTC", "F", "PLTR", "SNAP", "T"]  # replace with your list
VOLUME_MIN = 10_000_000
VOLUME_MAX = 50_000_000
OPTION_MIN_VOL = 1500
SCAN_INTERVAL = 30  # seconds between full scans
MAX_VWAP_DISTANCE = 0.3  # max $ distance from VWAP to allow an entry

# === MARKET TIME CONTROL (US Eastern Time) ===
def is_market_open():
    now = datetime.now(pytz.timezone("US/Eastern"))
    open_ = now.replace(hour=9, minute=30, second=0, microsecond=0)
    close_ = now.replace(hour=16, minute=0, second=0, microsecond=0)
    return now.weekday() < 5 and open_ <= now <= close_

# State tracking
active_trades = {}

def calculate_macd(data):
    exp1 = data['Close'].ewm(span=12, adjust=False).mean()
    exp2 = data['Close'].ewm(span=26, adjust=False).mean()
    macd = exp1 - exp2
    signal = macd.ewm(span=9, adjust=False).mean()
    histogram = macd - signal
    return macd, signal, histogram

def calculate_vwap(data):
    vwap = (data['Volume'] * (data['High'] + data['Low'] + data['Close']) / 3).cumsum() / data['Volume'].cumsum()
    return vwap

def get_option_signal(ticker, side):
    try:
        stock = yf.Ticker(ticker)
        options_dates = stock.options
        if not options_dates:
            return None
        time.sleep(1.5)  # delay to avoid rate limits
        chain = stock.option_chain(options_dates[0])
        options = chain.calls if side == "call" else chain.puts
    except Exception as e:
        print(f"[{ticker}] Options fetch error: {e}")
        return None

    try:
        spot = stock.history(period="1d")["Close"].iloc[-1]
    except Exception:
        return None

    options['Distance'] = abs(options['strike'] - spot)
    filtered = options[(options['volume'] > OPTION_MIN_VOL) & (options['openInterest'] > OPTION_MIN_VOL)]
    if filtered.empty:
        return None
    best = filtered.sort_values("Distance").iloc[0]
    return {
        "strike": best['strike'],
        "ask": best['ask'],
        "bid": best['bid'],
        "symbol": best['contractSymbol']
    }

def scan_ticker(ticker):
    try:
        data = yf.download(ticker, period="1d", interval="5m", progress=False, threads=False,
                           multi_level_index=False)
        if data.empty:
            return

        current_price = data['Close'].iloc[-1]
        open_price = data['Open'].iloc[0]
        change_pct = ((current_price - open_price) / open_price) * 100

        stock_info = yf.Ticker(ticker).info
        time.sleep(1.2)
        avg_daily_volume = stock_info.get("averageVolume", 0)
        if avg_daily_volume < VOLUME_MIN or avg_daily_volume > VOLUME_MAX:
            return

        if not (5 <= current_price <= 150):
            return

        if change_pct >= 2:
            signal_type = "call"
        elif change_pct <= -2:
            signal_type = "put"
        else:
            return

        macd, signal, histogram = calculate_macd(data)
        vwap = calculate_vwap(data)

        # For CALL
        if signal_type == "call":
            if not (macd.iloc[-1] > signal.iloc[-1] and histogram.iloc[-1] > 0 and histogram.iloc[-1] > histogram.iloc[-2]):
                return
            if abs(current_price - vwap.iloc[-1]) > MAX_VWAP_DISTANCE:
                return

        # For PUT
        if signal_type == "put":
            if not (macd.iloc[-1] < signal.iloc[-1] and histogram.iloc[-1] < 0 and histogram.iloc[-1] < histogram.iloc[-2]):
                return
            if abs(current_price - vwap.iloc[-1]) > MAX_VWAP_DISTANCE:
                return

        option_data = get_option_signal(ticker, signal_type)
        if option_data:
            log_trade(ticker, signal_type, option_data, current_price)

    except Exception as e:
        print(f"[{ticker}] Error: {e}")
        if "429" in str(e) or "Too Many Requests" in str(e):
            print("Rate limit hit. Sleeping 60 seconds...")
            time.sleep(60)

def log_trade(ticker, signal_type, option_data, price):
    now = datetime.now().strftime("%H:%M:%S")
    print(f"\n[{now}] Signal Detected: {ticker.upper()} | {signal_type.upper()} | Spot: ${price:.2f}")
    print(f"Option: {option_data['symbol']} | Strike: {option_data['strike']} | Bid: {option_data['bid']} | Ask: {option_data['ask']}")

    user_input = input(f"Confirm entry into {ticker.upper()} {signal_type.upper()}? (y/n): ").strip().lower()
    if user_input == 'y':
        active_trades[ticker] = {
            "type": signal_type,
            "entry": price
        }
        print(f"Trade for {ticker} marked as ACTIVE.")

def monitor_active_trades():
    to_remove = []
    for ticker, trade in active_trades.items():
        try:
            data = yf.download(ticker, period="1d", interval="5m", progress=False, threads=False,
                           multi_level_index=False)
            if data.empty:
                continue
            current_price = data['Close'].iloc[-1]
            entry = trade['entry']
            change_pct = ((current_price - entry) / entry) * 100
            print(f"[{ticker}] {trade['type'].upper()} | Entry: ${entry:.2f} | Current: ${current_price:.2f} | Change: {change_pct:.2f}%")

            if abs(change_pct) > 25:
                print(f"[{ticker}] Exiting trade due to 25% move.")
                to_remove.append(ticker)
        except Exception:
            continue
    for t in to_remove:
        del active_trades[t]

def main_loop():
    print("=== Trading Bot Started ===")
    while True:
        if not is_market_open():
            print("Market is closed. Waiting...")
            time.sleep(60)
            continue

        for ticker in TICKERS:
            if ticker in active_trades:
                continue
            scan_ticker(ticker)
            time.sleep(3)  # delay between tickers
        monitor_active_trades()
        time.sleep(SCAN_INTERVAL)

if __name__ == "__main__":
    main_loop()
