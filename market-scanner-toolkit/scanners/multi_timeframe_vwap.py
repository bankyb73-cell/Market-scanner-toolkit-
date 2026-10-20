"""
multi_timeframe_vwap.py

Scores momentum and reversal setups from VWAP +/- ATR bands and MACD on three
timeframes (1h / 15m / 5m), then reports a majority-vote decision and prints an
alert whenever it changes. Uses Twelve Data (free tier: 8 requests/minute).

Note: the VWAP comes from the `ta` library, which uses a rolling window
(default 14 bars) rather than a session-anchored VWAP.
Research/education only; not financial advice.
"""
import requests
import pandas as pd
import ta
import time
import os
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:  # python-dotenv is optional
    pass

# 🔑 Your Twelve Data API Key
API_KEY = os.getenv("TWELVE_DATA_API_KEY", "")

# Store last signals to avoid repeat alerts
last_signals = {}

# ----------------------------
# 🔹 Fetch OHLCV Data
# ----------------------------
def get_ohlcv(ticker, interval, outputsize=200):
    url = f"https://api.twelvedata.com/time_series?symbol={ticker}&interval={interval}&outputsize={outputsize}&apikey={API_KEY}"
    resp = requests.get(url).json()
    if "values" not in resp:
        print(f"❌ Error fetching {interval} data for {ticker}: {resp}")
        return None

    df = pd.DataFrame(resp["values"])
    df["datetime"] = pd.to_datetime(df["datetime"])
    df = df.sort_values("datetime")
    for col in ["open", "high", "low", "close", "volume"]:
        df[col] = pd.to_numeric(df[col])
    return df

# ----------------------------
# 🔹 VWAP with Bands
# ----------------------------
def add_vwap_bands(df, band_mult=0.5):
    vwap = ta.volume.VolumeWeightedAveragePrice(
        high=df["high"], low=df["low"], close=df["close"], volume=df["volume"]
    )
    df["VWAP"] = vwap.volume_weighted_average_price()
    atr = ta.volatility.AverageTrueRange(df["high"], df["low"], df["close"], window=14)
    df["ATR"] = atr.average_true_range()

    df["VWAP_upper"] = df["VWAP"] + band_mult * df["ATR"]
    df["VWAP_lower"] = df["VWAP"] - band_mult * df["ATR"]
    return df

# ----------------------------
# 🔹 Analyze Timeframe
# ----------------------------
def analyze(df, interval):
    if df is None or df.empty:
        return None

    df = add_vwap_bands(df)
    macd = ta.trend.MACD(df["close"])
    df["MACD"] = macd.macd()
    df["MACD_Signal"] = macd.macd_signal()

    latest = df.iloc[-1]
    current_price = latest["close"]

    trade, stop, target = None, None, None

    # 🔹 Momentum Long (Call)
    if current_price > latest["VWAP_upper"] and latest["MACD"] > latest["MACD_Signal"]:
        trade = "Momentum Long (Call)"
        stop = latest["VWAP"]
        target = current_price + latest["ATR"]

    # 🔹 Momentum Short (Put)
    elif current_price < latest["VWAP_lower"] and latest["MACD"] < latest["MACD_Signal"]:
        trade = "Momentum Short (Put)"
        stop = latest["VWAP"]
        target = current_price - latest["ATR"]

    # 🔹 Reversal Long (Call)
    elif abs(current_price - latest["VWAP_lower"]) < 0.1 * latest["ATR"] and latest["MACD"] > latest["MACD_Signal"]:
        trade = "Reversal Long (Call)"
        stop = latest["VWAP_lower"] - 0.5 * latest["ATR"]
        target = latest["VWAP"]

    # 🔹 Reversal Short (Put)
    elif abs(current_price - latest["VWAP_upper"]) < 0.1 * latest["ATR"] and latest["MACD"] < latest["MACD_Signal"]:
        trade = "Reversal Short (Put)"
        stop = latest["VWAP_upper"] + 0.5 * latest["ATR"]
        target = latest["VWAP"]

    return {
        "interval": interval,
        "current_price": round(current_price, 2),
        "vwap": round(latest["VWAP"], 2),
        "upper": round(latest["VWAP_upper"], 2),
        "lower": round(latest["VWAP_lower"], 2),
        "macd": round(latest["MACD"], 4),
        "macd_signal": round(latest["MACD_Signal"], 4),
        "atr": round(latest["ATR"], 2),
        "trade": trade if trade else "HOLD",
        "stop": round(stop, 2) if stop else None,
        "target": round(target, 2) if target else None,
    }

# ----------------------------
# 🔹 Run Signals
# ----------------------------
def get_signals(ticker):
    results = [
        analyze(get_ohlcv(ticker, "1h"), "1h"),
        analyze(get_ohlcv(ticker, "15min"), "15m"),
        analyze(get_ohlcv(ticker, "5min"), "5m"),
    ]
    return [r for r in results if r]

def majority_signal(results):
    trades = [r["trade"] for r in results if r["trade"] != "HOLD"]
    if not trades:
        return "HOLD"
    counts = {t: trades.count(t) for t in set(trades)}
    best = max(counts, key=counts.get)
    strength = "STRONG" if counts[best] == 3 else "MEDIUM" if counts[best] == 2 else "WEAK"
    return f"{strength} {best}"

# ----------------------------
# 🔹 Main Loop with Alerts
# ----------------------------
def main():
    if not API_KEY.strip():
        raise SystemExit("Set TWELVE_DATA_API_KEY (see .env.example).")
    tickers = input("Enter up to 2 tickers separated by a comma (e.g. SOFI, AAPL): ").strip().upper().split(",")
    tickers = [t.strip() for t in tickers if t.strip()][:2]

    print(f"\n🔄 Live scan for {', '.join(tickers)} (alerts only on NEW signals)...\n")

    while True:
        for ticker in tickers:
            results = get_signals(ticker)
            decision = majority_signal(results)

            # Compare with last known decision
            prev = last_signals.get(ticker, None)
            if decision != prev:
                print(f"\n📢 ALERT for {ticker}!")
                for res in results:
                    if res["trade"] != "HOLD":
                        print(f"[{res['interval']}] ➡️ {res['trade']} @ {res['current_price']} | Target: {res['target']} | Stop: {res['stop']}")
                print(f"✅ Majority Decision: {decision}")
                print("="*60)

            # Save latest decision
            last_signals[ticker] = decision

        time.sleep(60)

if __name__ == "__main__":
    main()
