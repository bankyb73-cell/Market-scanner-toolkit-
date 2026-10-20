"""
swing_reversal.py

2-5 day reversal screener: daily RSI/MACD reversal setups (60% weight) blended
with TextBlob polarity of recent NewsAPI headlines (40% weight).
Research/education only; not financial advice.
"""
import requests
import pandas as pd
import time
from datetime import datetime, timedelta
from textblob import TextBlob

import os
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:  # python-dotenv is optional
    pass

# --- API KEYS (set in .env, see .env.example) ---
TWELVE_API = os.getenv("TWELVE_DATA_API_KEY", "")
NEWS_API = os.getenv("NEWS_API_KEY", "")

INTERVAL = "1day"
LOOKBACK = 30

# --- FUNCTIONS ---

def fetch_data(symbol):
    url = f"https://api.twelvedata.com/time_series?symbol={symbol}&interval={INTERVAL}&apikey={TWELVE_API}&outputsize={LOOKBACK}"
    r = requests.get(url).json()
    if "values" not in r:
        return None
    df = pd.DataFrame(r["values"])
    df["datetime"] = pd.to_datetime(df["datetime"])
    df = df.set_index("datetime").astype(float).sort_index()
    return df

def calculate_rsi(series, period=14):
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

def calculate_macd(series, short=12, long=26, signal=9):
    exp1 = series.ewm(span=short, adjust=False).mean()
    exp2 = series.ewm(span=long, adjust=False).mean()
    macd = exp1 - exp2
    signal_line = macd.ewm(span=signal, adjust=False).mean()
    hist = macd - signal_line
    return macd, signal_line, hist

def analyze_technicals(symbol):
    df = fetch_data(symbol)
    if df is None:
        return None, None, None

    df["RSI"] = calculate_rsi(df["close"])
    df["MACD"], df["Signal"], df["Hist"] = calculate_macd(df["close"])

    latest = df.iloc[-1]
    prev = df.iloc[-2]

    score = 0
    reason = []

    if latest["RSI"] < 30 and latest["Hist"] > prev["Hist"]:
        score += 70
        reason.append("Bullish Reversal (RSI oversold + MACD improving).")
    elif latest["RSI"] > 70 and latest["Hist"] < prev["Hist"]:
        score -= 70
        reason.append("Bearish Reversal (RSI overbought + MACD weakening).")
    else:
        reason.append("No strong reversal setup.")

    return score, round(latest["RSI"], 2), reason

def fetch_news_sentiment(symbol):
    today = datetime.now()
    from_date = (today - timedelta(days=7)).strftime("%Y-%m-%d")
    url = f"https://newsapi.org/v2/everything?q={symbol}&from={from_date}&sortBy=publishedAt&apiKey={NEWS_API}"
    r = requests.get(url).json()

    if "articles" not in r or not r["articles"]:
        return 0, ["No recent news found."]

    headlines = [a["title"] for a in r["articles"][:5]]
    sentiment_scores = []
    for h in headlines:
        blob = TextBlob(h)
        sentiment_scores.append(blob.sentiment.polarity)

    avg_sentiment = sum(sentiment_scores) / len(sentiment_scores)

    news_score = avg_sentiment * 100
    reason = [f"News sentiment avg polarity: {round(avg_sentiment, 2)}"]

    return news_score, headlines[:2]

def final_decision(symbol):
    tech_score, rsi, tech_reason = analyze_technicals(symbol)
    news_score, headlines = fetch_news_sentiment(symbol)

    if tech_score is None:
        return f"{symbol}: ❌ No technical data available"

    final_score = (0.6 * tech_score) + (0.4 * news_score)

    if final_score > 20:
        bias = "🟢 BULLISH"
    elif final_score < -20:
        bias = "🔴 BEARISH"
    else:
        bias = "⚪ NEUTRAL / HOLD"

    output = f"""
=== {symbol} Reversal Screener ===
RSI: {rsi}
Technical Score: {tech_score} | Reasons: {tech_reason}
News Score: {round(news_score, 2)} | Headlines: {headlines}
➡ Final Bias: {bias} (Score={round(final_score, 2)})
    """
    return output

# --- MAIN ---
if __name__ == "__main__":
    if not TWELVE_API.strip() or not NEWS_API.strip():
        raise SystemExit("Set TWELVE_DATA_API_KEY and NEWS_API_KEY (see .env.example).")
    print("\n=== Hybrid Reversal Screener (2–5 Day Outlook) ===\n")
    ticker1 = input("Enter first ticker: ").upper()
    ticker2 = input("Enter second ticker: ").upper()

    for ticker in [ticker1, ticker2]:
        print(final_decision(ticker))
        time.sleep(2)
    print("\n✅ Scan Complete\n")
