"""
earnings_scanner.py

Classifies the latest EPS surprise (Strong Beat / Beat / Inline / Miss), scans
recent headlines for guidance language, and shows the latest 1-day price move.
Uses Finnhub + Twelve Data. Research/education only; not financial advice.
"""
import requests
from datetime import datetime, timedelta

import os
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:  # python-dotenv is optional
    pass

# 🔑 YOUR API KEYS
FINNHUB_KEY = os.getenv("FINNHUB_API_KEY", "")
TWELVE_KEY = os.getenv("TWELVE_DATA_API_KEY", "")

# ----------------------------
# 1️⃣ Get Earnings Data
# ----------------------------
def get_earnings_data(ticker):
    url = f"https://finnhub.io/api/v1/stock/earnings?symbol={ticker}&token={FINNHUB_KEY}"
    response = requests.get(url)
    data = response.json()

    if not isinstance(data, list) or not data:
        return None

    latest = data[0]

    return {
        "eps_actual": latest.get("actual"),
        "eps_estimate": latest.get("estimate"),
        "revenue_actual": latest.get("revenueActual"),
        "revenue_estimate": latest.get("revenueEstimate"),
        "period": latest.get("period")
    }

# ----------------------------
# 2️⃣ Surprise Classification
# ----------------------------
def classify(actual, estimate):
    if actual is None or not estimate:
        return "N/A", 0

    surprise_pct = (actual - estimate) / abs(estimate)

    if surprise_pct > 0.05:
        return "Strong Beat", surprise_pct
    elif surprise_pct > 0:
        return "Beat", surprise_pct
    elif surprise_pct == 0:
        return "Inline", surprise_pct
    else:
        return "Miss", surprise_pct

# ----------------------------
# 3️⃣ Guidance Sentiment (Headline Scan)
# ----------------------------
def get_guidance_sentiment(ticker):
    end = datetime.today().date()
    start = end - timedelta(days=30)
    url = (f"https://finnhub.io/api/v1/company-news?symbol={ticker}"
           f"&from={start}&to={end}&token={FINNHUB_KEY}")
    response = requests.get(url)
    news = response.json()

    keywords_positive = ["raises guidance", "increased outlook", "strong outlook"]
    keywords_negative = ["cuts guidance", "lower outlook", "macro headwinds"]

    if not isinstance(news, list):
        return "Neutral"

    for article in news[:10]:
        headline = article.get("headline", "").lower()
        for word in keywords_positive:
            if word in headline:
                return "Bullish"
        for word in keywords_negative:
            if word in headline:
                return "Bearish"

    return "Neutral"

# ----------------------------
# 4️⃣ 1-Day Price Reaction
# ----------------------------
def get_price_reaction(ticker):
    today = datetime.today()
    past = today - timedelta(days=5)

    url = f"https://api.twelvedata.com/time_series?symbol={ticker}&interval=1day&start_date={past.date()}&apikey={TWELVE_KEY}"
    response = requests.get(url)
    data = response.json()

    if "values" not in data:
        return "N/A"

    prices = data["values"]
    if len(prices) < 2:
        return "N/A"

    latest_close = float(prices[0]["close"])
    prev_close = float(prices[1]["close"])

    move = (latest_close - prev_close) / prev_close

    return round(move * 100, 2)

# ----------------------------
# 5️⃣ Main Scanner
# ----------------------------
def scan_earnings(ticker):
    earnings = get_earnings_data(ticker)

    if not earnings:
        print("No earnings data found.")
        return

    eps_result, eps_surprise = classify(
        earnings["eps_actual"],
        earnings["eps_estimate"]
    )

    rev_result, rev_surprise = classify(
        earnings["revenue_actual"],
        earnings["revenue_estimate"]
    )

    guidance = get_guidance_sentiment(ticker)
    price_move = get_price_reaction(ticker)

    print(f"\n📊 Earnings Scan for {ticker.upper()}")
    print("-" * 40)
    print(f"Period: {earnings['period']}")
    print(f"EPS: {earnings['eps_actual']} vs {earnings['eps_estimate']} → {eps_result}")
    if earnings["revenue_actual"] is not None:  # not every Finnhub plan/endpoint returns revenue
        print(f"Revenue: {earnings['revenue_actual']} vs {earnings['revenue_estimate']} → {rev_result}")
    print(f"Guidance Sentiment: {guidance}")
    print(f"Latest 1-day move: {price_move}%")

# ----------------------------
# 6️⃣ Manual Input
# ----------------------------
if __name__ == "__main__":
    if not FINNHUB_KEY.strip() or not TWELVE_KEY.strip():
        raise SystemExit("Set FINNHUB_API_KEY and TWELVE_DATA_API_KEY (see .env.example).")
    ticker = input("Enter ticker to scan: ")
    scan_earnings(ticker)
