#!/usr/bin/env python3
"""
Market Sentiment Scanner (RSS-based, no API keys needed)

- Pulls articles from curated financial RSS feeds
- Runs sentiment analysis (VADER + FinBERT)
- Saves results to SQLite for daily tracking
- Aggregates into Bullish / Bearish / Neutral / Reversals
"""

import feedparser
import requests
import sqlite3
import time
import re
import logging
import urllib.robotparser
from bs4 import BeautifulSoup
from datetime import datetime, timezone
import pandas as pd

# Sentiment tools
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from transformers import pipeline
import nltk

nltk.download("vader_lexicon", quiet=True)  # one-time lexicon download

# ------------- CONFIG ----------------

RSS_FEEDS = [
    "https://feeds.a.dj.com/rss/RSSMarketsMain.xml",   # WSJ Markets
    "https://www.cnbc.com/id/100003114/device/rss/rss.html",  # CNBC Top News
    "https://www.reutersagency.com/feed/?best-topics=business-finance",  # Reuters
    "https://www.nasdaq.com/feed/rssoutbound?category=Market%20News",  # Nasdaq
    "https://finance.yahoo.com/news/rssindex",  # Yahoo Finance Headlines
    "https://www.sec.gov/Archives/edgar/usgaap.rss.xml",  # SEC filings
    "https://www.prnewswire.com/rss/finance-business-latest-news.rss",  # Press releases
    "https://feeds.bloomberg.com/markets/news.rss",  # Bloomberg
]

# Tickers to track. Matching is case-sensitive and word-bounded; single-letter
# tickers (e.g. "F", "T") only match when written as a cashtag like "$F".
STOCK_TICKERS = [
    "AAPL", "MSFT", "NVDA", "AMZN", "GOOGL",
    "META", "TSLA", "AMD", "INTC", "PLTR",
]

# Storing full article text bloats the DB and can raise publisher copyright/ToS
# issues. Keep False unless the data stays private and you need the text.
STORE_CONTENT = False

DB_FILE = "market_sentiment.db"
USER_AGENT = "Mozilla/5.0 (compatible; MarketSentimentBot/1.0)"
REQUEST_TIMEOUT = 10
SLEEP_BETWEEN_REQUESTS = 2  # seconds

# ------------- LOGGING ----------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)

# ------------- SETUP ------------------

# Init SQLite
conn = sqlite3.connect(DB_FILE)
cursor = conn.cursor()
cursor.execute("""
CREATE TABLE IF NOT EXISTS sentiment (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    datetime TEXT,
    source TEXT,
    title TEXT,
    link TEXT,
    content TEXT,
    ticker TEXT,
    vader_score REAL,
    finbert_label TEXT,
    finbert_score REAL
)
""")
conn.commit()

# Init VADER
vader = SentimentIntensityAnalyzer()

# Init FinBERT
finbert = pipeline("sentiment-analysis", model="ProsusAI/finbert")

# Robots.txt checker
def allowed_to_scrape(url: str) -> bool:
    try:
        base_url = re.match(r"^https?://[^/]+", url).group(0)
        rp = urllib.robotparser.RobotFileParser()
        rp.set_url(base_url + "/robots.txt")
        rp.read()
        return rp.can_fetch(USER_AGENT, url)
    except Exception:
        return False

# Extract article text safely
def fetch_article(url: str) -> str:
    if not allowed_to_scrape(url):
        logging.warning(f"Skipping (robots.txt disallows): {url}")
        return ""
    try:
        headers = {"User-Agent": USER_AGENT}
        resp = requests.get(url, headers=headers, timeout=REQUEST_TIMEOUT)
        if resp.status_code != 200:
            return ""
        soup = BeautifulSoup(resp.text, "html.parser")
        paragraphs = [p.get_text() for p in soup.find_all("p")]
        return " ".join(paragraphs).strip()
    except Exception as e:
        logging.error(f"Error fetching {url}: {e}")
        return ""

# Find ticker mentions (case-sensitive, word-bounded; 1-letter tickers need a $cashtag)
_TICKER_RE = {}

def find_tickers(text: str):
    found = []
    for t in STOCK_TICKERS:
        if t not in _TICKER_RE:
            if len(t) == 1:
                pattern = rf"(?<![A-Za-z])\${t}(?![A-Za-z])"
            else:
                pattern = rf"(?<![A-Za-z])\$?{t}(?![A-Za-z])"
            _TICKER_RE[t] = re.compile(pattern)
        if _TICKER_RE[t].search(text):
            found.append(t)
    return found

# Run sentiment analysis
def analyze_sentiment(text: str):
    vader_score = vader.polarity_scores(text)["compound"]
    try:
        result = finbert(text[:2000], truncation=True, max_length=512)[0]
        return vader_score, result["label"].upper(), result["score"]
    except Exception as e:
        logging.error(f"FinBERT error: {e}")
        return vader_score, "NEUTRAL", 0.0

# Save results
def save_result(entry, content, ticker, vader_score, finbert_label, finbert_score):
    cursor.execute("""
        INSERT INTO sentiment (datetime, source, title, link, content, ticker, vader_score, finbert_label, finbert_score)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        datetime.now(timezone.utc).isoformat(),
        str(entry.get("source", "")),            # force string
        str(entry.get("title", "")),
        str(entry.get("link", "")),
        str(content) if STORE_CONTENT else "",
        str(ticker),
        float(vader_score),
        str(finbert_label),
        float(finbert_score)
    ))
    conn.commit()


# Main scan
def run_scan():
    for feed in RSS_FEEDS:
        logging.info(f"Fetching feed: {feed}")
        try:
            d = feedparser.parse(feed)
            if not d.entries:
                headers = {"User-Agent": USER_AGENT}
                resp = requests.get(feed, headers=headers, timeout=REQUEST_TIMEOUT)
                d = feedparser.parse(resp.text)
        except Exception as e:
            logging.error(f"Error parsing feed {feed}: {e}")
            continue

        for entry in d.entries[:10]:  # limit per feed
            link = entry.get("link", "")
            title = entry.get("title", "")
            logging.info(f"Processing: {title}")
            content = fetch_article(link)
            if not content:
                continue
            tickers_found = find_tickers(title + " " + content)
            if not tickers_found:
                tickers_found = ["GENERAL"]
            vader_score, finbert_label, finbert_score = analyze_sentiment(content)
            for ticker in tickers_found:
                save_result(entry, content, ticker, vader_score, finbert_label, finbert_score)
            time.sleep(SLEEP_BETWEEN_REQUESTS)

# -------- DAILY AGGREGATION --------
def aggregate_daily_sentiment():
    conn = sqlite3.connect(DB_FILE)
    df = pd.read_sql_query("""
        SELECT date(datetime) as day, ticker, finbert_label
        FROM sentiment
    """, conn)

    results = []
    for (day, ticker), group in df.groupby(["day", "ticker"]):
        pos = (group.finbert_label == "POSITIVE").sum()
        neg = (group.finbert_label == "NEGATIVE").sum()
        neu = (group.finbert_label == "NEUTRAL").sum()

        if pos > neg and pos > neu:
            sentiment = "Bullish"
        elif neg > pos and neg > neu:
            sentiment = "Bearish"
        else:
            sentiment = "Neutral"

        results.append((day, ticker, sentiment))

    daily = pd.DataFrame(results, columns=["day", "ticker", "sentiment"])

    # Detect reversals
    daily["final_sentiment"] = daily["sentiment"]
    for idx, row in daily.iterrows():
        prev = daily[(daily["ticker"] == row.ticker) & (daily["day"] < row.day)].tail(1)
        if not prev.empty:
            prev_sent = prev.sentiment.values[0]
            if prev_sent == "Bearish" and row.sentiment == "Bullish":
                daily.at[idx, "final_sentiment"] = "Bullish Reversal"
            elif prev_sent == "Bullish" and row.sentiment == "Bearish":
                daily.at[idx, "final_sentiment"] = "Bearish Reversal"

    # Save to DB
    cur = conn.cursor()
    cur.execute("""
        CREATE TABLE IF NOT EXISTS daily_sentiment (
            day TEXT,
            ticker TEXT,
            sentiment TEXT,
            PRIMARY KEY (day, ticker)
        )
    """)
    cur.executemany(
        "INSERT OR REPLACE INTO daily_sentiment (day, ticker, sentiment) VALUES (?, ?, ?)",
        daily[["day", "ticker", "final_sentiment"]].values.tolist()
    )
    conn.commit()
    conn.close()
    logging.info("Daily sentiment aggregated and stored.")

# ------------- MAIN ------------------
if __name__ == "__main__":
    run_scan()
    aggregate_daily_sentiment()
    logging.info("Scan + aggregation complete.")

    # Show today's results
    conn = sqlite3.connect(DB_FILE)
    cur = conn.cursor()
    cur.execute("SELECT ticker, sentiment FROM daily_sentiment WHERE day=date('now')")
    rows = cur.fetchall()
    conn.close()

    print("\n📊 Today's Sentiment Calls:")
    for ticker, sentiment in rows:
        print(f"{ticker:6} -> {sentiment}")
