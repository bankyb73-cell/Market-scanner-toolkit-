# Premarket Screener for a list of tickers
# Ranks symbols by gap %, premarket relative volume, and simple news sentiment
# --- Requirements ---
# pip install yfinance pandas numpy pytz

import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta, time
import pytz

# =====================
# USER SETTINGS
# =====================
TICKERS = ["AAPL", "NVDA", "TSLA", "AMD", "PLTR"]  # <-- replace with your list
LOOKBACK_DAYS_FOR_AVG_VOL = 20      # for average daily volume
NEWS_LOOKBACK_HOURS = 24            # how far back to read headlines
WEIGHTS = {                         # adjust how much each factor matters
    "gap": 0.45,
    "rvol": 0.35,
    "sent": 0.20,
}

# Premarket window for US equities
NY = pytz.timezone("America/New_York")
PREMARKET_START = time(4, 0)    # 04:00 ET
REG_OPEN = time(9, 30)          # 09:30 ET

# =====================
# Utilities
# =====================
POS_WORDS = {
    "beat","beats","surprise","strong","upgrade","raises","raise","hike","record",
    "positive","bullish","outperform","buy","wins","profit","guidance raised","growth",
}
NEG_WORDS = {
    "miss","misses","weak","downgrade","cuts","cut","slash","slump","fall",
    "negative","bearish","underperform","sell","loss","guidance lowered","probe","investigation",
}

def simple_headline_sentiment(headlines: list[str]) -> float:
    if not headlines:
        return 0.0
    scores = []
    for h in headlines:
        h_low = h.lower()
        pos = sum(word in h_low for word in POS_WORDS)
        neg = sum(word in h_low for word in NEG_WORDS)
        scores.append(pos - neg)
    # normalize to [-1, 1]
    s = np.mean(scores)
    if s == 0:
        return 0.0
    max_mag = max(1.0, max(abs(x) for x in scores))
    return float(np.clip(s / max_mag, -1, 1))


def get_prev_close(ticker: str) -> float:
    hist = yf.download(ticker, period="6d", interval="1d", auto_adjust=False,
                       progress=False, multi_level_index=False)
    hist.dropna(inplace=True)
    if len(hist) < 2:
        return float(hist["Close"].iloc[-1])
    return float(hist["Close"].iloc[-2])


def get_today_intraday(ticker: str) -> pd.DataFrame:
    # pull today 1-min with pre/post to include premarket
    df = yf.download(ticker, period="1d", interval="1m", prepost=True,
                     progress=False, multi_level_index=False)
    if df.empty:
        return df
    if df.index.tz is None:
        df = df.tz_localize("UTC")
    df = df.tz_convert(NY)
    df.index.name = "Datetime"
    return df


def compute_premarket_metrics(df_today: pd.DataFrame, prev_close: float) -> dict:
    if df_today.empty:
        return {"prem_last": np.nan, "prem_vol": 0, "gap_pct": np.nan}
    # filter to premarket window [04:00, 09:30)
    mask = (df_today.index.time >= PREMARKET_START) & (df_today.index.time < REG_OPEN)
    prem = df_today.loc[mask]
    if prem.empty:
        return {"prem_last": np.nan, "prem_vol": 0, "gap_pct": np.nan}
    prem_last = float(prem["Close"].iloc[-1])
    prem_vol = float(prem["Volume"].sum())
    gap_pct = ((prem_last - prev_close) / prev_close) * 100.0 if prev_close else np.nan
    return {"prem_last": prem_last, "prem_vol": prem_vol, "gap_pct": gap_pct}


def estimate_rvol(premarket_volume: float, ticker: str) -> float:
    # heuristic: compare premarket volume to proportion of average daily volume
    hist = yf.download(ticker, period=f"{LOOKBACK_DAYS_FOR_AVG_VOL}d", interval="1d",
                       auto_adjust=False, progress=False, multi_level_index=False)
    if hist.empty:
        return np.nan
    adv = float(hist["Volume"].mean())
    # how many minutes of premarket elapsed so far (up to now or 9:30, whichever is earlier)
    now_ny = datetime.now(NY)
    start_dt = datetime.combine(now_ny.date(), PREMARKET_START, tzinfo=NY)
    end_dt = datetime.combine(now_ny.date(), REG_OPEN, tzinfo=NY)
    elapsed = max(0, int((min(now_ny, end_dt) - start_dt).total_seconds() // 60))
    # trading day has ~390 minutes regular hours; we approximate premarket factor using 390 baseline
    expected_volume_share = max(1, int(390))
    expected_so_far = adv * (elapsed / expected_volume_share)
    if expected_so_far <= 0:
        return np.nan
    return float(premarket_volume / expected_so_far)


def get_recent_headlines(ticker: str, lookback_hours: int = NEWS_LOOKBACK_HOURS) -> list[str]:
    try:
        news_items = yf.Ticker(ticker).news or []
    except Exception:
        news_items = []
    if not news_items:
        return []
    cutoff = datetime.now(tz=pytz.UTC) - timedelta(hours=lookback_hours)
    headlines = []
    for item in news_items:
        # Older yfinance: flat dict with title + providerPublishTime (epoch).
        # Newer yfinance: {"content": {"title": ..., "pubDate": ISO-8601}}.
        content = item.get("content") or {}
        title = (item.get("title") or content.get("title") or "").strip()
        dt = None
        ts = item.get("providerPublishTime")
        if ts is not None:
            dt = datetime.fromtimestamp(ts, tz=pytz.UTC)
        elif content.get("pubDate"):
            try:
                dt = pd.to_datetime(content["pubDate"], utc=True).to_pydatetime()
            except Exception:
                dt = None
        if not title or dt is None:
            continue
        if dt >= cutoff:
            headlines.append(title)
    return headlines


def score_row(gap_pct: float, rvol: float, sent: float) -> float:
    # scale features safely
    g = 0 if np.isnan(gap_pct) else np.tanh(gap_pct / 5.0)           # compress big gaps
    r = 0 if np.isnan(rvol) else np.tanh(rvol - 1.0)                 # >1 means heavy vs expected
    s = 0 if np.isnan(sent) else sent                                # already in [-1,1]
    return WEIGHTS["gap"] * g + WEIGHTS["rvol"] * r + WEIGHTS["sent"] * s


def analyze_ticker(t: str) -> dict:
    prev_close = get_prev_close(t)
    intraday = get_today_intraday(t)
    pm = compute_premarket_metrics(intraday, prev_close)
    rvol = estimate_rvol(pm["prem_vol"], t)
    headlines = get_recent_headlines(t)
    sent = simple_headline_sentiment(headlines)
    score = score_row(pm["gap_pct"], rvol, sent)
    return {
        "ticker": t,
        "prev_close": prev_close,
        "prem_last": pm["prem_last"],
        "gap_pct": pm["gap_pct"],
        "prem_vol": pm["prem_vol"],
        "rvol": rvol,
        "headlines": "; ".join(headlines[:5]),
        "sentiment": sent,
        "score": score,
    }


def run_screen(tickers: list[str]) -> pd.DataFrame:
    rows = []
    for t in tickers:
        try:
            rows.append(analyze_ticker(t))
        except Exception as e:
            rows.append({"ticker": t, "error": str(e), "score": -999})
    df = pd.DataFrame(rows)
    df = df.sort_values("score", ascending=False).reset_index(drop=True)
    # pretty rounding
    for c in ["prev_close","prem_last","gap_pct","rvol","sentiment"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df

if __name__ == "__main__":
    df = run_screen(TICKERS)
    # display compact table
    pd.set_option("display.max_colwidth", 120)
    print(df[[
        "ticker","gap_pct","rvol","sentiment","prem_vol","prem_last","prev_close","score","headlines"
    ]])
    print("\nTop pick(s):")
    print(df.head(3)[["ticker","score","gap_pct","rvol","sentiment"]])
