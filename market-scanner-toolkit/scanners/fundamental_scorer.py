#!/usr/bin/env python3
"""
fundamental_scorer.py
Purpose: Fetch **fundamental** data only (no technicals) for a list of tickers,
score each on fundamentals, and label a "fundamental bias" per ticker to combine
with your own technical engine.

Data source: Finnhub (https://finnhub.io/) — requires API key.

Outputs:
- ./fa_output.csv      (one row per ticker with features + FA score + bias)
- ./fa_output.jsonl    (JSON lines, one object per ticker, with "reasons")

Usage:
1) Set FINNHUB_API_KEY in your environment or in a .env file (see .env.example).
2) Put your tickers in the TICKERS list, or pass on the command line:
     python fundamental_scorer.py AAPL MSFT TSLA
   To inspect the metric field names Finnhub returns for a symbol:
     python fundamental_scorer.py --keys AAPL
   If no CLI tickers given, script uses the TICKERS list.
"""

import os
import sys
import time
import json
import math
import csv
import datetime as dt
from typing import Dict, Any, List, Optional

import urllib.request
import urllib.parse
import ssl
import urllib.error

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:  # python-dotenv is optional
    pass

API_KEY = os.getenv("FINNHUB_API_KEY", "")
BASE = "https://finnhub.io/api/v1"

# Default tickers (ignored if you pass tickers on CLI)
TICKERS = ["AAPL", "MSFT", "NVDA", "AMD", "INTC", "SOFI", "F", "T"]  # replace with your list

# --------- Helper: simple HTTP GET with retries ---------
def http_get(path: str, params: Dict[str, Any]) -> Any:
    params = {**params, "token": API_KEY}
    url = f"{BASE}{path}?{urllib.parse.urlencode(params)}"
    ctx = ssl.create_default_context()
    backoff = 0.8
    for attempt in range(4):
        try:
            with urllib.request.urlopen(url, context=ctx, timeout=20) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as e:
            # 401/403/404 won't fix themselves (bad key or premium-only endpoint)
            if e.code not in (429, 500, 502, 503, 504) or attempt == 3:
                raise
        except Exception:
            if attempt == 3:
                raise
        time.sleep(backoff)
        backoff *= 1.8

# --------- Finnhub fetchers (fundamentals-only) ---------
def get_metrics(symbol: str) -> Dict[str, Any]:
    # Comprehensive ratios/metrics (TTM/Annual) when available
    return http_get("/stock/metric", {"symbol": symbol, "metric": "all"})

def get_earnings(symbol: str) -> List[Dict[str, Any]]:
    # Quarterly EPS actual/estimate/surprise
    out = http_get("/stock/earnings", {"symbol": symbol})
    return out if isinstance(out, list) else []

def get_recos(symbol: str) -> List[Dict[str, Any]]:
    # Analyst recommendation trends by month
    return http_get("/stock/recommendation", {"symbol": symbol})

def get_price_target(symbol: str) -> Dict[str, Any]:
    # Latest analyst price target stats
    return http_get("/stock/price-target", {"symbol": symbol})

def get_quote(symbol: str) -> Dict[str, Any]:
    # Current/prev close price (for upside calc)
    return http_get("/quote", {"symbol": symbol})

def get_insiders(symbol: str, since_days: int = 365) -> List[Dict[str, Any]]:
    # Corporate insider transactions
    since = (dt.date.today() - dt.timedelta(days=since_days)).isoformat()
    data = http_get("/stock/insider-transactions", {"symbol": symbol})
    tx = data.get("data", []) if isinstance(data, dict) else []
    # Filter by date
    keep = []
    for t in tx:
        d = t.get("transactionDate")
        if d and d >= since:
            keep.append(t)
    return keep

def get_institutional(symbol: str) -> Dict[str, Any]:
    # Institutional ownership (if available)
    return http_get("/stock/investor-ownership", {"symbol": symbol})

def get_news_sentiment(symbol: str) -> Dict[str, Any]:
    # Company news sentiment (score, buzz)
    return http_get("/news-sentiment", {"symbol": symbol})

# --------- Small helpers ---------
_warned = set()

def safe_fetch(fn, symbol, default):
    """Call a fetcher; if an endpoint is unavailable (e.g. premium-only), continue without it."""
    try:
        return fn(symbol)
    except Exception as e:
        if fn.__name__ not in _warned:
            print(f"warning: {fn.__name__} unavailable ({e}); continuing without it.", file=sys.stderr)
            _warned.add(fn.__name__)
        return default

def num(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return None

def first(d, *keys):
    """First non-None value among several candidate metric names."""
    for k in keys:
        v = d.get(k) if isinstance(d, dict) else None
        if v is not None:
            return v
    return None

# --------- Feature engineering ---------
def safe_get(d: Dict[str, Any], *keys, default=None):
    cur = d
    for k in keys:
        if isinstance(cur, dict) and k in cur:
            cur = cur[k]
        else:
            return default
    return cur

def pct(x):
    try:
        return float(x) * 100.0
    except Exception:
        return None

def avg(vals: List[Optional[float]]) -> Optional[float]:
    cleaned = [float(v) for v in vals if isinstance(v, (int, float)) and not math.isnan(v)]
    if not cleaned:
        return None
    return sum(cleaned) / len(cleaned)

def sum_safe(vals: List[Optional[float]]) -> float:
    return sum(float(v) for v in vals if isinstance(v, (int, float)) and not math.isnan(v))

def sign(x: Optional[float]) -> int:
    return 0 if x is None else (1 if x > 0 else (-1 if x < 0 else 0))

def earnings_surprise_avg(earnings: List[Dict[str, Any]], n: int = 4) -> Optional[float]:
    if not earnings:
        return None
    # Use last n quarters (recent first)
    es = []
    # Finnhub returns most-recent first in many cases; just sort by period descending to be safe
    earnings_sorted = sorted(earnings, key=lambda r: r.get("period",""), reverse=True)
    for e in earnings_sorted[:n]:
        actual = e.get("actual")
        estimate = e.get("estimate")
        if actual is not None and estimate not in (None, 0):
            es.append((actual - estimate) / abs(estimate) * 100.0)
    return avg(es) if es else None

def insider_net_usd(trans: List[Dict[str, Any]], days: int = 90) -> Optional[float]:
    cutoff = (dt.date.today() - dt.timedelta(days=days)).isoformat()
    net = 0.0
    for t in trans:
        d = t.get("transactionDate")
        if not d or d < cutoff:
            continue
        # Shares * sharePrice with sign by transactionCode (P=Purchase positive, S=Sale negative)
        shares = t.get("change")
        price = t.get("transactionPrice")
        code = (t.get("transactionCode") or "").upper()
        if isinstance(shares, (int,float)) and isinstance(price, (int,float)):
            usd = float(shares) * float(price)
            if code == "P":
                net += usd
            elif code == "S":
                net -= usd
    return net

def recos_net_strength(recos: List[Dict[str, Any]]) -> Optional[float]:
    if not recos:
        return None
    # Use most recent month’s line
    latest = sorted(recos, key=lambda r: r.get("period",""), reverse=True)[0]
    sb = latest.get("strongBuy", 0) or 0
    b  = latest.get("buy", 0) or 0
    h  = latest.get("hold", 0) or 0
    s  = latest.get("sell", 0) or 0
    ss = latest.get("strongSell", 0) or 0
    # Net = buys - sells, normalized by total
    total = sb + b + h + s + ss
    if total == 0:
        return None
    return (sb + b - s - ss) / total * 100.0

def build_features(symbol: str) -> Dict[str, Any]:
    metrics = safe_fetch(get_metrics, symbol, {})
    earnings = safe_fetch(get_earnings, symbol, [])
    recos = safe_fetch(get_recos, symbol, [])
    pt = safe_fetch(get_price_target, symbol, {})
    quote = safe_fetch(get_quote, symbol, {})
    insiders = safe_fetch(get_insiders, symbol, [])
    news = safe_fetch(get_news_sentiment, symbol, {})
    if not isinstance(recos, list):
        recos = []

    md = safe_get(metrics, "metric", default={})

    # Growth
    # Finnhub reports growth/margins/returns already in percent. Field names vary,
    # so run `--keys AAPL` to confirm them against a live response.
    rev_g = first(md, "revenueGrowthTTMYoy", "revenueGrowthTTM")
    eps_g = first(md, "epsGrowthTTMYoy", "epsGrowthTTM")

    # Profitability & returns
    net_margin = safe_get(md, "netProfitMarginTTM")
    op_margin  = safe_get(md, "operatingMarginTTM")
    gross_margin = safe_get(md, "grossMarginTTM")
    roe = safe_get(md, "roeTTM")
    roa = safe_get(md, "roaTTM")
    fcf_margin = safe_get(md, "freeCashFlowMarginTTM")

    # Leverage & liquidity
    de = safe_get(md, "totalDebt/totalEquityAnnual") or safe_get(md, "debtToEquityAnnual")
    curr_ratio = safe_get(md, "currentRatioAnnual")

    # Valuation
    pe = safe_get(md, "peTTM") or safe_get(md, "peAnnual")
    ps = safe_get(md, "psTTM") or safe_get(md, "psAnnual")
    pb = safe_get(md, "pbAnnual")
    peg = safe_get(md, "pegAnnual") or safe_get(md, "pegTTM")

    # Earnings & sentiment
    es_avg = earnings_surprise_avg(earnings, n=4)
    recos_net = recos_net_strength(recos)

    c = quote.get("c") or None  # current price
    target_mean = pt.get("targetMean") if isinstance(pt, dict) else None
    upside = None
    if c and target_mean:
        try:
            upside = (float(target_mean) - float(c)) / float(c) * 100.0
        except Exception:
            upside = None

    insider_90d = insider_net_usd(insiders, days=90)

    # News sentiment (Finnhub returns scores ~[-1,1], buzz etc.)
    news_score = news.get("companyNewsScore") if isinstance(news, dict) else None
    sector_avg_score = news.get("sectorAverageBullishPercent") if isinstance(news, dict) else None
    bullish_percent = safe_get(news, "sentiment", "bullishPercent")

    return {
        "symbol": symbol,
        "revGrowthTTM_pct": num(rev_g),
        "epsGrowthTTM_pct": num(eps_g),
        "grossMarginTTM_pct": num(gross_margin),
        "operatingMarginTTM_pct": num(op_margin),
        "netMarginTTM_pct": num(net_margin),
        "fcfMarginTTM_pct": num(fcf_margin),
        "roeTTM_pct": num(roe),
        "roaTTM_pct": num(roa),
        "debtToEquity": de,
        "currentRatio": curr_ratio,
        "peTTM": pe,
        "psTTM": ps,
        "pbAnnual": pb,
        "peg": peg,
        "earningsSurpriseAvg_pct": es_avg,
        "recosNetStrength_pct": recos_net,
        "priceTargetUpside_pct": upside,
        "insiderNet90d_USD": insider_90d,
        "newsScore": news_score,
        "bullishPercent_pct": pct(bullish_percent) if bullish_percent is not None else None,
        "sectorAvgBullish_pct": pct(sector_avg_score) if sector_avg_score is not None else None,
        "price": c,
        "targetMean": target_mean,
    }

# --------- Scoring (fundamentals only) ---------
def clamp(x, lo, hi):
    return max(lo, min(hi, x))

def score_fundamentals(feat: Dict[str, Any]) -> Dict[str, Any]:
    reasons = []

    # Growth (20 pts)
    g_rev = feat.get("revGrowthTTM_pct")
    g_eps = feat.get("epsGrowthTTM_pct")
    g_combo = avg([g_rev, g_eps])
    growth_pts = 0.0
    if g_combo is not None:
        # 0–30% growth mapped to 0–20 points (cap)
        growth_pts = clamp((g_combo/30.0)*20.0, 0, 20)
        if g_combo >= 10: reasons.append(f"Solid growth ({g_combo:.1f}% avg of revenue/EPS).")
        elif g_combo is not None: reasons.append(f"Low growth ({g_combo:.1f}%).")

    # Profitability (25 pts)
    nm = feat.get("netMarginTTM_pct")
    om = feat.get("operatingMarginTTM_pct")
    roe = feat.get("roeTTM_pct")
    prof_anchor = avg([nm, om, roe])
    prof_pts = 0.0
    if prof_anchor is not None:
        # -10% to +30% maps to 0–25
        prof_pts = clamp(((prof_anchor + 10.0)/40.0)*25.0, 0, 25)
        if prof_anchor >= 10: reasons.append(f"Healthy profitability/returns (anchor {prof_anchor:.1f}%).")
        else: reasons.append(f"Weak profitability/returns (anchor {prof_anchor:.1f}%).")

    # Balance sheet (15 pts) — favor lower D/E and decent current ratio
    de = feat.get("debtToEquity")
    cr = feat.get("currentRatio")
    bal_pts = 0.0
    if de is not None:
        # D/E: 0–2.0 maps to 15–5 pts (penalize high leverage)
        if de <= 0.5: bal_pts += 15.0
        elif de <= 1.0: bal_pts += 12.0
        elif de <= 2.0: bal_pts += 8.0
        else: bal_pts += 4.0
        reasons.append(f"Debt/Equity={de:.2f}.")
    if cr is not None:
        # Current ratio bonus/penalty +/- 3 pts
        if cr >= 1.5: bal_pts += 3.0
        elif cr < 1.0: bal_pts -= 3.0
        reasons.append(f"CurrentRatio={cr:.2f}.")

    # Valuation (20 pts) — lower PEG/PE earns more
    pe = feat.get("peTTM")
    peg = feat.get("peg")
    val_pts = 0.0
    if peg is not None and peg > 0:
        # PEG: 0.5–2.0 maps to 20–5
        if peg <= 0.7: val_pts += 20.0; reasons.append(f"Attractive PEG={peg:.2f}.")
        elif peg <= 1.0: val_pts += 17.0; reasons.append(f"Reasonable PEG={peg:.2f}.")
        elif peg <= 1.5: val_pts += 12.0; reasons.append(f"Elevated PEG={peg:.2f}.")
        elif peg <= 2.0: val_pts += 7.0;  reasons.append(f"Rich PEG={peg:.2f}.")
        else: val_pts += 3.0; reasons.append(f"Very rich PEG={peg:.2f}.")
    elif pe is not None and pe > 0:
        # PE backup scaling: 10–40 maps to 20–5
        if pe <= 12: val_pts += 20.0; reasons.append(f"Low PE={pe:.1f}.")
        elif pe <= 20: val_pts += 17.0; reasons.append(f"Reasonable PE={pe:.1f}.")
        elif pe <= 30: val_pts += 12.0; reasons.append(f"High PE={pe:.1f}.")
        elif pe <= 40: val_pts += 7.0;  reasons.append(f"Very high PE={pe:.1f}.")
        else: val_pts += 3.0; reasons.append(f"Extreme PE={pe:.1f}.")

    # Earnings & sentiment (20 pts)
    es = feat.get("earningsSurpriseAvg_pct")
    rec = feat.get("recosNetStrength_pct")
    up = feat.get("priceTargetUpside_pct")
    ins = feat.get("insiderNet90d_USD")
    news = feat.get("newsScore")

    es_pts = 0.0
    if es is not None:
        # -10% to +10% maps to 0–8
        es_pts = clamp(((es + 10.0)/20.0)*8.0, 0, 8)
        reasons.append(f"Avg EPS surprise {es:+.1f}%.")

    rec_pts = 0.0
    if rec is not None:
        # -50 to +50 maps to 0–6
        rec_pts = clamp(((rec + 50.0)/100.0)*6.0, 0, 6)
        reasons.append(f"Analyst net {rec:+.1f} pts.")

    up_pts = 0.0
    if up is not None:
        # -20% to +20% maps to 0–4
        up_pts = clamp(((up + 20.0)/40.0)*4.0, 0, 4)
        reasons.append(f"Target upside {up:+.1f}%.")

    ins_pts = 0.0
    if ins is not None:
        # +2 for net insider buy > $250k in 90d, -2 if net sell < -$250k
        if ins > 250_000: ins_pts = 2.0; reasons.append("Net insider buying (90d).")
        elif ins < -250_000: ins_pts = 0.0; reasons.append("Net insider selling (90d).")

    news_pts = 0.0
    if isinstance(news, (int, float)):
        # -0.3 to +0.6 maps to 0–2
        news_pts = clamp(((news + 0.3)/0.9)*2.0, 0, 2)

    sentiment_pts = es_pts + rec_pts + up_pts + ins_pts + news_pts
    reasons.append(f"Sentiment pts={sentiment_pts:.1f}.")

    total = growth_pts + prof_pts + bal_pts + val_pts + sentiment_pts
    total = clamp(total, 0, 100)

    # Bias thresholds (fundamentals-only)
    if total >= 60:
        bias = "Bullish"
    elif total <= 40:
        bias = "Bearish"
    else:
        bias = "Neutral"

    return {
        "FA_score": round(total, 1),
        "fundamental_bias": bias,
        "reasons": reasons
    }

# --------- Main ---------
def main():
    if not API_KEY.strip():
        print("ERROR: set FINNHUB_API_KEY (see .env.example).", file=sys.stderr)
        sys.exit(1)

    if len(sys.argv) > 2 and sys.argv[1] == "--keys":
        # Debug helper: list the metric field names Finnhub returns for a symbol
        md = safe_get(get_metrics(sys.argv[2]), "metric", default={})
        for k in sorted(md):
            print(f"{k}: {md[k]}")
        return

    tickers = sys.argv[1:] if len(sys.argv) > 1 else TICKERS
    results: List[Dict[str, Any]] = []

    for i, sym in enumerate(tickers, 1):
        try:
            print(f"[{i}/{len(tickers)}] Fetching fundamentals for {sym} ...")
            feat = build_features(sym)
            scored = score_fundamentals(feat)
            row = {**feat, **scored}
            results.append(row)
            # Be kind to rate limits
            time.sleep(0.15)
        except Exception as e:
            print(f"Failed {sym}: {e}", file=sys.stderr)

    # Write CSV
    fieldnames = sorted({k for r in results for k in r.keys()})
    with open("fa_output.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in results:
            # Flatten reasons to a single string
            r_csv = dict(r)
            r_csv["reasons"] = " | ".join(r.get("reasons", []))
            w.writerow(r_csv)

    # Write JSONL (keeps reasons as list)
    with open("fa_output.jsonl", "w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    # Console summary (compact)
    print("\n=== Fundamental Bias Summary ===")
    for r in results:
        print(f"{r['symbol']:>6} : {r['fundamental_bias']:>7}  (FA {r['FA_score']})")

if __name__ == "__main__":
    main()
