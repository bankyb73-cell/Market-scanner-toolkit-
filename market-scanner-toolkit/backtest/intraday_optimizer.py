#!/usr/bin/env python3
"""
Intraday Rule Optimizer for Calls (bullish) and Puts (bearish)
- 15-minute timeframe, last ~3 months of data
- Uses RSI(14) and Volume relative to 20-bar MA
- Grid-searches RSI thresholds, volume multipliers, target multipliers
- Backtests intraday reversal setups:
    * Bullish ("call") = prior RSI <= threshold + bullish candle + volume spike
    * Bearish ("put")  = prior RSI >= threshold + bearish candle + volume spike
- Entry at next-bar open, stop at setup bar low/high, target = entry +/- target_mult * risk
- Max hold = 2 hours (8 bars)
- Outputs per-ticker CSVs and summary file.

NOTE: parameters are selected on in-sample PnL, so results are optimistic.
Research/education only; not financial advice.
"""

import os
import time
from datetime import datetime, timezone
import requests
import numpy as np
import pandas as pd

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:  # python-dotenv is optional
    pass

# ---------------- USER CONFIG ----------------
# Twelve Data API key (set TWELVE_DATA_API_KEY in .env, see .env.example)
TWELVEDATA_APIKEY = os.getenv("TWELVE_DATA_API_KEY", "")

WATCHLIST = ["SPY", "QQQ", "AAPL", "AMD", "F", "T", "KO"]  # replace with your list

INTERVAL = "15min"

# We'll request a safe number of bars roughly covering ~3 months.
# 3 months trading: ~63 trading days -> 63 * (6.5 hours * 4 fifteen-min bars) ≈ 1638 bars.
# Use a margin and request 2000-3000 bars to be safe.
LOOKBACK_BARS = 2200

PAUSE_BETWEEN_REQUESTS_SECONDS = 8.5  # avoid hitting basic-plan rate limits

# Indicators / Backtest settings
RSI_PERIOD = 14
VOL_MA_PERIOD = 20
MAX_HOLD_CANDLES = 8  # 2 hours
MIN_TRADES = 20       # ignore parameter sets with fewer trades (too noisy to trust)

# Grid to search (adjustable)
RSI_THRESHOLDS = [25, 28, 30, 33, 35]
VOL_MULTS = [1.2, 1.5, 1.8]
TARGET_MULTS = [1.2, 1.5, 2.0]

# Output files
SUMMARY_OUT = "summary_all_tickers.csv"
# ---------------------------------------------

# ---------------- helpers ---------------------
def fetch_candles_twelvedata(symbol, interval="15min", outputsize=2000, apikey=None):
    base = "https://api.twelvedata.com/time_series"
    params = {
        "symbol": symbol,
        "interval": interval,
        "outputsize": outputsize,
        "format": "JSON"
    }
    if apikey:
        params["apikey"] = apikey
    r = requests.get(base, params=params, timeout=30)
    j = r.json()
    if "status" in j and j.get("status") == "error":
        raise Exception(f"TwelveData error for {symbol}: {j.get('message')}")
    if "values" not in j:
        raise Exception(f"Unexpected response for {symbol}: {j}")
    rows = j["values"]
    df = pd.DataFrame(rows)
    df = df.rename(columns={
        "datetime": "datetime",
        "open": "open",
        "high": "high",
        "low": "low",
        "close": "close",
        "volume": "volume"
    })
    df["datetime"] = pd.to_datetime(df["datetime"])
    df = df.sort_values("datetime").reset_index(drop=True)
    for c in ["open","high","low","close","volume"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df[["datetime","open","high","low","close","volume"]]

def compute_rsi(series, period=14):
    delta = series.diff()
    up = delta.clip(lower=0)
    down = -1 * delta.clip(upper=0)
    ma_up = up.ewm(alpha=1/period, adjust=False).mean()
    ma_down = down.ewm(alpha=1/period, adjust=False).mean()
    rs = ma_up / (ma_down + 1e-12)
    rsi = 100 - (100 / (1 + rs))
    return rsi

def engineer(df):
    df = df.copy()
    df["rsi"] = compute_rsi(df["close"], period=RSI_PERIOD)
    df["vol_ma"] = df["volume"].rolling(VOL_MA_PERIOD).mean()
    df["vol_mult"] = df["volume"] / (df["vol_ma"] + 1e-12)
    return df

def find_setups(df, rsi_thresh, vol_mult, target_mult, side="CALL"):
    """
    side == "CALL" -> bullish reversal (entry long)
    side == "PUT"  -> bearish reversal (entry short)
    Returns list of trade dicts simulated.
    """
    trades = []
    n = len(df)
    # scan bars leaving room for forward look
    for i in range(2, n - MAX_HOLD_CANDLES - 1):
        prev_rsi = df.loc[i-1, "rsi"]
        prev2_rsi = df.loc[i-2, "rsi"]
        cur_rsi = df.loc[i, "rsi"]
        if pd.isna(prev_rsi) or pd.isna(cur_rsi):
            continue
        vol_spike = df.loc[i, "volume"] > (df.loc[i, "vol_ma"] * vol_mult if not pd.isna(df.loc[i, "vol_ma"]) else 1e12)
        bullish = df.loc[i, "close"] > df.loc[i, "open"]
        bearish = df.loc[i, "close"] < df.loc[i, "open"]

        if side == "CALL":
            # prior RSI <= threshold AND bullish candle + vol spike
            if (prev_rsi <= rsi_thresh or prev2_rsi <= rsi_thresh) and bullish and vol_spike:
                entry = df.loc[i+1, "open"]
                stop = df.loc[i, "low"]
                risk = entry - stop
                if risk <= 0: 
                    continue
                target = entry + target_mult * risk
                # forward simulate up to MAX_HOLD_CANDLES
                exit_price = None
                exit_reason = None
                exit_index = None
                success = 0
                for j in range(i+1, i+1+MAX_HOLD_CANDLES):
                    # If one bar touches both levels we can't know the order, so assume the stop hit first.
                    if df.loc[j, "low"] <= stop:
                        exit_price = stop; exit_reason = "STOP_HIT"; exit_index = j; success = 0; break
                    if df.loc[j, "high"] >= target:
                        exit_price = target; exit_reason = "TARGET_HIT"; exit_index = j; success = 1; break
                if exit_index is None:
                    j = i+MAX_HOLD_CANDLES
                    exit_price = df.loc[j, "close"]
                    exit_reason = "TIME_EXIT"
                    exit_index = j
                    success = 1 if exit_price >= target else 0
                pnl = exit_price - entry
                trades.append({
                    "setup_index": i,
                    "setup_time": df.loc[i, "datetime"],
                    "side": "CALL",
                    "entry": entry,
                    "stop": stop,
                    "target": target,
                    "exit_index": exit_index,
                    "exit_time": df.loc[exit_index, "datetime"],
                    "exit_price": exit_price,
                    "exit_reason": exit_reason,
                    "success": success,
                    "pnl": pnl,
                    "hold_bars": exit_index - (i+1) + 1
                })
        else:  # PUT
            if (prev_rsi >= rsi_thresh or prev2_rsi >= rsi_thresh) and bearish and vol_spike:
                entry = df.loc[i+1, "open"]
                stop = df.loc[i, "high"]
                risk = stop - entry
                if risk <= 0:
                    continue
                target = entry - target_mult * risk
                exit_price = None
                exit_reason = None
                exit_index = None
                success = 0
                for j in range(i+1, i+1+MAX_HOLD_CANDLES):
                    # Same conservative rule: if both levels are touched in one bar, the stop hit first.
                    if df.loc[j, "high"] >= stop:
                        exit_price = stop; exit_reason = "STOP_HIT"; exit_index = j; success = 0; break
                    if df.loc[j, "low"] <= target:
                        exit_price = target; exit_reason = "TARGET_HIT"; exit_index = j; success = 1; break
                if exit_index is None:
                    j = i+MAX_HOLD_CANDLES
                    exit_price = df.loc[j, "close"]
                    exit_reason = "TIME_EXIT"
                    exit_index = j
                    success = 1 if exit_price <= target else 0
                pnl = entry - exit_price
                trades.append({
                    "setup_index": i,
                    "setup_time": df.loc[i, "datetime"],
                    "side": "PUT",
                    "entry": entry,
                    "stop": stop,
                    "target": target,
                    "exit_index": exit_index,
                    "exit_time": df.loc[exit_index, "datetime"],
                    "exit_price": exit_price,
                    "exit_reason": exit_reason,
                    "success": success,
                    "pnl": pnl,
                    "hold_bars": exit_index - (i+1) + 1
                })
    return trades

def metrics_from_trades(trades):
    if not trades:
        return {"num_trades":0,"total_pnl":0.0,"avg_pnl":0.0,"win_rate":0.0,"avg_hold_bars":0.0,"profit_factor":0.0}
    pnls = np.array([t["pnl"] for t in trades])
    wins = pnls[pnls>0]; losses = -pnls[pnls<=0]
    total_pnl = pnls.sum()
    avg_pnl = pnls.mean()
    win_rate = (pnls>0).sum()/len(pnls)
    avg_hold = np.mean([t["hold_bars"] for t in trades])
    profit_factor = wins.sum() / (losses.sum()+1e-12) if losses.sum()>0 else float('inf')
    return {"num_trades":int(len(trades)),"total_pnl":float(total_pnl),"avg_pnl":float(avg_pnl),"win_rate":float(win_rate),"avg_hold_bars":float(avg_hold),"profit_factor":float(profit_factor)}

# --------------- optimizer per ticker ----------------
def optimize_ticker(symbol):
    print(f"\nProcessing {symbol} at {datetime.now(timezone.utc).isoformat()}")
    try:
        df = fetch_candles_twelvedata(symbol, INTERVAL, outputsize=LOOKBACK_BARS, apikey=TWELVEDATA_APIKEY)
    except Exception as e:
        print(f"Data fetch error for {symbol}: {e}")
        return None
    print(f"Fetched {len(df)} bars for {symbol}. Engineering features...")
    df = engineer(df).reset_index(drop=True).dropna(subset=["rsi","vol_ma"])
    results = []
    for r in RSI_THRESHOLDS:
        for vm in VOL_MULTS:
            for tm in TARGET_MULTS:
                # CALLs
                trades_call = find_setups(df, r, vm, tm, side="CALL")
                m_call = metrics_from_trades(trades_call)
                results.append({"symbol":symbol,"side":"CALL","rsi_thresh":r,"vol_mult":vm,"target_mult":tm,**m_call})
                # PUTs
                trades_put = find_setups(df, r, vm, tm, side="PUT")
                m_put = metrics_from_trades(trades_put)
                results.append({"symbol":symbol,"side":"PUT","rsi_thresh":r,"vol_mult":vm,"target_mult":tm,**m_put})
    results_df = pd.DataFrame(results)
    if results_df.empty:
        print(f"No results for {symbol}.")
        return None
    # Save full grid
    tune_name = f"tuning_{symbol}.csv"
    results_df.to_csv(tune_name, index=False)
    print(f"Saved tuning grid to {tune_name} ({len(results_df)} rows).")

    # Pick best per side by total_pnl, require at least 3 trades ideally
    bests = []
    for side in ["CALL","PUT"]:
        side_df = results_df[results_df["side"]==side].copy()
        if side_df.empty:
            continue
        # prefer those with num_trades >=3; if none, pick highest total_pnl anyway
        candidates = side_df[side_df["num_trades"] >= MIN_TRADES]
        if candidates.empty:
            print(f"{symbol} {side}: no parameter set produced >= {MIN_TRADES} trades; skipping.")
            continue
        best_row = candidates.sort_values(["total_pnl","profit_factor"], ascending=False).iloc[0].to_dict()
        bests.append(best_row)
        # Save trades for that best config
        trades = find_setups(df, best_row["rsi_thresh"], best_row["vol_mult"], best_row["target_mult"], side="CALL" if side=="CALL" else "PUT")
        trades_df = pd.DataFrame(trades)
        out_trades = f"trades_{symbol}_{side}.csv"
        if not trades_df.empty:
            trades_df.to_csv(out_trades, index=False)
        else:
            # write header-only file
            pd.DataFrame(columns=[
                "setup_index","setup_time","side","entry","stop","target","exit_index","exit_time","exit_price","exit_reason","success","pnl","hold_bars"
            ]).to_csv(out_trades, index=False)
        print(f"Saved simulated trades for {symbol} {side} to {out_trades} (rows: {len(trades_df)})")

    # create summary best overall choose side with highest total_pnl
    best_overall = sorted(bests, key=lambda x: (x["total_pnl"], x["profit_factor"]), reverse=True)[0] if bests else None
    if best_overall:
        summary = {
            "symbol": symbol,
            "best_side": best_overall["side"],
            "best_rsi_thresh": best_overall["rsi_thresh"],
            "best_vol_mult": best_overall["vol_mult"],
            "best_target_mult": best_overall["target_mult"],
            "num_trades": int(best_overall["num_trades"]),
            "total_pnl": float(best_overall["total_pnl"]),
            "avg_pnl": float(best_overall["avg_pnl"]),
            "win_rate": float(best_overall["win_rate"]),
            "avg_hold_bars": float(best_overall["avg_hold_bars"]),
            "profit_factor": float(best_overall["profit_factor"])
        }
        return summary
    return None

# ---------------- main ----------------
def main():
    summaries = []
    for idx, t in enumerate(WATCHLIST):
        try:
            s = optimize_ticker(t)
            if s:
                summaries.append(s)
        except Exception as e:
            print(f"Error optimizing {t}: {e}")
        # pause between requests to respect Twelve Data basic plan
        if idx < len(WATCHLIST)-1:
            print(f"Sleeping {PAUSE_BETWEEN_REQUESTS_SECONDS}s before next ticker...")
            time.sleep(PAUSE_BETWEEN_REQUESTS_SECONDS)
    if summaries:
        df_sum = pd.DataFrame(summaries)
        df_sum.to_csv(SUMMARY_OUT, index=False)
        print(f"\nWROTE summary for all tickers to {SUMMARY_OUT}")
        print(df_sum.sort_values("total_pnl", ascending=False).head(10).to_string(index=False))
    else:
        print("No summaries produced.")

if __name__ == "__main__":
    main()
