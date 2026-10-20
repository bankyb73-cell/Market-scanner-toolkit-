#!/usr/bin/env python3
"""
options_alert_bot.py

Watches a list of tickers on 1-minute bars. When a bullish/bearish engulfing
candle is confirmed by the MACD histogram, it picks a cheap, liquid option
contract in the matching direction and sends an alert (console, email, and an
optional spoken alert on macOS). After an alert it tracks that contract and
warns on drawdowns or an opposite-direction engulfing candle.

It ONLY ALERTS. It never places orders. Research/education only; not
financial advice.

Setup: copy .env.example to .env and fill in the ALERT_* variables (use a
Gmail *app password*, never your real password). Email is skipped if they are
not set.
"""

import os
import smtplib
import subprocess
import sys
import time
from datetime import datetime, time as dtime
from email.mime.text import MIMEText
from zoneinfo import ZoneInfo

import pandas as pd
import ta
import yfinance as yf

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:  # python-dotenv is optional
    pass

# === CONFIG ===
EMAIL_ADDRESS = os.getenv("ALERT_EMAIL_ADDRESS", "")
APP_PASSWORD = os.getenv("ALERT_EMAIL_APP_PASSWORD", "")
TO_EMAIL = os.getenv("ALERT_TO_EMAIL", EMAIL_ADDRESS)

TICKERS = ["AAPL", "MSFT", "NVDA", "AMD", "TSLA", "SPY", "QQQ"]  # replace with your list
MAX_PREMIUM = 200          # max cost of one contract in dollars (ask * 100)
SCAN_INTERVAL = 60         # seconds between scan cycles
EXPIRATION_INDEX = 0       # 0 = nearest expiry (can be same-day), 1 = next, ...
DRAWDOWN_LEVELS = [0.3, 0.4, 0.5, 0.6]  # alert when the contract is down 30%..60%
MIN_BARS_FOR_MACD = 35     # MACD(12,26,9) needs roughly this many bars
USE_SPOKEN_ALERTS = False  # macOS only (uses the built-in `say` command)

# US market hours are evaluated in New York time, wherever you run this.
NY = ZoneInfo("America/New_York")
MARKET_OPEN = dtime(9, 30)
MARKET_CLOSE = dtime(16, 0)

active_position = None


# === HELPERS ===
def is_market_open() -> bool:
    now = datetime.now(NY)
    return now.weekday() < 5 and MARKET_OPEN <= now.time() <= MARKET_CLOSE


def send_email(subject, body):
    if not (EMAIL_ADDRESS and APP_PASSWORD and TO_EMAIL):
        return
    msg = MIMEText(body)
    msg["Subject"] = subject
    msg["From"] = EMAIL_ADDRESS
    msg["To"] = TO_EMAIL
    try:
        with smtplib.SMTP_SSL("smtp.gmail.com", 465) as server:
            server.login(EMAIL_ADDRESS, APP_PASSWORD)
            server.send_message(msg)
    except Exception as e:
        print(f"[EMAIL ERROR] {e}")


def audio_alert(message):
    if USE_SPOKEN_ALERTS and sys.platform == "darwin":
        subprocess.run(["say", message], check=False)


def download_bars(ticker):
    df = yf.download(ticker, interval="1m", period="1d",
                     progress=False, multi_level_index=False)
    return pd.DataFrame() if df is None else df.dropna()


def detect_patterns(df):
    """Return 'Bullish Engulfing' / 'Bearish Engulfing' for the last two bars, else None."""
    if df is None or len(df) < 2:
        return None
    last = df.iloc[-1]
    prev = df.iloc[-2]
    if (last["Close"] > last["Open"] and prev["Close"] < prev["Open"]
            and last["Close"] > prev["Open"] and last["Open"] < prev["Close"]):
        return "Bullish Engulfing"
    if (last["Close"] < last["Open"] and prev["Close"] > prev["Open"]
            and last["Close"] < prev["Open"] and last["Open"] > prev["Close"]):
        return "Bearish Engulfing"
    return None


def macd_histogram(df):
    """Latest MACD histogram value, or None if there isn't enough data."""
    if len(df) < MIN_BARS_FOR_MACD:
        return None
    value = ta.trend.MACD(df["Close"]).macd_diff().iloc[-1]
    return None if pd.isna(value) else float(value)


def get_best_option(ticker, direction):
    """Most liquid affordable contract (calls for CALL, puts for PUT) on the chosen expiry."""
    try:
        stock = yf.Ticker(ticker)
        expirations = stock.options
        if not expirations:
            return None
        chain = stock.option_chain(expirations[min(EXPIRATION_INDEX, len(expirations) - 1)])
        table = chain.calls if direction == "CALL" else chain.puts
        df = table[(table["ask"] > 0)
                   & (table["ask"] * 100 <= MAX_PREMIUM)
                   & (table["volume"] > 0)].copy()
        if df.empty:
            return None
        df["spread"] = df["ask"] - df["bid"]
        row = df.sort_values(["volume", "spread"], ascending=[False, True]).iloc[0]
        return {
            "symbol": row["contractSymbol"],
            "ask": float(row["ask"]),
            "bid": float(row["bid"]),
            "strike": float(row["strike"]),
            "volume": int(row["volume"]),
            "type": direction,
        }
    except Exception as e:
        print(f"[OPTION ERROR] {ticker}: {e}")
        return None


# === SCAN / MONITOR ===
def scan_ticker(ticker):
    global active_position
    try:
        df = download_bars(ticker)
        if len(df) < MIN_BARS_FOR_MACD:
            return
        pattern = detect_patterns(df)
        hist = macd_histogram(df)
        if not pattern or hist is None:
            return

        if pattern.startswith("Bullish") and hist > 0:
            direction = "CALL"
        elif pattern.startswith("Bearish") and hist < 0:
            direction = "PUT"
        else:
            return

        option = get_best_option(ticker, direction)
        if not option:
            return

        active_position = {
            "ticker": ticker,
            "pattern": pattern,
            "direction": direction,
            "entry": option["ask"],
            "symbol": option["symbol"],
            "alerted": set(),
        }
        alert = (
            f"\n[ENTRY ALERT] {ticker}\n"
            f"Pattern: {pattern}\n"
            f"MACD histogram: {hist:+.4f}\n"
            f"Option: {option['symbol']} ({direction}) | Ask: ${option['ask']:.2f} | "
            f"Strike: {option['strike']} | Volume: {option['volume']}\n"
        )
        print(alert)
        send_email(f"[ENTRY] {ticker} {pattern}", alert)
        audio_alert("entry alert detected")
    except Exception as e:
        print(f"[SCAN ERROR] {ticker}: {e}")


def monitor_position():
    global active_position
    if not active_position:
        return
    pos = active_position
    try:
        option = yf.Ticker(pos["symbol"]).history(period="1d", interval="1m")
        if not option.empty and "Close" in option:
            last = float(option["Close"].iloc[-1])
            drawdown = (pos["entry"] - last) / pos["entry"]
            for level in DRAWDOWN_LEVELS:
                if drawdown >= level and level not in pos["alerted"]:
                    warn = f"[DRAWDOWN WARNING] {pos['symbol']} is down {int(level * 100)}% from entry."
                    print(warn)
                    send_email("Drawdown Alert", warn)
                    audio_alert(f"warning position down {int(level * 100)} percent")
                    pos["alerted"].add(level)

        df = download_bars(pos["ticker"])
        reverse = detect_patterns(df)
        if reverse and not reverse.startswith(pos["pattern"].split()[0]):
            exit_msg = (f"[EXIT ALERT] {pos['ticker']} reversal pattern detected "
                        f"({reverse}). Consider exiting {pos['symbol']}.")
            print(exit_msg)
            send_email("Exit Alert", exit_msg)
            audio_alert("exit alert detected")
            active_position = None
    except Exception as e:
        print(f"[MONITOR ERROR] {pos['ticker']}: {e}")


def main():
    print("Options alert bot started. Ctrl+C to stop.")
    try:
        while True:
            if is_market_open():
                if active_position is None:
                    for ticker in TICKERS:
                        scan_ticker(ticker)
                        if active_position is not None:
                            break
                        time.sleep(1)  # be gentle with Yahoo
                else:
                    monitor_position()
            else:
                print("[INFO] US market closed. Waiting...")
            time.sleep(SCAN_INTERVAL)
    except KeyboardInterrupt:
        print("\nStopped.")


if __name__ == "__main__":
    main()
