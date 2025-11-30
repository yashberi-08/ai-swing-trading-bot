# ================================
# ai_trader.py
# Automated AI Swing Trading Bot
# Sends BUY / EXIT / STOPLOSS alerts daily
# ================================

import pandas as pd
import numpy as np
import yfinance as yf
import joblib, pickle
import requests
from datetime import datetime, timedelta

# ================================
# CONFIGURATION
# ================================

BOT_TOKEN = "8548779438:AAEorzEzvrxBchf5VV_99MnXf1A7GhonLAo"
CHAT_ID = "6907221342"
K = 5                   # number of stocks to buy daily
STOP_LOSS = 0.03        # 3% SL
TARGET = 0.05           # 5% Target


# ================================
# TELEGRAM MESSAGE FUNCTION
# ================================

def send_telegram(msg):
    url = f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage"
    data = {"chat_id": CHAT_ID, "text": msg, "parse_mode": "Markdown"}
    requests.post(url, data=data)


# ================================
# LOAD MODEL + FEATURES
# ================================

lgbm = joblib.load("ai_swing_model.pkl")
features = pickle.load(open("features.pkl", "rb"))

tickers = list(features.keys())
feature_cols = ["close","ret1","ret5","sma20","sma50","sma200","rsi14","atr"]


# ================================
# FIXED YFINANCE — SINGLE TICKER DOWNLOAD
# ================================

def get_latest_prices(tickers):
    all_prices = {}

    print("Downloading prices (1-by-1 mode)...")

    for t in tickers:
        try:
            df = yf.download(
                t,
                start="2024-01-01",
                end=datetime.today().strftime("%Y-%m-%d"),
                progress=False,
                timeout=20
            )["Close"]

            all_prices[t] = df

        except Exception as e:
            print(f"Failed: {t} — {e}")

    close = pd.DataFrame(all_prices)
    close = close.ffill().bfill()
    return close


close = get_latest_prices(tickers)
today = close.index[-1]


# ================================
# UPDATE FEATURES
# ================================

def update_features(df, price):
    df["close"] = price
    df["ret1"] = price.pct_change()
    df["ret5"] = price.pct_change(5)
    df["sma20"] = price.rolling(20).mean()
    df["sma50"] = price.rolling(50).mean()
    df["sma200"] = price.rolling(200).mean()

    delta = price.diff()
    up = delta.clip(lower=0)
    down = -delta.clip(upper=0)
    rs = up.rolling(14).mean() / (down.rolling(14).mean() + 1e-6)
    df["rsi14"] = 100 - (100 / (1 + rs))

    df["atr"] = price.pct_change().rolling(14).std()
    return df.ffill().bfill()


for t in tickers:
    features[t] = update_features(features[t], close[t])


# ================================
# BUY SIGNAL ENGINE
# ================================

def get_buy_signals():
    picks = []

    for t in tickers:
        df = features[t]

        X = df[feature_cols].iloc[-1:].values
        pred = lgbm.predict(X)[0]

        trend_ok = df["sma20"].iloc[-1] > df["sma50"].iloc[-1]

        if pred == 1 and trend_ok:
            price = close[t].iloc[-1]
            picks.append((t, price))

    picks = sorted(picks, key=lambda x: x[1], reverse=True)[:K]

    result = []
    for t, price in picks:
        result.append({
            "Stock": t,
            "Buy": round(price, 2),
            "SL": round(price * (1 - STOP_LOSS), 2),
            "Target": round(price * (1 + TARGET), 2)
        })
    return pd.DataFrame(result)


# ================================
# EXIT SIGNAL ENGINE
# ================================

def check_exit_signals():
    try:
        portfolio = pd.read_csv("open_positions.csv")
    except:
        portfolio = pd.DataFrame(columns=["Stock","Buy","SL","Target","Entry"])

    exit_rows = []

    for idx, row in portfolio.iterrows():
        t = row["Stock"]
        if t not in close.columns:
            continue

        price = close[t].iloc[-1]

        if price <= row["SL"]:
            exit_rows.append({"Stock": t, "ExitPrice": price, "Reason": "STOPLOSS"})
            continue

        if price >= row["Target"]:
            exit_rows.append({"Stock": t, "ExitPrice": price, "Reason": "TARGET"})
            continue

        df = features[t]
        pred = lgbm.predict(df[feature_cols].iloc[-1:].values)[0]
        trend_ok = df["sma20"].iloc[-1] > df["sma50"].iloc[-1]

        if not trend_ok or pred != 1:
            exit_rows.append({"Stock": t, "ExitPrice": price, "Reason": "REVERSAL"})

    for e in exit_rows:
        portfolio = portfolio[portfolio["Stock"] != e["Stock"]]

    portfolio.to_csv("open_positions.csv", index=False)
    return pd.DataFrame(exit_rows)


# ================================
# SAVE TODAY'S BUY SIGNALS
# ================================

buy_df = get_buy_signals()
exit_df = check_exit_signals()

try:
    portfolio = pd.read_csv("open_positions.csv")
except:
    portfolio = pd.DataFrame(columns=["Stock","Buy","SL","Target","Entry"])

for idx, row in buy_df.iterrows():
    row["Entry"] = today
    portfolio = pd.concat([portfolio, pd.DataFrame([row])], ignore_index=True)

portfolio.to_csv("open_positions.csv", index=False)


# ================================
# SEND TELEGRAM MESSAGE
# ================================

msg = f"📅 *AI Swing Trading Signals — {today.date()}*\n\n"

if len(buy_df) > 0:
    msg += "🟢 *BUY Signals:*\n"
    for i, r in buy_df.iterrows():
        msg += f"• {r['Stock']} @ {r['Buy']}  (SL: {r['SL']}  Target: {r['Target']})\n"
else:
    msg += "🟢 BUY: None\n"

msg += "\n"

if len(exit_df) > 0:
    msg += "🔵 *EXIT Signals:*\n"
    for i, r in exit_df.iterrows():
        msg += f"• {r['Stock']} — {r['Reason']} @ {r['ExitPrice']}\n"
else:
    msg += "🔵 EXIT: None\n"

msg += "\n"

send_telegram(msg)
print("Alert sent!")
