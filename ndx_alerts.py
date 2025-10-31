# ============================================================
# 📊 NDX Alerts Bot - Telegram Auto Update (Nasdaq Strategy)
# ============================================================
# Envoie un message Telegram à 08h, 12h, 16h, 20h (heure Paris)
# avec prix, RSI, MACD, RRI, stop loss, take profit, etc.
# ============================================================

import os, requests, pandas as pd, numpy as np
from datetime import datetime
import pytz, holidays
import yfinance as yf

# ========= CONFIGURATION =========
TELEGRAM_BOT_TOKEN = os.getenv("TG_TOKEN", "")
TELEGRAM_CHAT_ID   = os.getenv("TG_CHAT", "")
SYMBOL_PRICE       = os.getenv("SYMBOL_PRICE", "^NDX")   # ou "QQQ"
TZ                 = pytz.timezone("Europe/Paris")
TRIGGER_HOURS      = {8, 12, 16, 20}
LEV                = 20.0
SL_PCT             = -0.009
TP_PCT             =  0.022
RISK_PER_TRADE     = 0.02
RRI_HALF           = 10
RRI_OUT            = 25
# =================================

def send_telegram(text):
    """Envoie un message Telegram via Bot API"""
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        print("❌ Config Telegram manquante.")
        return
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    try:
        requests.get(url, params={"chat_id": TELEGRAM_CHAT_ID, "text": text})
    except Exception as e:
        print("Erreur Telegram:", e)

def rsi(series, period=14):
    delta = series.diff()
    up, down = delta.clip(lower=0), -delta.clip(upper=0)
    roll_up = up.ewm(alpha=1/period, adjust=False).mean()
    roll_down = down.ewm(alpha=1/period, adjust=False).mean().replace(0, np.nan)
    rs = roll_up / roll_down
    return (100 - (100 / (1 + rs))).fillna(50)

def ema(arr, n):
    return pd.Series(arr).ewm(span=n, adjust=False).mean()

def now_paris():
    return datetime.now(TZ)

def is_us_trading_day(d):
    us_holidays = holidays.US()
    return (d.weekday() < 5) and (d.date() not in us_holidays)

def compute_indicators():
    """Télécharge les données et calcule RSI, MACD, VXN proxy, etc."""
    try:
        df = yf.download(SYMBOL_PRICE, period="60d", interval="30m", progress=False)
        if df.empty:
            raise ValueError("Pas de données 30m.")
    except Exception:
        df = yf.download(SYMBOL_PRICE, period="1y", interval="1d", progress=False)

    closes = df["Close"].dropna()
    last_price = float(closes.iloc[-1])
    rsi14 = float(rsi(closes, 14).iloc[-1])
    ema12 = ema(closes.values, 12)
    ema26 = ema(closes.values, 26)
    macd_line = ema12 - ema26
    macd_sig  = macd_line.ewm(span=9, adjust=False).mean()
    macd_val, macd_sigv = float(macd_line.iloc[-1]), float(macd_sig.iloc[-1])
    rets = closes.pct_change().dropna()
    vxn_proxy = float(rets.tail(20).std() * np.sqrt(252) * 100) if len(rets) >= 20 else 20.0
    vxn_proxy = min(max(vxn_proxy, 10), 50)
    try:
        tnx = yf.download("^TNX", period="5d", interval="1d", progress=False)
        us10y = float(tnx["Close"].iloc[-1]) / 10.0 if not tnx.empty else 4.30
    except Exception:
        us10y = 4.30
    try:
        cpc = yf.download("^CPC", period="10d", interval="1d", progress=False)
        put_call = float(cpc["Close"].dropna().iloc[-1]) if not cpc.empty else 1.00
    except Exception:
        put_call = 1.00
    rri = (0.25*(1 if rsi14>70 else 0)
         + 0.25*(1 if vxn_proxy>22 else 0)
         + 0.20*(1 if macd_val<macd_sigv else 0)
         + 0.15*(1 if us10y>4.5 else 0)
         + 0.15*(1 if put_call>1.1 else 0)) * 100
    sl = last_price * (1 + SL_PCT)
    tp = last_price * (1 + TP_PCT)
    expo_cap = RISK_PER_TRADE / (abs(SL_PCT) * LEV)
    base = 1.0 if rri <= RRI_HALF else (0.5 if rri <= RRI_OUT else 0.0)
    exposure = min(base, expo_cap)
    if rri > RRI_OUT: action = "SORTIE (RRI > seuil)"
    elif rri > RRI_HALF: action = "DEMI-EXPO (zone orange)"
    else: action = "PLEINE EXPO (zone verte)"
    return dict(price=last_price, sl=sl, tp=tp, rsi=rsi14,
                macd=macd_val, sig=macd_sigv, vxn=vxn_proxy,
                us10y=us10y, putcall=put_call, rri=rri,
                expo=exposure, action=action)

def build_message(info):
    return (
f"📈 Nasdaq Plan – Update\n"
f"Symbole: {SYMBOL_PRICE}\n"
f"Prix: {info['price']:.2f}\n"
f"SL (-0.9%): {info['sl']:.2f} | TP (+2.2%): {info['tp']:.2f}\n"
f"RSI14: {info['rsi']:.1f} | MACD: {info['macd']:.2f}/{info['sig']:.2f}\n"
f"VXN proxy: {info['vxn']:.1f}% | US10Y: {info['us10y']:.2f}% | P/C: {info['putcall']:.2f}\n"
f"RRI: {info['rri']:.0f}% | Expo: {info['expo']:.3f}\n"
f"Action: {info['action']}\n"
f"Horodatage: {now_paris().strftime('%Y-%m-%d %H:%M %Z')}"
)

def main():
    now = now_paris()
    if now.hour not in TRIGGER_HOURS or not is_us_trading_day(now):
        return
    info = compute_indicators()
    send_telegram(build_message(info))

if __name__ == "__main__":
    main()
