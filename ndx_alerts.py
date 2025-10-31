# ============================================================
# 📊 NDX Alerts Bot - Telegram Auto Update (Nasdaq Strategy)
# ============================================================
# Test 09h (heure Paris) -> TRIGGER_HOURS={9}
# Puis remets {8,12,16,20} après le test.
# ============================================================

import os, sys, time, requests, pandas as pd, numpy as np
from datetime import datetime
import pytz, holidays
import yfinance as yf

# --- Config ---
TG_TOKEN = os.getenv("TG_TOKEN", "")
TG_CHAT  = os.getenv("TG_CHAT", "")
SYMBOL   = os.getenv("SYMBOL_PRICE", "^NDX")  # ou "QQQ"
TZ       = pytz.timezone("Europe/Paris")
TRIGGER_HOURS = {9}   # TEST à 09h ; remets {8,12,16,20} ensuite

LEV, SL_PCT, TP_PCT, RISK = 20.0, -0.009, 0.022, 0.02
RRI_HALF, RRI_OUT = 10, 25

def now_paris(): return datetime.now(TZ)
def is_us_trading_day(d): return (d.weekday() < 5) and (d.date() not in holidays.US())

def send_tg(text):
    if not TG_TOKEN or not TG_CHAT:
        print("Telegram non configuré (TG_TOKEN/TG_CHAT).")
        return
    try:
        url = f"https://api.telegram.org/bot{TG_TOKEN}/sendMessage"
        requests.get(url, params={"chat_id": TG_CHAT, "text": text}, timeout=20)
    except Exception as e:
        print("Telegram error:", e)

def rsi(series, n=14):
    d = series.diff()
    up, dn = d.clip(lower=0), -d.clip(upper=0)
    ru = up.ewm(alpha=1/n, adjust=False).mean()
    rd = dn.ewm(alpha=1/n, adjust=False).mean().replace(0, np.nan)
    rs = ru / rd
    return (100 - 100/(1+rs)).fillna(50)

def ema_1d(arr, n):
    """EMA sur array 1D (corrige l'erreur ndarray (n,1))."""
    x = np.asarray(arr, dtype=float).ravel()
    return pd.Series(x).ewm(span=n, adjust=False).mean()

def dl_yf(ticker, period, interval, retries=3):
    last_err = None
    for _ in range(retries):
        try:
            df = yf.download(ticker, period=period, interval=interval,
                             progress=False, threads=False)
            if not df.empty:
                return df
        except Exception as e:
            last_err = e
        time.sleep(3)
    if last_err: print("yfinance error:", last_err)
    return pd.DataFrame()

def compute():
    df = dl_yf(SYMBOL, "60d", "30m")
    if df.empty:  # fallback daily
        df = dl_yf(SYMBOL, "1y", "1d")
    if df.empty:
        raise RuntimeError("Aucune donnée dispo pour le symbole.")

    closes = df["Close"].dropna()
    last = float(closes.iloc[-1])

    rsi14 = float(rsi(closes).iloc[-1])

    # --- MACD (corrigé en 1D) ---
    c_arr = closes.to_numpy(dtype=float).ravel()
    ema12 = ema_1d(c_arr, 12)
    ema26 = ema_1d(c_arr, 26)
    macd_line = ema12 - ema26
    macd_sig  = macd_line.ewm(span=9, adjust=False).mean()
    macd = float(macd_line.iloc[-1])
    sig  = float(macd_sig.iloc[-1])

    # --- VXN proxy ---
    rets = closes.pct_change().dropna()
    vxn = float(rets.tail(20).std() * np.sqrt(252) * 100) if len(rets) >= 20 else 20.0
    vxn = min(max(vxn, 10), 50)

    # --- US10Y & Put/Call (avec fallback) ---
    tnx = dl_yf("^TNX", "5d", "1d")
    us10y = float(tnx["Close"].iloc[-1]) / 10.0 if not tnx.empty else 4.30

    cpc = dl_yf("^CPC", "10d", "1d")
    pc = float(cpc["Close"].dropna().iloc[-1]) if not cpc.empty else 1.00

    # --- RRI ---
    rri = (0.25*(rsi14 > 70) +
           0.25*(vxn > 22) +
           0.20*(macd < sig) +
           0.15*(us10y > 4.5) +
           0.15*(pc > 1.1)) * 100

    # --- Niveaux & exposition ---
    sl = last * (1 + SL_PCT)
    tp = last * (1 + TP_PCT)
    expo_cap = RISK / (abs(SL_PCT) * LEV)      # ≈ 0.111
    base = 1.0 if rri <= RRI_HALF else (0.5 if rri <= RRI_OUT else 0.0)
    expo = min(base, expo_cap)
    action = "PLEINE EXPO" if rri <= RRI_HALF else ("DEMI-EXPO" if rri <= RRI_OUT else "SORTIE (RRI>seuil)")

    return dict(price=last, sl=sl, tp=tp, rsi=rsi14, macd=macd, sig=sig,
                vxn=vxn, us10y=us10y, pc=pc, rri=rri, expo=expo, action=action)

def build_msg(d):
    return (f"📈 Nasdaq Plan – Update\n"
            f"Symbole: {SYMBOL}\n"
            f"Prix: {d['price']:.2f}\n"
            f"SL(-0.9%): {d['sl']:.2f} | TP(+2.2%): {d['tp']:.2f}\n"
            f"RSI14: {d['rsi']:.1f} | MACD: {d['macd']:.2f}/{d['sig']:.2f}\n"
            f"VXN: {d['vxn']:.1f}% | US10Y: {d['us10y']:.2f}% | P/C: {d['pc']:.2f}\n"
            f"RRI: {d['rri']:.0f}% | Expo: {d['expo']:.3f}\n"
            f"Action: {d['action']}\n"
            f"Horodatage: {now_paris().strftime('%Y-%m-%d %H:%M %Z')}")

def main():
    try:
        now = now_paris()
        if now.hour not in TRIGGER_HOURS or not is_us_trading_day(now):
            print("⏱ Pas l'heure/ jour non US → exit 0")
            return 0
        d = compute()
        send_tg(build_msg(d))
        print("✅ Message envoyé")
        return 0
    except Exception as e:
        # On ne “fail” pas le job : on log et on sort 0 pour éviter les emails “failed”
        print("⚠️ Run sans envoi:", repr(e))
        send_tg(f"⚠️ NDX Alerts: run sans envoi – {e}")
        return 0

if __name__ == "__main__":
    sys.exit(main())
