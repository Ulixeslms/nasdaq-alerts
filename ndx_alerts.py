# ============================================================
# 📊 NDX Alerts Bot - Production (analyse + fraîcheur + distances + lien)
# Heures Paris: 08h, 09h, 12h, 16h, 20h
# BUY / HOLD / CLOSE / SELL + Risk sizing & RRI + Résumé analyse
# ============================================================

import os, sys, time, requests, pandas as pd, numpy as np
from datetime import datetime, timezone
import pytz, holidays
import yfinance as yf

# ---------- Config ----------
TG_TOKEN = os.getenv("TG_TOKEN", "")
TG_CHAT  = os.getenv("TG_CHAT", "")
PRIMARY_SYMBOL   = os.getenv("SYMBOL_PRICE", "^NDX")   # symbole principal
FALLBACK_SYMBOL  = "QQQ"                               # fallback si ^NDX trop ancien
DASHBOARD_URL    = os.getenv("DASHBOARD_URL", "https://ton-dashboard.example")  # ← mets ici le lien Netlify/GitHub Pages
TZ_PARIS = pytz.timezone("Europe/Paris")
TRIGGER_HOURS = {8, 9, 12, 16, 20}                     # ← production

# Stratégie
LEV, SL_PCT, TP_PCT, RISK = 20.0, -0.009, 0.022, 0.02
RRI_HALF, RRI_OUT = 10, 25
CAPITAL_USD = float(os.getenv("CAPITAL_USD", "100000"))
SHORT_ALLOWED = int(os.getenv("SHORT_ALLOWED", "1"))   # SELL autorisé

# Seuil de fraîcheur des données (minutes)
FRESH_LIMIT_MIN = 90

# ---------- Utils ----------
def now_paris(): return datetime.now(TZ_PARIS)
def is_us_trading_day(d): return (d.weekday() < 5) and (d.date() not in holidays.US())
def send_tg(text):
    if not TG_TOKEN or not TG_CHAT: return
    try:
        requests.get(f"https://api.telegram.org/bot{TG_TOKEN}/sendMessage",
                     params={"chat_id": TG_CHAT, "text": text}, timeout=20)
    except Exception as e:
        print("Telegram error:", e)

def rsi(series, n=14):
    d = series.diff()
    up, dn = d.clip(lower=0), -d.clip(upper=0)
    ru = up.ewm(alpha=1/n, adjust=False).mean()
    rd = dn.ewm(alpha=1/n, adjust=False).mean().replace(0, np.nan)
    rs = ru/rd
    return (100 - 100/(1+rs)).fillna(50)

def ema_1d(arr, n):
    x = np.asarray(arr, dtype=float).ravel()
    return pd.Series(x).ewm(span=n, adjust=False).mean()

def dl_yf(ticker, period, interval, retries=3):
    last_err = None
    for _ in range(retries):
        try:
            df = yf.download(ticker, period=period, interval=interval,
                             progress=False, threads=False)
            if not df.empty: return df
        except Exception as e: last_err = e
        time.sleep(2)
    if last_err: print("yfinance error:", last_err)
    return pd.DataFrame()

def get_fresh_series(symbol):
    df = dl_yf(symbol, "60d", "30m")
    if df.empty: df = dl_yf(symbol, "1y", "1d")
    if df.empty: return None, pd.DataFrame(), 10**9

    last_ts = df.index[-1]
    if last_ts.tz is None: last_ts = last_ts.tz_localize(timezone.utc)
    age_min = (datetime.now(timezone.utc) - last_ts.tz_convert(timezone.utc)).total_seconds()/60.0

    if (age_min > FRESH_LIMIT_MIN) and (symbol == PRIMARY_SYMBOL):
        df_fb = dl_yf(FALLBACK_SYMBOL, "60d", "30m")
        if not df_fb.empty:
            last_ts_fb = df_fb.index[-1]
            if last_ts_fb.tz is None: last_ts_fb = last_ts_fb.tz_localize(timezone.utc)
            age_min_fb = (datetime.now(timezone.utc) - last_ts_fb.tz_convert(timezone.utc)).total_seconds()/60.0
            if age_min_fb < age_min:
                return FALLBACK_SYMBOL, df_fb, age_min_fb

    return symbol, df, age_min

# ---------- Calculs ----------
def compute_metrics():
    symbol, df, age_min = get_fresh_series(PRIMARY_SYMBOL)
    if df.empty: raise RuntimeError("Aucune donnée disponible.")
    closes = df["Close"].dropna()
    last   = float(closes.iloc[-1])

    # RSI
    rsi14 = float(rsi(closes).iloc[-1])

    # MACD (12,26,9)
    carr = closes.to_numpy(dtype=float).ravel()
    ema12 = ema_1d(carr, 12); ema26 = ema_1d(carr, 26)
    macd_line = ema12 - ema26
    macd_sig  = macd_line.ewm(span=9, adjust=False).mean()
    macd, sig = float(macd_line.iloc[-1]), float(macd_sig.iloc[-1])

    # VXN proxy
    rets = closes.pct_change().dropna()
    vxn = float(rets.tail(20).std() * np.sqrt(252) * 100) if len(rets)>=20 else 20.0
    vxn = min(max(vxn, 10), 50)

    # US10Y / Put-Call
    tnx = dl_yf("^TNX", "5d", "1d")
    us10y = float(tnx["Close"].iloc[-1]) / 10.0 if not tnx.empty else 4.30
    cpc = dl_yf("^CPC", "10d", "1d")
    pc = float(cpc["Close"].dropna().iloc[-1]) if not cpc.empty else 1.00

    # Tendance (EMA50/200)
    ema50  = float(ema_1d(carr, 50).iloc[-1]) if len(carr)>=50 else np.nan
    ema200 = float(ema_1d(carr, 200).iloc[-1]) if len(carr)>=200 else np.nan
    trend = "haussière" if (not np.isnan(ema50) and not np.isnan(ema200) and ema50 >= ema200) else "neutre/baissière"

    # RRI
    rri = (0.25*(rsi14>70) + 0.25*(vxn>22) + 0.20*(macd<sig) + 0.15*(us10y>4.5) + 0.15*(pc>1.1)) * 100

    # Niveaux & exposition
    sl = last * (1 + SL_PCT)
    tp = last * (1 + TP_PCT)
    expo_cap = RISK / (abs(SL_PCT) * LEV)      # ~0.111
    base = 1.0 if rri <= RRI_HALF else (0.5 if rri <= RRI_OUT else 0.0)
    expo = min(base, expo_cap)

    # Distances en points (Δ > 0 si le prix est au-dessus du SL)
    dist_to_sl = last - sl
    dist_to_tp = tp - last

    # Risk sizing
    nominal    = CAPITAL_USD * expo * LEV
    risk_usd   = CAPITAL_USD * expo * abs(SL_PCT) * LEV
    reward_usd = CAPITAL_USD * expo * TP_PCT * LEV
    rr         = (reward_usd / risk_usd) if risk_usd > 0 else np.nan

    # Décision
    if rri > RRI_OUT:
        decision, rationale = "CLOSE (FLAT)", "RRI > seuil sortie"
    elif (rri <= RRI_HALF) and (macd >= sig) and (45 <= rsi14 <= 70):
        decision, rationale = "BUY", "Zone verte + MACD haussier + RSI 45–70"
    elif (rri <= RRI_HALF) and (rsi14 > 70):
        decision, rationale = "HOLD", "Zone verte mais RSI>70 (surachat)"
    elif (RRI_HALF < rri <= RRI_OUT) and (macd >= sig) and (rsi14 >= 45):
        decision, rationale = "HOLD", "Zone orange mais momentum positif"
    elif SHORT_ALLOWED and (rri >= 55) and (macd < sig) and (rsi14 < 50):
        decision, rationale = "SELL (short)", "Risque élevé + momentum négatif"
    else:
        decision, rationale = "HOLD", "Pas de signal fort"

    # Synthèse & fraîcheur
    last_ts = df.index[-1]
    if last_ts.tz is None: last_ts = last_ts.tz_localize(timezone.utc)
    last_paris = last_ts.tz_convert(TZ_PARIS)
    freshness = "✅ données fraîches" if age_min <= FRESH_LIMIT_MIN else f"⚠️ données âgées ~{age_min:.0f} min"
    analysis = " | ".join([
        (f"Tendance {trend} (EMA50{'>' if (not np.isnan(ema50) and not np.isnan(ema200) and ema50>=ema200) else '<'}EMA200)" if not np.isnan(ema50) and not np.isnan(ema200) else "Tendance: n/a"),
        f"Momentum MACD: {'haussier' if macd>=sig else 'baissier'} ; RSI={rsi14:.1f}",
        f"Vol proxy (VXN) ~{vxn:.1f}% ; US10Y={us10y:.2f}% ; P/C={pc:.2f}"
    ])

    return dict(symbol=symbol, price=last, sl=sl, tp=tp,
                dist_sl=dist_to_sl, dist_tp=dist_to_tp,
                rsi=rsi14, macd=macd, sig=sig, vxn=vxn, us10y=us10y, pc=pc,
                rri=rri, expo=expo, nominal=nominal, risk_usd=risk_usd,
                reward_usd=reward_usd, rr=rr, decision=decision, rationale=rationale,
                analysis=analysis, last_paris=last_paris.strftime('%Y-%m-%d %H:%M'),
                age_min=age_min, freshness=freshness)

# ---------- Message ----------
def fmt_usd(x):
    try: return f"${x:,.0f}".replace(",", " ")
    except: return f"{x:.0f}"

def build_msg(d):
    link = f"\n🔗 Dashboard: {DASHBOARD_URL}" if DASHBOARD_URL else ""
    return (
f"📈 Nasdaq Plan – Update\n"
f"Symbole: {d['symbol']}\n"
f"Prix: {d['price']:.2f}\n"
f"SL(-0.9%): {d['sl']:.2f} (ΔSL {d['dist_sl']:.1f} pts) | "
f"TP(+2.2%): {d['tp']:.2f} (ΔTP {d['dist_tp']:.1f} pts)\n"
f"RSI14: {d['rsi']:.1f} | MACD: {d['macd']:.2f}/{d['sig']:.2f}\n"
f"VXN: {d['vxn']:.1f}% | US10Y: {d['us10y']:.2f}% | P/C: {d['pc']:.2f}\n"
f"RRI: {d['rri']:.0f}% | Expo: {d['expo']:.3f}\n"
f"— Risk/Sizing (Capital {fmt_usd(CAPITAL_USD)}): Notionnel≈ {fmt_usd(d['nominal'])} | "
f"Risque@SL≈ {fmt_usd(d['risk_usd'])} | Gain@TP≈ {fmt_usd(d['reward_usd'])} | R/R≈ {d['rr']:.2f}\n"
f"— Stratégie: {d['decision']} — {d['rationale']}\n"
f"— Analyse: {d['analysis']}\n"
f"Horodatage (Paris): {now_paris().strftime('%Y-%m-%d %H:%M %Z')} | "
f"Dernier point: {d['last_paris']} | {d['freshness']}"
f"{link}"
)

# ---------- Main ----------
def main():
    try:
        now = now_paris()
        if now.hour not in TRIGGER_HOURS or not is_us_trading_day(now):
            print("⏱ Pas l'heure / jour non US → exit 0"); return 0
        d = compute_metrics()
        send_tg(build_msg(d))
        print("✅ Message envoyé"); return 0
    except Exception as e:
        print("⚠️ Run sans envoi:", repr(e))
        send_tg(f"⚠️ NDX Alerts: run sans envoi – {e}")
        return 0

if __name__ == "__main__":
    sys.exit(main())
