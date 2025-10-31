# ============================================================
# 📊 NDX Alerts Bot – NDX-first (live synth depuis QQQ si besoin)
# Envois Paris: 08h, 09h, 10h, 12h, 14h, 16h, 20h (jours US)
# Matin: prix live via Finnhub QQQ → NDX = QQQ * K (K = Close(^NDX)/Close(QQQ))
# Après 14:30: ^NDX Yahoo si frais, sinon synthèse QQQ*K
# Analyse technique + actus (Reuters/CNBC) + distances SL/TP + lien dashboard
# Affiche source & fraîcheur
# ============================================================

import os, sys, time, re
from datetime import datetime, timezone, time as dt_time
import numpy as np
import pandas as pd
import requests
import pytz, holidays
import yfinance as yf
import feedparser
from pandas_datareader import data as pdr

# ---------------- Config ----------------
TG_TOKEN = os.getenv("TG_TOKEN", "")
TG_CHAT  = os.getenv("TG_CHAT", "")

DASHBOARD_URL   = os.getenv("DASHBOARD_URL", "")
CAPITAL_USD     = float(os.getenv("CAPITAL_USD", "100000"))
NEWS_WINDOW_MIN = int(os.getenv("NEWS_WINDOW_MIN", "180"))  # fenêtre actus (min)

TZ_PARIS = pytz.timezone("Europe/Paris")
TRIGGER_HOURS = {8, 11, 12, 14, 15, 16, 20}

# Paramètres stratégie
LEV, SL_PCT, TP_PCT, RISK = 20.0, -0.009, 0.022, 0.02
RRI_HALF, RRI_OUT = 10, 25
SHORT_ALLOWED = int(os.getenv("SHORT_ALLOWED", "1"))

# Fraîcheur acceptable
FRESH_LIMIT_MIN = 90

# Flux d’actus (RSS)
NEWS_SOURCES = [
    "https://feeds.reuters.com/reuters/businessNews",
    "https://feeds.reuters.com/reuters/technologyNews",
    "https://www.cnbc.com/id/100003114/device/rss/rss.html",
]

# ---------------- Utils ----------------
def now_paris(): return datetime.now(TZ_PARIS)
def is_us_trading_day(d): return (d.weekday() < 5) and (d.date() not in holidays.US())

def send_tg(text):
    if not TG_TOKEN or not TG_CHAT:
        print("Telegram non configuré"); return
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

# -------- Téléchargement multi-sources --------
def dl_one(ticker, period, interval):
    try:
        df = yf.download(ticker, period=period, interval=interval,
                         progress=False, threads=False, auto_adjust=False, prepost=True)
        if df is not None and not df.empty: return df
    except Exception as e:
        print("yfd err", ticker, period, interval, e)
    return pd.DataFrame()

def dl_history(ticker, period, interval):
    try:
        df = yf.Ticker(ticker).history(period=period, interval=interval,
                                       prepost=True, actions=False, raise_errors=False)
        if df is not None and not df.empty:
            return df.rename(columns={c: c.capitalize() for c in df.columns})
    except Exception as e:
        print("yfh err", ticker, period, interval, e)
    return pd.DataFrame()

def dl_stooq(ticker):
    try:
        sym = "QQQ.US" if ticker.upper()=="QQQ" else ticker
        df = pdr.DataReader(sym, "stooq")
        if df is not None and not df.empty:
            return df.sort_index()
    except Exception as e:
        print("stooq err", ticker, e)
    return pd.DataFrame()

def finnhub_quote(ticker):
    token = os.getenv("FINNHUB_TOKEN", "")
    if not token: return pd.DataFrame()
    try:
        r = requests.get("https://finnhub.io/api/v1/quote",
                         params={"symbol": ticker, "token": token}, timeout=15)
        js = r.json()
        if "c" in js and js["c"]:
            ts = pd.to_datetime(datetime.now(timezone.utc))
            c  = float(js["c"])
            return pd.DataFrame({"Close":[c], "Open":[c], "High":[c], "Low":[c]}, index=[ts])
    except Exception as e:
        print("finnhub err", ticker, e)
    return pd.DataFrame()

def dl_robust_any(ticker):
    attempts = [
        ("yfd", lambda: dl_one(ticker,"60d","30m")),
        ("yfd", lambda: dl_one(ticker,"60d","60m")),
        ("yfd", lambda: dl_one(ticker,"30d","15m")),
        ("yfd", lambda: dl_one(ticker,"7d","5m")),
        ("yfh", lambda: dl_history(ticker,"60d","30m")),
        ("yfh", lambda: dl_history(ticker,"60d","60m")),
        ("yfh", lambda: dl_history(ticker,"30d","15m")),
        ("yfd", lambda: dl_one(ticker,"1y","1d")),
        ("yfh", lambda: dl_history(ticker,"1y","1d")),
        ("stooq", lambda: dl_stooq(ticker)),
        ("finnhub", lambda: finnhub_quote(ticker)),
    ]
    for tag, fn in attempts:
        df = fn()
        if df is not None and not df.empty:
            return df, tag
        time.sleep(0.8)
    return pd.DataFrame(), "none"

# ------------- Facteur NDX↔QQQ -------------
def ndx_qqq_scale_fallback():
    """K ~ 40 si aucun daily frais n’est dispo."""
    return 40.0

def ndx_qqq_scale_dynamic():
    """K = Close(^NDX)/Close(QQQ) (daily le plus récent)"""
    ndx_d, _ = dl_robust_any("^NDX")
    qqq_d, _ = dl_robust_any("QQQ")
    try:
        c_ndx = float(ndx_d["Close"].dropna().iloc[-1])
        c_qqq = float(qqq_d["Close"].dropna().iloc[-1])
        if c_ndx>0 and c_qqq>0:
            return c_ndx / c_qqq
    except Exception as e:
        print("scale err", e)
    return ndx_qqq_scale_fallback()

# ------------- Construction série NDX -------------
def get_ndx_series():
    """
    Renvoie une série/prix en NASDAQ-100 (NDX) + meta source/fraîcheur.
    - Avant 14:30 Paris: prix live via Finnhub (QQQ) → NDX = QQQ * K
    - Après 14:30: tente ^NDX intraday Yahoo ; sinon QQQ*K
    Retour: (symbol_label, df_ndx, age_min, src_note)
    """
    t = now_paris().time()
    K = ndx_qqq_scale_dynamic()  # dynamique via daily
    if t < dt_time(14, 30):
        # Hard Finnhub AM sur QQQ → synthèse NDX
        df_q = finnhub_quote("QQQ")
        if not df_q.empty:
            df_ndx = df_q.copy()
            for col in ["Open","High","Low","Close"]:
                df_ndx[col] = df_ndx[col] * K
            return f"^NDX (synth QQQ*K={K:.3f})", df_ndx, 0.0, "finnhub(qqq)*scale"
        # fallback QQQ via Yahoo/Stooq
        df_y, tag = dl_robust_any("QQQ")
        if not df_y.empty:
            last_ts = df_y.index[-1]
            if last_ts.tz is None: last_ts = last_ts.tz_localize(timezone.utc)
            age = (datetime.now(timezone.utc)-last_ts.tz_convert(timezone.utc)).total_seconds()/60.0
            for col in ["Open","High","Low","Close"]:
                df_y[col] = df_y[col] * K
            return f"^NDX (synth QQQ*K={K:.3f})", df_y, age, f"{tag}(qqq)*scale"
        # dernier recours : futur NQ=F approx NDX
        df_nq, tag = dl_robust_any("NQ=F")
        if not df_nq.empty:
            last_ts = df_nq.index[-1]
            if last_ts.tz is None: last_ts = last_ts.tz_localize(timezone.utc)
            age = (datetime.now(timezone.utc)-last_ts.tz_convert(timezone.utc)).total_seconds()/60.0
            return "NQ=F (proxy NDX)", df_nq, age, tag
        return "^NDX", pd.DataFrame(), 10**9, "none"

    # Après 14:30 → priorité ^NDX intraday
    df_ndx, tag = dl_robust_any("^NDX")
    if not df_ndx.empty:
        last_ts = df_ndx.index[-1]
        if last_ts.tz is None: last_ts = last_ts.tz_localize(timezone.utc)
        age = (datetime.now(timezone.utc)-last_ts.tz_convert(timezone.utc)).total_seconds()/60.0
        return "^NDX", df_ndx, age, tag

    # sinon, synthèse via QQQ*K
    df_q, tag = dl_robust_any("QQQ")
    if not df_q.empty:
        last_ts = df_q.index[-1]
        if last_ts.tz is None: last_ts = last_ts.tz_localize(timezone.utc)
        age = (datetime.now(timezone.utc)-last_ts.tz_convert(timezone.utc)).total_seconds()/60.0
        for col in ["Open","High","Low","Close"]:
            df_q[col] = df_q[col] * K
        return f"^NDX (synth QQQ*K={K:.3f})", df_q, age, f"{tag}(qqq)*scale"

    # dernier recours : NQ=F
    df_nq, tag = dl_robust_any("NQ=F")
    if not df_nq.empty:
        last_ts = df_nq.index[-1]
        if last_ts.tz is None: last_ts = last_ts.tz_localize(timezone.utc)
        age = (datetime.now(timezone.utc)-last_ts.tz_convert(timezone.utc)).total_seconds()/60.0
        return "NQ=F (proxy NDX)", df_nq, age, tag

    return "^NDX", pd.DataFrame(), 10**9, "none"

# -------------- Actu (RSS) ---------------
POS_WORDS = r"(beat|beats|tops|surprise|strong|accelerat|cooling inflation|soft landing|accommodative|dovish|upgrades?)"
NEG_WORDS = r"(miss|misses|below|weak|slowing|hot inflation|reaccelerat|hawkish|tighten|higher for longer|warning|probe|ban|sanction|downgrades?)"
FED_WORDS = r"(Powell|FOMC|Fed|Federal Reserve|rate|dot plot|hike|cut|QE|QT|inflation|CPI|PPI|PCE|jobs|payrolls|unemployment)"
MEGA_WORDS = r"(Nvidia|NVDA|Apple|AAPL|Microsoft|MSFT|Alphabet|GOOGL|GOOG|Meta|META|Amazon|AMZN|Tesla|TSLA|Broadcom|AVGO)"

def fetch_recent_news():
    items = []
    cutoff = datetime.now(timezone.utc).timestamp() - NEWS_WINDOW_MIN*60
    for url in NEWS_SOURCES:
        try:
            feed = feedparser.parse(url)
            for e in feed.entries[:30]:
                ts = None
                for k in ("published_parsed","updated_parsed"):
                    if getattr(e,k, None):
                        ts = datetime(*getattr(e,k)[:6], tzinfo=timezone.utc).timestamp()
                        break
                ts = ts or datetime.now(timezone.utc).timestamp()
                if ts < cutoff: continue
                title = getattr(e,"title",""); summary = getattr(e,"summary","")
                items.append(title + " — " + summary)
        except Exception: continue
    out, seen = [], set()
    for s in items:
        key = s[:90].lower()
        if key not in seen:
            seen.add(key); out.append(s)
    return out[:50]

def analyze_news(items):
    if not items:
        return dict(score=0, label="neutre", bullets=["Actu calme"], drivers="RAS")
    txt = " ".join(items).lower()
    pos = len(re.findall(POS_WORDS, txt))
    neg = len(re.findall(NEG_WORDS, txt))
    fed = len(re.findall(FED_WORDS, txt))
    mega= len(re.findall(MEGA_WORDS, txt))
    score = pos - neg
    if fed>0 and ("hawkish" in txt or "hot inflation" in txt or "higher for longer" in txt): score -= 2
    if mega>0 and any(w in txt for w in ["miss","probe","ban","downgrade"]): score -= 1
    label = "vent porteur" if score>=2 else ("vent de face" if score<=-2 else "neutre")
    bullets = []
    for s in items[:3]:
        s = re.sub(r"\s+"," ", s); bullets.append("• " + (s[:180] + ("…" if len(s)>180 else "")))
    drivers = []
    if fed>0: drivers.append("Fed/taux")
    if mega>0: drivers.append("méga-caps tech")
    if any(k in txt for k in ["earnings","results","guidance"]): drivers.append("résultats")
    if any(k in txt for k in ["cpi","ppi","pce","inflation"]): drivers.append("inflation")
    if any(k in txt for k in ["geopolit","sanction","ban "]): drivers.append("géopolitique/régulation")
    if not drivers: drivers.append("mix news")
    return dict(score=score, label=label, bullets=bullets, drivers=", ".join(drivers))

# ----------- Calculs marché (NDX) ------------
def compute_metrics():
    symbol_label, df, age_min, src_tag = get_ndx_series()
    if df.empty: raise RuntimeError("Données NDX introuvables")
    closes = df["Close"].dropna()
    last   = float(closes.iloc[-1])

    # RSI/MACD
    rsi14 = float(rsi(closes).iloc[-1])
    carr  = closes.to_numpy(dtype=float).ravel()
    ema12 = ema_1d(carr, 12); ema26 = ema_1d(carr, 26)
    macd_line = ema12 - ema26
    macd_sig  = macd_line.ewm(span=9, adjust=False).mean()
    macd, sig = float(macd_line.iloc[-1]), float(macd_sig.iloc[-1])

    # Vol proxy (annualisée)
    rets = closes.pct_change().dropna()
    vxn = float(rets.tail(20).std() * np.sqrt(252) * 100) if len(rets)>=20 else 20.0
    vxn = min(max(vxn, 10), 50)

    # Taux / Put-Call (daily)
    tnx, _ = dl_robust_any("^TNX")
    us10y = float(tnx["Close"].iloc[-1])/10.0 if not tnx.empty else 4.30
    cpc, _ = dl_robust_any("^CPC")
    pc  = float(cpc["Close"].dropna().iloc[-1]) if not cpc.empty else 1.00

    # RRI
    rri = (0.25*(rsi14>70) + 0.25*(vxn>22) + 0.20*(macd<sig) +
           0.15*(us10y>4.5) + 0.15*(pc>1.1)) * 100

    # Niveaux/expo (en NDX)
    sl = last*(1+SL_PCT); tp = last*(1+TP_PCT)
    dist_to_sl = last - sl; dist_to_tp = tp - last
    expo_cap = RISK/(abs(SL_PCT)*LEV)     # ≈ 0.111
    base = 1.0 if rri<=RRI_HALF else (0.5 if rri<=RRI_OUT else 0.0)
    expo = min(base, expo_cap)

    # Sizing
    nominal    = CAPITAL_USD*expo*LEV
    risk_usd   = CAPITAL_USD*expo*abs(SL_PCT)*LEV
    reward_usd = CAPITAL_USD*expo*TP_PCT*LEV
    rr         = (reward_usd/risk_usd) if risk_usd>0 else np.nan

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

    # Fraîcheur & actu
    last_ts = df.index[-1]
    if last_ts.tz is None: last_ts = last_ts.tz_localize(timezone.utc)
    last_paris = last_ts.tz_convert(TZ_PARIS).strftime('%Y-%m-%d %H:%M')
    freshness = "✅ données fraîches" if age_min <= FRESH_LIMIT_MIN else f"⚠️ données âgées ~{age_min:.0f} min"

    news = fetch_recent_news(); news_ai = analyze_news(news)
    if news_ai["label"] == "vent de face" and decision == "BUY":
        decision, rationale = "HOLD", "Actu: vent de face (prudence)"
    analysis_news = f"Contexte actus: {news_ai['label']} (score {news_ai['score']}) ; drivers: {news_ai['drivers']}"
    top_lines = "\n".join(news_ai["bullets"])

    return dict(symbol=symbol_label, price=last, sl=sl, tp=tp,
                dist_sl=dist_to_sl, dist_tp=dist_to_tp,
                rsi=rsi14, macd=macd, sig=sig, vxn=vxn, us10y=us10y, pc=pc,
                rri=rri, expo=expo, nominal=nominal, risk_usd=risk_usd,
                reward_usd=reward_usd, rr=rr, decision=decision, rationale=rationale,
                last_paris=last_paris, freshness=freshness,
                analysis_news=analysis_news, news_lines=top_lines,
                src_tag=src_tag)

# -------------- Message ---------------
def fmt_usd(x):
    try: return f"${x:,.0f}".replace(",", " ")
    except: return f"{x:.0f}"

def build_msg(d):
    link = f"\n🔗 Dashboard: {DASHBOARD_URL}" if DASHBOARD_URL else ""
    fresh_line = f"Source prix: {d.get('src_tag','?')} | Fraîcheur: {'OK' if '✅' in d.get('freshness','') else d.get('freshness','')}"
    return (
f"📈 Nasdaq Plan – Update (NDX)\n"
f"Symbole: {d['symbol']}\n"
f"{fresh_line}\n"
f"Prix: {d['price']:.2f}\n"
f"SL(-0.9%): {d['sl']:.2f} (ΔSL {d['dist_sl']:.1f} pts) | TP(+2.2%): {d['tp']:.2f} (ΔTP {d['dist_tp']:.1f} pts)\n"
f"RSI14: {d['rsi']:.1f} | MACD: {d['macd']:.2f}/{d['sig']:.2f} | VXN: {d['vxn']:.1f}% | US10Y: {d['us10y']:.2f}% | P/C: {d['pc']:.2f}\n"
f"RRI: {d['rri']:.0f}% | Expo: {d['expo']:.3f}\n"
f"— Risk/Sizing (Capital {fmt_usd(CAPITAL_USD)}): Notionnel≈ {fmt_usd(d['nominal'])} | Risque@SL≈ {fmt_usd(d['risk_usd'])} | Gain@TP≈ {fmt_usd(d['reward_usd'])} | R/R≈ {d['rr']:.2f}\n"
f"— Stratégie: {d['decision']} — {d['rationale']}\n"
f"— Actu (3× titres):\n{d['news_lines']}\n"
f"— Analyse IA (actu): {d['analysis_news']}\n"
f"Horodatage (Paris): {now_paris().strftime('%Y-%m-%d %H:%M %Z')} | Dernier point: {d['last_paris']} | {d['freshness']}"
f"{link}"
)

# ---------------- Main -----------------
def main():
    try:
        now = now_paris()
        if now.hour not in TRIGGER_HOURS or not is_us_trading_day(now):
            print("⏱ Pas l'heure/jour US → exit 0"); return 0
        d = compute_metrics()
        send_tg(build_msg(d))
        print("✅ Message envoyé"); return 0
    except Exception as e:
        msg = f"⚠️ NDX Alerts: run sans envoi – {e}"
        print(msg)
        if TG_TOKEN and TG_CHAT:
            try: send_tg(msg)
            except: pass
        return 0

if __name__ == "__main__":
    sys.exit(main())
