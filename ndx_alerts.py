# ============================================================
# 📊 NDX Alerts Bot - Production (robuste + analyse + actus)
# Envois Paris: 08h, 09h, 10h, 12h, 14h, 16h, 20h (jours US)
# Sélection dynamique du symbole (matin QQQ, US hours ^NDX), fallback NQ=F
# Analyse technique + contexte actus (Reuters/CNBC RSS) + sizing & niveaux
# ============================================================

import os, sys, time, re
from datetime import datetime, timezone, time as dt_time
import numpy as np
import pandas as pd
import requests
import pytz, holidays
import yfinance as yf
import feedparser

# ---------------- Config ----------------
TG_TOKEN = os.getenv("TG_TOKEN", "")
TG_CHAT  = os.getenv("TG_CHAT", "")

PRIMARY_SYMBOL = os.getenv("SYMBOL_PRICE", "^NDX")   # ^NDX (indice) ou QQQ (ETF)
DASHBOARD_URL  = os.getenv("DASHBOARD_URL", "")
CAPITAL_USD    = float(os.getenv("CAPITAL_USD", "100000"))
NEWS_WINDOW_MIN = int(os.getenv("NEWS_WINDOW_MIN", "180"))  # fenêtre actus (min)

TZ_PARIS = pytz.timezone("Europe/Paris")
TZ_NY    = pytz.timezone("America/New_York")

TRIGGER_HOURS = {8, 9, 10, 12, 14, 16, 20}          # heures Paris

# Stratégie
LEV, SL_PCT, TP_PCT, RISK = 20.0, -0.009, 0.022, 0.02  # SL -0.9%, TP +2.2%, 2% capital/trade
RRI_HALF, RRI_OUT = 10, 25
SHORT_ALLOWED = int(os.getenv("SHORT_ALLOWED", "1"))

# Fraîcheur acceptable (minutes) pour considérer la série "live"
FRESH_LIMIT_MIN = 90

# Flux d’actus (RSS)
NEWS_SOURCES = [
    "https://feeds.reuters.com/reuters/businessNews",
    "https://feeds.reuters.com/reuters/technologyNews",
    "https://www.cnbc.com/id/100003114/device/rss/rss.html",  # CNBC Markets
]

# ---------------- Utils ----------------
def now_paris() -> datetime:
    return datetime.now(TZ_PARIS)

def is_us_trading_day(d: datetime) -> bool:
    return (d.weekday() < 5) and (d.date() not in holidays.US())

def send_tg(text: str) -> None:
    if not TG_TOKEN or not TG_CHAT:
        print("Telegram non configuré (TG_TOKEN/TG_CHAT)."); return
    try:
        requests.get(
            f"https://api.telegram.org/bot{TG_TOKEN}/sendMessage",
            params={"chat_id": TG_CHAT, "text": text},
            timeout=20
        )
    except Exception as e:
        print("Telegram error:", e)

def rsi(series: pd.Series, n: int = 14) -> pd.Series:
    d = series.diff()
    up, dn = d.clip(lower=0), -d.clip(upper=0)
    ru = up.ewm(alpha=1/n, adjust=False).mean()
    rd = dn.ewm(alpha=1/n, adjust=False).mean().replace(0, np.nan)
    rs = ru / rd
    return (100 - 100/(1+rs)).fillna(50)

def ema_1d(arr: np.ndarray, n: int) -> pd.Series:
    x = np.asarray(arr, dtype=float).ravel()
    return pd.Series(x).ewm(span=n, adjust=False).mean()

# -------- Téléchargement ultra-robuste --------
def dl_one(ticker: str, period: str, interval: str) -> pd.DataFrame:
    try:
        df = yf.download(
            ticker, period=period, interval=interval,
            progress=False, threads=False, auto_adjust=False, prepost=True
        )
        if df is not None and not df.empty:
            return df
    except Exception as e:
        print("download err", ticker, period, interval, e)
    return pd.DataFrame()

def dl_history(ticker: str, period: str, interval: str) -> pd.DataFrame:
    try:
        df = yf.Ticker(ticker).history(
            period=period, interval=interval,
            prepost=True, actions=False, raise_errors=False
        )
        if df is not None and not df.empty:
            cols = {c: c.capitalize() for c in df.columns}
            return df.rename(columns=cols)
    except Exception as e:
        print("history err", ticker, period, interval, e)
    return pd.DataFrame()

def dl_robust_any(ticker: str) -> pd.DataFrame:
    # essais intraday → daily (avec double méthode download/history)
    attempts = [
        ("60d","30m"), ("60d","60m"), ("30d","15m"),
        ("7d","5m"),   ("1y","1d")
    ]
    for (p,i) in attempts:
        df = dl_one(ticker, p, i)
        if not df.empty: return df
        time.sleep(1.3)
        df = dl_history(ticker, p, i)
        if not df.empty: return df
        time.sleep(1.3)
    return pd.DataFrame()

def choose_symbol_by_clock() -> str:
    # Avant 14:30 Paris → QQQ (plus fiable le matin) ; sinon ^NDX (ou valeur choisie)
    t = now_paris().time()
    return "QQQ" if t < dt_time(14, 30) else (PRIMARY_SYMBOL or "^NDX")

def get_series_dynamic():
    """
    Essaie {symbole horaire, QQQ, ^NDX, NQ=F} et prend la série la plus fraîche.
    Renvoie (symbol, df, age_minutes). Peut être "stale" si Yahoo indispo, mais jamais vide si un des tickers répond.
    """
    pref = choose_symbol_by_clock()
    candidates = [pref] + [s for s in ["QQQ","^NDX","NQ=F"] if s != pref]

    best = (None, pd.DataFrame(), 10**9)
    for sym in candidates:
        df = dl_robust_any(sym)
        if df.empty:
            continue
        last_ts = df.index[-1]
        if last_ts.tz is None:
            last_ts = last_ts.tz_localize(timezone.utc)
        age_min = (datetime.now(timezone.utc) - last_ts.tz_convert(timezone.utc)).total_seconds()/60.0
        if age_min < best[2]:
            best = (sym, df, age_min)
        if age_min <= FRESH_LIMIT_MIN:
            return sym, df, age_min
    return best

# -------------- Actu ---------------
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
                if ts < cutoff:
                    continue
                title = getattr(e,"title","")
                summary = getattr(e,"summary","")
                items.append(title + " — " + summary)
        except Exception:
            continue
    # dédup basique
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
    if fed>0 and ("hawkish" in txt or "hot inflation" in txt or "higher for longer" in txt):
        score -= 2
    if mega>0 and any(w in txt for w in ["miss","probe","ban","downgrade"]):
        score -= 1

    label = "neutre"
    if score>=2: label="vent porteur"
    elif score<=-2: label="vent de face"

    bullets = []
    for s in items[:3]:
        s = re.sub(r"\s+"," ", s)
        bullets.append("• " + (s[:180] + ("…" if len(s)>180 else "")))

    drivers = []
    if fed>0: drivers.append("Fed/taux")
    if mega>0: drivers.append("méga-caps tech")
    if any(k in txt for k in ["earnings","results","guidance"]): drivers.append("résultats")
    if any(k in txt for k in ["cpi","ppi","pce","inflation"]): drivers.append("inflation")
    if any(k in txt for k in ["geopolit","sanction","ban "]): drivers.append("géopolitique/régulation")
    if not drivers: drivers.append("mix news")

    return dict(score=score, label=label, bullets=bullets, drivers=", ".join(drivers))

# ----------- Calculs marché ------------
def compute_metrics():
    symbol, df, age_min = get_series_dynamic()
    if df.empty:
        raise RuntimeError("Données introuvables")

    closes = df["Close"].dropna()
    last   = float(closes.iloc[-1])

    # RSI/MACD
    rsi14 = float(rsi(closes).iloc[-1])
    carr  = closes.to_numpy(dtype=float).ravel()
    ema12 = ema_1d(carr, 12); ema26 = ema_1d(carr, 26)
    macd_line = ema12 - ema26
    macd_sig  = macd_line.ewm(span=9, adjust=False).mean()
    macd, sig = float(macd_line.iloc[-1]), float(macd_sig.iloc[-1])

    # VXN proxy
    rets = closes.pct_change().dropna()
    vxn = float(rets.tail(20).std() * np.sqrt(252) * 100) if len(rets)>=20 else 20.0
    vxn = min(max(vxn, 10), 50)

    # US10Y / Put-Call
    tnx = dl_robust_any("^TNX")
    us10y = float(tnx["Close"].iloc[-1])/10.0 if not tnx.empty else 4.30
    cpc = dl_robust_any("^CPC")
    pc  = float(cpc["Close"].dropna().iloc[-1]) if not cpc.empty else 1.00

    # RRI
    rri = (0.25*(rsi14>70) + 0.25*(vxn>22) + 0.20*(macd<sig) +
           0.15*(us10y>4.5) + 0.15*(pc>1.1)) * 100

    # Niveaux/expo
    sl = last*(1+SL_PCT); tp = last*(1+TP_PCT)
    dist_to_sl = last - sl; dist_to_tp = tp - last
    expo_cap = RISK/(abs(SL_PCT)*LEV)       # ≈ 0.111
    base = 1.0 if rri<=RRI_HALF else (0.5 if rri<=RRI_OUT else 0.0)
    expo = min(base, expo_cap)

    # Sizing
    nominal = CAPITAL_USD*expo*LEV
    risk_usd = CAPITAL_USD*expo*abs(SL_PCT)*LEV
    reward_usd = CAPITAL_USD*expo*TP_PCT*LEV
    rr = (reward_usd/risk_usd) if risk_usd>0 else np.nan

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

    # Fraîcheur / horodatage
    last_ts = df.index[-1]
    if last_ts.tz is None: last_ts = last_ts.tz_localize(timezone.utc)
    last_paris = last_ts.tz_convert(TZ_PARIS).strftime('%Y-%m-%d %H:%M')
    freshness = "✅ données fraîches" if age_min <= FRESH_LIMIT_MIN else f"⚠️ données âgées ~{age_min:.0f} min"

    # Actu
    news = fetch_recent_news()
    news_ai = analyze_news(news)
    if news_ai["label"] == "vent de face" and decision == "BUY":
        decision, rationale = "HOLD", "Actu: vent de face (prudence)"

    analysis_news = f"Contexte actus: {news_ai['label']} (score {news_ai['score']}) ; drivers: {news_ai['drivers']}"
    top_lines = "\n".join(news_ai["bullets"])

    return dict(symbol=symbol, price=last, sl=sl, tp=tp,
                dist_sl=dist_to_sl, dist_tp=dist_to_tp,
                rsi=rsi14, macd=macd, sig=sig, vxn=vxn, us10y=us10y, pc=pc,
                rri=rri, expo=expo, nominal=nominal, risk_usd=risk_usd,
                reward_usd=reward_usd, rr=rr, decision=decision, rationale=rationale,
                last_paris=last_paris, freshness=freshness,
                analysis_news=analysis_news, news_lines=top_lines)

# -------------- Message ---------------
def fmt_usd(x: float) -> str:
    try: return f"${x:,.0f}".replace(",", " ")
    except: return f"{x:.0f}"

def build_msg(d: dict) -> str:
    link = f"\n🔗 Dashboard: {DASHBOARD_URL}" if DASHBOARD_URL else ""
    return (
f"📈 Nasdaq Plan – Update\n"
f"Symbole: {d['symbol']}\n"
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
        # Filet de sécurité : prévenir plutôt que planter
        msg = f"⚠️ NDX Alerts: run sans envoi – {e}"
        print(msg); 
        if TG_TOKEN and TG_CHAT:
            try: send_tg(msg)
            except: pass
        return 0

if __name__ == "__main__":
    sys.exit(main())
