# utils.py — shared constants, data, models, helpers
import warnings; warnings.filterwarnings("ignore")
import pandas as pd
import numpy as np
import yfinance as yf
import feedparser
from datetime import datetime, timedelta
from statsmodels.tsa.arima.model import ARIMA
from sklearn.ensemble import RandomForestClassifier
import streamlit as st

# ── Colours ───────────────────────────────────────────────────
BG      = "#0a0a0f"
BG2     = "#111118"
BG3     = "#1a1a24"
BORDER  = "#2a2a38"
TEXT    = "#e8e8f0"
MUTED   = "#6b6b80"
ACCENT  = "#f5a623"
UP      = "#00d084"
DOWN    = "#ff3b5c"
BLUE    = "#4d9fff"
PURPLE  = "#9b6dff"

# rgba helpers — avoids hex+opacity TypeError in Plotly
def rgba(hex_col, alpha):
    h = hex_col.lstrip("#")
    r,g,b = int(h[0:2],16),int(h[2:4],16),int(h[4:6],16)
    return f"rgba({r},{g},{b},{alpha})"

# ── Chart helpers ─────────────────────────────────────────────
def DC():
    """Base dark chart layout — NO xaxis/yaxis to avoid duplicate key errors."""
    return dict(
        plot_bgcolor=BG2, paper_bgcolor=BG2,
        font=dict(family="Arial,sans-serif", size=11, color=TEXT),
        margin=dict(l=8, r=8, t=36, b=8),
        hovermode="x unified",
        legend=dict(bgcolor=BG3, bordercolor=BORDER,
                    borderwidth=1, font_color=TEXT),
        hoverlabel=dict(bgcolor=BG3, bordercolor=BORDER,
                        font_color=TEXT, font_size=12),
    )

def DCM(l=4, r=4, t=32, b=4):
    """Dark chart layout with custom margin."""
    d = DC()
    d["margin"] = dict(l=l, r=r, t=t, b=b)
    return d

# Standard axis configs — pass explicitly to each chart
XA  = dict(showgrid=True,  gridcolor=BORDER, zeroline=False, color=MUTED)
YA  = dict(showgrid=True,  gridcolor=BORDER, zeroline=False, color=MUTED)
YAN = dict(showgrid=False, color=MUTED)

# ── Tickers ───────────────────────────────────────────────────
INDICES = {
    "Nifty 50":   "^NSEI",
    "Sensex":     "^BSESN",
    "Nifty Bank": "^NSEBANK",
    "Nifty IT":   "^CNXIT",
    "Nifty Auto": "^CNXAUTO",
    "Nifty FMCG": "^CNXFMCG",
    "India VIX":  "^INDIAVIX",
}
FX = {"USD/INR":"INR=X","EUR/INR":"EURINR=X","GBP/INR":"GBPINR=X"}
COMMODITIES = {"Gold":"GC=F","Crude":"CL=F","Silver":"SI=F"}

NIFTY500 = {
    "RELIANCE.NS":  ("Reliance Industries","Energy"),
    "TCS.NS":       ("TCS","IT"),
    "HDFCBANK.NS":  ("HDFC Bank","Banks"),
    "INFY.NS":      ("Infosys","IT"),
    "ICICIBANK.NS": ("ICICI Bank","Banks"),
    "HINDUNILVR.NS":("HUL","FMCG"),
    "SBIN.NS":      ("SBI","Banks"),
    "BAJFINANCE.NS":("Bajaj Finance","NBFC"),
    "TATAMOTORS.NS":("Tata Motors","Auto"),
    "SUNPHARMA.NS": ("Sun Pharma","Pharma"),
    "LT.NS":        ("L&T","Cap Goods"),
    "WIPRO.NS":     ("Wipro","IT"),
    "HCLTECH.NS":   ("HCL Tech","IT"),
    "KOTAKBANK.NS": ("Kotak Bank","Banks"),
    "AXISBANK.NS":  ("Axis Bank","Banks"),
    "MARUTI.NS":    ("Maruti","Auto"),
    "TITAN.NS":     ("Titan","Consumer"),
    "NESTLEIND.NS": ("Nestle","FMCG"),
    "ASIANPAINT.NS":("Asian Paints","Consumer"),
    "ITC.NS":       ("ITC","FMCG"),
    "POWERGRID.NS": ("Power Grid","Energy"),
    "ONGC.NS":      ("ONGC","Energy"),
    "BPCL.NS":      ("BPCL","Energy"),
    "TATASTEEL.NS": ("Tata Steel","Metals"),
    "JSWSTEEL.NS":  ("JSW Steel","Metals"),
    "ADANIPORTS.NS":("Adani Ports","Cap Goods"),
    "BAJAJ-AUTO.NS":("Bajaj Auto","Auto"),
    "DRREDDY.NS":   ("Dr Reddy's","Pharma"),
    "CIPLA.NS":     ("Cipla","Pharma"),
    "EICHERMOT.NS": ("Eicher Motors","Auto"),
    "HEROMOTOCO.NS":("Hero MotoCorp","Auto"),
    "HINDALCO.NS":  ("Hindalco","Metals"),
    "BHARTIARTL.NS":("Airtel","Telecom"),
    "TECHM.NS":     ("Tech Mahindra","IT"),
    "ULTRACEMCO.NS":("UltraTech","Cement"),
    "DIVISLAB.NS":  ("Divi's Labs","Pharma"),
    "APOLLOHOSP.NS":("Apollo Hosp","Pharma"),
    "BAJAJFINSV.NS":("Bajaj Finserv","NBFC"),
    "CHOLAFIN.NS":  ("Chola Finance","NBFC"),
    "VEDL.NS":      ("Vedanta","Metals"),
    "COAL.NS":      ("Coal India","Energy"),
    "BRITANNIA.NS": ("Britannia","FMCG"),
    "DABUR.NS":     ("Dabur","FMCG"),
    "GRASIM.NS":    ("Grasim","Cement"),
}

SECTOR_INDICES = {
    "Bank":    "^NSEBANK", "IT":      "^CNXIT",
    "Auto":    "^CNXAUTO", "FMCG":    "^CNXFMCG",
    "Pharma":  "^CNXPHARMA","Metal":  "^CNXMETAL",
    "Energy":  "^CNXENERGY","Realty": "^CNXREALTY",
    "Infra":   "^CNXINFRA", "PSU Bank":"^CNXPSUBANK",
}

NEWS_FEEDS = [
    ("ET",  "https://economictimes.indiatimes.com/markets/rssfeeds/1977021501.cms"),
    ("MC",  "https://www.moneycontrol.com/rss/marketreports.xml"),
    ("BS",  "https://www.business-standard.com/rss/markets-106.rss"),
]

CPI_HISTORY = np.array([
    5.47,5.76,5.77,6.07,5.05,4.31,4.20,3.63,3.41,3.17,3.65,3.81,
    2.99,2.18,1.46,2.36,3.28,3.28,3.58,4.88,5.21,5.07,4.44,4.28,
    4.58,4.87,5.00,4.17,3.69,3.70,3.38,2.33,2.19,1.97,2.57,2.86,
    2.92,3.05,3.18,3.15,3.28,3.99,4.62,5.54,7.35,7.59,6.58,5.84,
    7.22,6.27,6.09,6.93,6.69,7.27,7.61,6.93,4.59,4.06,5.03,5.52,
    4.23,6.30,6.26,5.59,5.30,4.35,4.48,4.91,5.66,6.01,6.07,6.95,
    7.79,7.04,7.01,6.71,7.00,7.41,6.77,5.88,5.72,6.52,6.44,5.66,
    4.70,4.25,4.81,7.44,6.83,5.02,4.87,5.55,5.69,5.10,5.09,4.85,
    4.83,4.75,5.08,3.54,3.65,5.49,6.21,5.48,5.22,
])

MACRO = {
    "cpi":5.22,"repo":6.50,"real_rate":1.28,
    "gdp":5.40,"iip":3.20,"gst":1.87,
    "phase":"Goldilocks","rbi":"CUT","rbi_prob":42,
}

PLAYBOOK = {
    "Goldilocks":    {"over":["IT","Pvt Banks","Auto","Cap Goods","FMCG"],
                      "under":["Metals","Oil & Gas","Pharma"],"c":UP},
    "Reflation":     {"over":["Metals","Oil & Gas","PSU Banks","Infra"],
                      "under":["IT","FMCG","NBFCs"],"c":ACCENT},
    "Stagflation":   {"over":["FMCG","Pharma","Utilities","Gold"],
                      "under":["Auto","Cap Goods","Banks","IT"],"c":DOWN},
    "Deflation risk":{"over":["Gilt Funds","HFCs","Pharma","FMCG"],
                      "under":["Metals","PSU Banks","Commodities"],"c":BLUE},
}

SCENARIOS = {
    "RBI cuts 25bps": {
        "icon":"📉","sentiment":+0.7,
        "sectors":{"Banks":+2.1,"NBFCs":+3.5,"IT":+0.8,"Auto":+1.6,
                   "FMCG":+0.6,"Metals":-0.3,"Energy":-0.2,"Pharma":+0.4,
                   "Realty":+2.8,"Infra":+1.9}},
    "RBI hikes 25bps": {
        "icon":"📈","sentiment":-0.5,
        "sectors":{"Banks":-1.2,"NBFCs":-2.8,"IT":-0.5,"Auto":-1.4,
                   "FMCG":-0.4,"Metals":+0.2,"Energy":+0.1,"Pharma":-0.2,
                   "Realty":-3.1,"Infra":-1.5}},
    "Crude oil +20%": {
        "icon":"🛢️","sentiment":-0.4,
        "sectors":{"Energy":+4.2,"Metals":+1.1,"IT":-1.6,"Auto":-2.2,
                   "FMCG":-1.9,"Banks":-0.9,"NBFCs":-0.6,"Pharma":-0.4,
                   "Realty":-0.8,"Infra":+0.5}},
    "War / Geopolitical shock": {
        "icon":"⚠️","sentiment":-0.9,
        "sectors":{"Energy":+1.8,"Metals":+2.1,"IT":-2.5,"Auto":-2.8,
                   "FMCG":+1.3,"Banks":-3.2,"NBFCs":-2.1,"Pharma":+1.9,
                   "Realty":-2.5,"Infra":-1.2}},
    "CPI spikes to 7%": {
        "icon":"🔥","sentiment":-0.6,
        "sectors":{"FMCG":-1.8,"IT":-1.2,"Banks":-0.8,"Auto":-1.5,
                   "Metals":+1.2,"Energy":+0.9,"NBFCs":-2.1,"Pharma":-0.5,
                   "Realty":-2.2,"Infra":-0.8}},
    "Union Budget — populist": {
        "icon":"📋","sentiment":+0.5,
        "sectors":{"Infra":+3.5,"FMCG":+2.1,"Auto":+1.8,"Banks":+1.2,
                   "IT":-0.4,"Metals":+1.5,"Energy":+0.8,"Pharma":+0.9,
                   "Realty":+1.6,"NBFCs":+0.8}},
}

BULLISH = ['surge','rally','gain','record','growth','profit','beat',
           'upgrade','buy','positive','strong','rise','high','boom',
           'recovery','soar','jump','outperform','bullish','expand']
BEARISH = ['crash','fall','drop','loss','miss','downgrade','sell',
           'negative','weak','war','crisis','inflation','risk','concern',
           'plunge','tumble','slump','decline','bearish','contract']

# ── Data fetchers ─────────────────────────────────────────────
@st.cache_data(ttl=30)
def get_quote(ticker):
    try:
        info = yf.Ticker(ticker).fast_info
        p = float(getattr(info,"last_price",0) or 0)
        c = float(getattr(info,"previous_close",0) or 0)
        if p and c:
            return {"p":round(p,2),"c":round(c,2),
                    "chg":round(p-c,2),"pct":round((p/c-1)*100,2)}
    except Exception:
        pass
    return {"p":None,"c":None,"chg":0,"pct":0}

@st.cache_data(ttl=300)
def get_ohlcv(ticker, period="1y", interval="1d"):
    try:
        df = yf.download(ticker,period=period,interval=interval,
                         progress=False,auto_adjust=True)
        if df.empty: return pd.DataFrame()
        if isinstance(df.columns,pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        return df[["Open","High","Low","Close","Volume"]].dropna()
    except Exception:
        return pd.DataFrame()

@st.cache_data(ttl=30)
def get_bulk(tickers_tuple):
    rows = []
    for t in tickers_tuple:
        d = get_quote(t)
        if d["p"]:
            rows.append({
                "Ticker":t,"Symbol":t.replace(".NS",""),
                "Name":NIFTY500.get(t,("",""))[0],
                "Sector":NIFTY500.get(t,("","Unknown"))[1],
                "Price":d["p"],"Chg":d["chg"],"Pct":d["pct"],
            })
    return pd.DataFrame(rows) if rows else pd.DataFrame()

@st.cache_data(ttl=600)
def get_news():
    items = []
    for src,url in NEWS_FEEDS:
        try:
            feed = feedparser.parse(url)
            for e in feed.entries[:7]:
                try:
                    ts = datetime(*e.published_parsed[:6]).strftime("%H:%M")
                except Exception:
                    ts = "--:--"
                txt = e.title
                tl  = txt.lower()
                bull = sum(1 for w in BULLISH if w in tl)
                bear = sum(1 for w in BEARISH if w in tl)
                tot  = max(bull+bear,1)
                score= round((bull-bear)/tot,2)
                label= ("BULLISH" if score>0.2 else
                        "BEARISH" if score<-0.2 else "NEUTRAL")
                items.append({"time":ts,"src":src,"txt":txt[:120],
                    "url":getattr(e,"link","#"),
                    "score":score,"label":label})
        except Exception:
            continue
    items.sort(key=lambda x:x["time"],reverse=True)
    seen,out = set(),[]
    for i in items:
        k=i["txt"][:30]
        if k not in seen:
            seen.add(k); out.append(i)
    return out[:25]

@st.cache_data(ttl=86400)
def get_arima():
    try:
        fit = ARIMA(CPI_HISTORY,order=(2,1,2)).fit()
        fc  = fit.get_forecast(steps=6)
        mn  = fc.predicted_mean
        ci  = fc.conf_int(alpha=0.20)
        mn  = mn.values if hasattr(mn,"values") else np.array(mn)
        lo  = ci.iloc[:,0].values if hasattr(ci,"iloc") else np.array(ci)[:,0]
        hi  = ci.iloc[:,1].values if hasattr(ci,"iloc") else np.array(ci)[:,1]
        dates = pd.date_range("2025-01-01",periods=6,freq="MS")
        return dates,np.round(mn,2),np.round(lo,2),np.round(hi,2)
    except Exception:
        d = pd.date_range("2025-01-01",periods=6,freq="MS")
        f = np.full(6,CPI_HISTORY[-1])
        return d,f,f-0.5,f+0.5

@st.cache_data(ttl=86400)
def get_rbi_model():
    try:
        repo = np.array([
            6.50,6.50,6.25,6.25,6.25,6.00,6.00,6.00,6.25,6.50,6.50,6.50,
            6.25,6.00,5.75,5.40,5.15,5.15,4.40,4.00,4.00,4.00,4.00,4.00,
            4.40,4.90,5.40,5.90,6.25,6.50,6.50,6.50,6.50,6.50,6.50,6.50,
        ])
        cpi_s = CPI_HISTORY[-len(repo):]
        df = pd.DataFrame({"repo":repo,"cpi":cpi_s})
        for c in ["repo","cpi"]:
            df[f"{c}_l1"]=df[c].shift(1)
            df[f"{c}_l2"]=df[c].shift(2)
        df["real"]=df["repo"]-df["cpi"]
        df["dec"]=df["repo"].diff(1).apply(
            lambda x:"hike" if x>0.01 else("cut" if x<-0.01 else "hold"))
        df=df.dropna()
        X=df[["repo_l1","cpi_l1","repo_l2","cpi_l2","real"]]
        y=df["dec"]
        clf=RandomForestClassifier(n_estimators=200,random_state=42,
                                   class_weight="balanced")
        clf.fit(X,y)
        proba=clf.predict_proba(X.iloc[[-1]])[0]
        return {c:round(float(p)*100,1) for c,p in zip(clf.classes_,proba)}
    except Exception:
        return {"hike":25.0,"hold":33.0,"cut":42.0}

# ── Technical indicators ──────────────────────────────────────
def get_indicators(df):
    c=df["Close"]
    d=c.diff()
    g=d.clip(lower=0).rolling(14).mean()
    l=(-d.clip(upper=0)).rolling(14).mean()
    rs=g/l.replace(0,np.nan)
    rsi=100-100/(1+rs)
    ema12=c.ewm(span=12).mean()
    ema26=c.ewm(span=26).mean()
    macd=ema12-ema26
    sig=macd.ewm(span=9).mean()
    hist=macd-sig
    bm=c.rolling(20).mean()
    bs=c.rolling(20).std()
    ma20=bm; ma50=c.rolling(50).mean()
    return {
        "rsi":    round(float(rsi.iloc[-1]),1)   if not rsi.empty  else 50,
        "macd":   round(float(macd.iloc[-1]),3)  if not macd.empty else 0,
        "macd_sig":round(float(sig.iloc[-1]),3),
        "bb_up":  round(float((bm+2*bs).iloc[-1]),2),
        "bb_lo":  round(float((bm-2*bs).iloc[-1]),2),
        "bb_mid": round(float(bm.iloc[-1]),2),
        "ma20":   round(float(ma20.iloc[-1]),2),
        "ma50":   round(float(ma50.iloc[-1]),2) if not ma50.isna().all() else None,
        "rsi_s":  pd.Series(rsi.values,index=df.index),
        "macd_s": pd.Series(macd.values,index=df.index),
        "sig_s":  pd.Series(sig.values,index=df.index),
        "hist_s": pd.Series(hist.values,index=df.index),
        "bb_up_s":(bm+2*bs),"bb_lo_s":(bm-2*bs),"bb_mid_s":bm,
        "ma20_s": ma20,"ma50_s":ma50,
    }

def get_beta(stock_df,index_df):
    try:
        sr=stock_df["Close"].pct_change().dropna()
        ir=index_df["Close"].pct_change().dropna()
        al=pd.concat([sr,ir],axis=1,join="inner").dropna()
        al.columns=["s","i"]
        cov=np.cov(al["s"],al["i"])
        return round(cov[0,1]/cov[1,1],3)
    except Exception:
        return 1.0

# ── CSS ───────────────────────────────────────────────────────
def inject_css():
    st.markdown(f"""
<style>
.stApp,[data-testid="stAppViewContainer"]{{background:{BG}!important;}}
.block-container{{padding:0.6rem 1rem 1.5rem!important;max-width:100%!important;}}
section[data-testid="stSidebar"]{{background:{BG2}!important;border-right:1px solid {BORDER}!important;}}
.card{{background:{BG2};border:1px solid {BORDER};border-radius:8px;padding:14px 16px;margin-bottom:10px;}}
.card-title{{font-size:9px;font-weight:700;letter-spacing:1.6px;text-transform:uppercase;color:{MUTED};border-bottom:1px solid {BORDER};padding-bottom:7px;margin-bottom:10px;}}
.tile{{background:{BG2};border:1px solid {BORDER};border-radius:7px;padding:10px 12px;text-align:center;}}
.tile-name{{font-size:9px;font-weight:700;color:{MUTED};letter-spacing:1px;text-transform:uppercase;}}
.tile-price{{font-size:19px;font-weight:700;color:{TEXT};line-height:1.1;margin:3px 0 1px;}}
.tile-chg{{font-size:12px;font-weight:600;}}
.up{{color:{UP}!important;}}.down{{color:{DOWN}!important;}}.flat{{color:{MUTED}!important;}}
.trow{{display:flex;justify-content:space-between;align-items:center;padding:5px 0;border-bottom:1px solid {BORDER};font-size:12px;color:{TEXT};}}
.tsym{{color:{BLUE};font-weight:600;min-width:90px;}}
.tpx{{color:{TEXT};font-weight:500;text-align:right;min-width:70px;}}
.tch{{font-weight:600;text-align:right;min-width:60px;}}
.nrow{{display:flex;gap:8px;padding:6px 0;border-bottom:1px solid {BORDER};align-items:flex-start;}}
.ntime{{font-size:10px;color:{MUTED};white-space:nowrap;min-width:34px;margin-top:2px;}}
.nsrc{{font-size:9px;font-weight:700;color:{ACCENT};background:{ACCENT}33;padding:2px 6px;border-radius:3px;white-space:nowrap;}}
.ntxt{{font-size:12px;color:{TEXT};line-height:1.4;}}
.nbull{{font-size:9px;font-weight:700;color:{UP};background:{UP}33;padding:1px 5px;border-radius:3px;white-space:nowrap;}}
.nbear{{font-size:9px;font-weight:700;color:{DOWN};background:{DOWN}33;padding:1px 5px;border-radius:3px;white-space:nowrap;}}
.nneut{{font-size:9px;font-weight:700;color:{MUTED};background:{MUTED}33;padding:1px 5px;border-radius:3px;white-space:nowrap;}}
.badge{{background:{BG3};border:1px solid {BORDER};border-radius:7px;padding:10px 14px;text-align:center;}}
.badge-lbl{{font-size:9px;color:{MUTED};font-weight:700;text-transform:uppercase;letter-spacing:1px;}}
.badge-val{{font-size:22px;font-weight:700;color:{TEXT};margin:2px 0 1px;line-height:1;}}
.badge-sub{{font-size:11px;color:{MUTED};}}
.sh{{font-size:10px;font-weight:700;letter-spacing:1.4px;text-transform:uppercase;color:{ACCENT};border-left:3px solid {ACCENT};padding-left:8px;margin:16px 0 8px;}}
.pill{{display:inline-block;padding:4px 12px;border-radius:20px;font-size:11px;font-weight:700;margin:2px;}}
#MainMenu,footer,header{{visibility:hidden;}}
.stDeployButton{{display:none;}}
</style>""", unsafe_allow_html=True)
