# app.py — Page 1: Live Market Overview
# India Terminal v2 — Bloomberg Dark Theme
# ─────────────────────────────────────────
# DEPLOY:
#   1. Create new GitHub repo: india-terminal-v2
#   2. Upload: app.py, utils.py, requirements.txt
#      and the entire pages/ folder
#   3. Streamlit Cloud → New app → Main file: app.py
# ─────────────────────────────────────────

import warnings; warnings.filterwarnings("ignore")
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from streamlit_autorefresh import st_autorefresh
from datetime import datetime, timedelta
from utils import (inject_css, dark_chart, q, ohlcv, bulk_quotes,
                   news, arima_forecast, rbi_model,
                   INDICES, FX, COMMODITIES, SECTOR_INDICES,
                   NIFTY500, PLAYBOOK, MACRO, SCENARIOS,
                   BG, BG2, BG3, BORDER, TEXT, MUTED,
                   ACCENT, UP, DOWN, BLUE, PURPLE)

st.set_page_config(
    page_title="India Terminal",
    page_icon="🇮🇳",
    layout="wide",
    initial_sidebar_state="collapsed",
)
inject_css()

# ── Auto-refresh every 30 seconds ────────────────────────────
# This is the proper fix — no page reload, just data refresh
count = st_autorefresh(interval=30_000, key="market_refresh")

# ── IST clock ────────────────────────────────────────────────
ist = datetime.utcnow() + timedelta(hours=5, minutes=30)
ist_s = ist.strftime("%d %b %Y  %H:%M:%S IST")
mkt_open = (ist.weekday() < 5 and
            datetime(ist.year,ist.month,ist.day,9,15) <= ist
            <= datetime(ist.year,ist.month,ist.day,15,30))
mkt_col   = UP if mkt_open else DOWN
mkt_label = "● LIVE" if mkt_open else "● CLOSED"

st.markdown(f"""
<div style="display:flex;justify-content:space-between;align-items:center;
            background:{BG2};border:1px solid {BORDER};border-radius:9px;
            padding:9px 20px;margin-bottom:10px;">
  <div style="display:flex;align-items:center;gap:14px;">
    <span style="font-size:20px;font-weight:700;color:{TEXT};
                 letter-spacing:.5px;">🇮🇳 INDIA TERMINAL</span>
    <span style="font-size:10px;font-weight:700;color:{mkt_col};
                 background:{mkt_col}22;padding:3px 10px;
                 border-radius:4px;letter-spacing:.8px;">{mkt_label}</span>
    <span style="font-size:10px;color:{MUTED};">
        Refreshes every 30s · Refresh #{count}
    </span>
  </div>
  <span style="font-size:12px;color:{MUTED};
               font-variant-numeric:tabular-nums;">{ist_s}</span>
</div>
""", unsafe_allow_html=True)


# ── ROW 1: Index tiles ────────────────────────────────────────
cols = st.columns(7)
for col, (name, ticker) in zip(cols, INDICES.items()):
    d = q(ticker)
    p = d["p"]
    pct = d["pct"]
    clr = UP if pct > 0 else (DOWN if pct < 0 else MUTED)
    arr = "▲" if pct > 0 else ("▼" if pct < 0 else "—")
    with col:
        st.markdown(f"""
        <div class="tile">
          <div class="tile-name">{name}</div>
          <div class="tile-price">{f"{p:,.2f}" if p else "—"}</div>
          <div class="tile-chg" style="color:{clr};">
              {arr} {pct:+.2f}%
          </div>
        </div>""", unsafe_allow_html=True)

# Commodity + FX strip
strip_items = []
for name, ticker in {**FX, **COMMODITIES}.items():
    d = q(ticker)
    if d["p"]:
        clr = UP if d["pct"]>0 else DOWN
        arr = "▲" if d["pct"]>0 else "▼"
        strip_items.append(
            f'<span style="margin-right:24px;font-size:12px;">'
            f'<span style="color:{MUTED};font-weight:600;">{name}</span> '
            f'<span style="color:{TEXT};font-weight:700;">{d["p"]:,.3f}</span>'
            f'<span style="color:{clr};"> {arr}{d["pct"]:+.2f}%</span>'
            f'</span>')

st.markdown(
    f'<div style="background:{BG2};border:1px solid {BORDER};'
    f'border-radius:7px;padding:7px 16px;margin:6px 0;">'
    + ("".join(strip_items) or
       f'<span style="color:{MUTED};font-size:12px;">Loading...</span>')
    + "</div>", unsafe_allow_html=True)


# ── ROW 2: Nifty chart | Sector heatmap | Macro + RBI ────────
c1, c2, c3 = st.columns([2.5, 1.6, 1.1])

with c1:
    st.markdown(f'<div class="card"><div class="card-title">'
                f'Nifty 50 — Price Chart</div>', unsafe_allow_html=True)

    p_map = {"Today":"1d","5D":"5d","1M":"1mo","3M":"3mo",
             "1Y":"1y","5Y":"5y"}
    p_sel = st.radio("", list(p_map), horizontal=True, index=2,
                     label_visibility="collapsed")
    pk    = p_map[p_sel]
    ivl   = ("5m" if pk=="1d" else "15m" if pk=="5d" else
             "1h" if pk in ["1mo","3mo"] else "1d")

    df_n  = ohlcv("^NSEI", pk, ivl)
    if not df_n.empty:
        lp  = float(df_n["Close"].iloc[-1])
        fp  = float(df_n["Close"].iloc[0])
        chg = (lp/fp-1)*100
        clr = UP if chg >= 0 else DOWN

        fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                            row_heights=[0.78,0.22],
                            vertical_spacing=0.02)
        fig.add_trace(go.Candlestick(
            x=df_n.index,
            open=df_n["Open"],high=df_n["High"],
            low=df_n["Low"],  close=df_n["Close"],
            increasing=dict(line_color=UP,  fillcolor=UP+"44"),
            decreasing=dict(line_color=DOWN,fillcolor=DOWN+"44"),
            name="Nifty",
        ), row=1, col=1)
        if len(df_n) >= 20:
            fig.add_trace(go.Scatter(
                x=df_n.index,
                y=df_n["Close"].rolling(20).mean(),
                mode="lines",line=dict(color=ACCENT,width=1,dash="dot"),
                name="20 MA",showlegend=False,
            ), row=1, col=1)
        if len(df_n) >= 50:
            fig.add_trace(go.Scatter(
                x=df_n.index,
                y=df_n["Close"].rolling(50).mean(),
                mode="lines",line=dict(color=BLUE,width=1,dash="dot"),
                name="50 MA",showlegend=False,
            ), row=1, col=1)
        vol_c = [UP if c>=o else DOWN
                 for c,o in zip(df_n["Close"],df_n["Open"])]
        fig.add_trace(go.Bar(
            x=df_n.index, y=df_n["Volume"],
            marker_color=vol_c, opacity=0.5,
            showlegend=False,
        ), row=2, col=1)
        dc = dark_chart()
        dc.pop("xaxis", None); dc.pop("yaxis", None)
        fig.update_layout(
            **dc, height=340,
            title=dict(
                text=(f"Nifty 50  "
                      f"<span style='color:{clr};'>"
                      f"{lp:,.2f}  "
                      f"{'▲' if chg>=0 else '▼'} {chg:+.2f}%</span>"),
                font_size=13),
            xaxis_rangeslider_visible=False,
            showlegend=False,
            xaxis=dict(showgrid=True,gridcolor=BORDER,color=MUTED),
            yaxis=dict(tickformat=",.0f",showgrid=True,
                       gridcolor=BORDER,color=MUTED),
            yaxis2=dict(showgrid=False,color=MUTED),
            xaxis2=dict(showgrid=True,gridcolor=BORDER,color=MUTED),
        )
        st.plotly_chart(fig, use_container_width=True,
                        config={"displayModeBar":False})
    else:
        st.info("Chart loading...")
    st.markdown("</div>", unsafe_allow_html=True)


with c2:
    st.markdown(f'<div class="card"><div class="card-title">'
                f'Sector Heatmap — Today</div>', unsafe_allow_html=True)

    sec_data = []
    for name, ticker in SECTOR_INDICES.items():
        d = q(ticker)
        sec_data.append({"Sector":name,"Chg":d["pct"] if d["p"] else 0})
    sec_df = pd.DataFrame(sec_data).sort_values("Chg", ascending=False)

    # Treemap-style heatmap
    fig_h = go.Figure(go.Bar(
        y=sec_df["Sector"],
        x=sec_df["Chg"],
        orientation="h",
        marker=dict(
            color=sec_df["Chg"],
            colorscale=[[0,DOWN],[0.5,"#333"],[1,UP]],
            cmin=-3, cmax=3,
        ),
        text=[f"{v:+.2f}%" for v in sec_df["Chg"]],
        textposition="outside",
        textfont=dict(size=11,color=TEXT),
        hovertemplate="%{y}: <b>%{x:+.2f}%</b><extra></extra>",
    ))
    dc = dark_chart(); dc.pop("yaxis",None)
    fig_h.update_layout(
        **dc, height=340, showlegend=False,
        title=dict(text="NSE sector performance",font_size=12),
        margin=dict(l=4,r=50,t=36,b=4),
        xaxis=dict(ticksuffix="%",showgrid=True,
                   gridcolor=BORDER,color=MUTED),
        yaxis=dict(color=TEXT,showgrid=False),
    )
    st.plotly_chart(fig_h, use_container_width=True,
                    config={"displayModeBar":False})
    st.markdown("</div>", unsafe_allow_html=True)


with c3:
    st.markdown(f'<div class="card"><div class="card-title">'
                f'Macro Snapshot</div>', unsafe_allow_html=True)

    macro_rows = [
        ("CPI",       f"{MACRO['cpi']}%",
         UP if MACRO["cpi"]<4 else DOWN),
        ("Repo Rate", f"{MACRO['repo']}%",   MUTED),
        ("Real Rate", f"+{MACRO['real_rate']}%", UP),
        ("GDP",       f"{MACRO['gdp']}%",    UP),
        ("IIP",       f"{MACRO['iip']}%",    UP),
        ("GST",       f"₹{MACRO['gst']}L Cr", MUTED),
    ]
    for lbl, val, clr in macro_rows:
        st.markdown(
            f'<div class="trow">'
            f'<span style="color:{MUTED};">{lbl}</span>'
            f'<span style="color:{clr};font-weight:600;">{val}</span>'
            f'</div>', unsafe_allow_html=True)

    pb   = PLAYBOOK.get(MACRO["phase"], PLAYBOOK["Goldilocks"])
    probs= rbi_model()
    dec  = max(probs, key=probs.get)
    dcol = UP if dec=="cut" else (DOWN if dec=="hike" else MUTED)

    st.markdown(f"""
    <div style="background:{dcol}18;border:1px solid {dcol}44;
                border-radius:7px;padding:9px;text-align:center;
                margin:10px 0;">
      <div style="font-size:9px;font-weight:700;color:{dcol};
                  text-transform:uppercase;letter-spacing:1px;">
          RBI Next Move
      </div>
      <div style="font-size:22px;font-weight:700;color:{dcol};">
          {dec.upper()}
      </div>
      <div style="display:flex;justify-content:center;gap:8px;
                  margin-top:4px;">
        {''.join(f'<span style="font-size:10px;color:{MUTED};">'
                 f'{k.upper()} {v:.0f}%</span>'
                 for k,v in probs.items())}
      </div>
    </div>
    <div style="font-size:9px;font-weight:700;color:{pb["c"]};
                text-transform:uppercase;letter-spacing:1px;
                margin:8px 0 4px;">Phase: {MACRO["phase"]}</div>
    <div style="font-size:10px;color:{UP};margin-bottom:4px;">
        ▲ {' · '.join(pb["over"][:3])}
    </div>
    <div style="font-size:10px;color:{DOWN};">
        ▼ {' · '.join(pb["under"][:2])}
    </div>
    """, unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)


# ── ROW 3: Gainers/Losers | News with Sentiment ──────────────
c4, c5 = st.columns([1.4, 2.6])

with c4:
    st.markdown(f'<div class="card"><div class="card-title">'
                f'Gainers & Losers</div>', unsafe_allow_html=True)

    all_q = bulk_quotes(tuple(NIFTY500.keys()))
    if not all_q.empty:
        gainers = all_q[all_q["Pct"]>0].sort_values("Pct",
                  ascending=False).head(7)
        losers  = all_q[all_q["Pct"]<0].sort_values("Pct").head(7)

        st.markdown(f'<div class="sh">Top Gainers</div>',
                    unsafe_allow_html=True)
        for _, r in gainers.iterrows():
            st.markdown(
                f'<div class="trow">'
                f'<span class="tsym">{r["Symbol"]}</span>'
                f'<span class="tpx">₹{r["Price"]:,.1f}</span>'
                f'<span class="tch up">▲{r["Pct"]:.2f}%</span>'
                f'</div>', unsafe_allow_html=True)

        st.markdown(f'<div class="sh">Top Losers</div>',
                    unsafe_allow_html=True)
        for _, r in losers.iterrows():
            st.markdown(
                f'<div class="trow">'
                f'<span class="tsym">{r["Symbol"]}</span>'
                f'<span class="tpx">₹{r["Price"]:,.1f}</span>'
                f'<span class="tch down">▼{r["Pct"]:.2f}%</span>'
                f'</div>', unsafe_allow_html=True)
    else:
        st.info("Fetching market data...")
    st.markdown("</div>", unsafe_allow_html=True)


with c5:
    st.markdown(f'<div class="card"><div class="card-title">'
                f'Live News — Sentiment Scored</div>',
                unsafe_allow_html=True)

    items = news()

    # Overall sentiment meter
    if items:
        scores = [i["score"] for i in items]
        avg    = np.mean(scores)
        mkt_sent = ("BULLISH" if avg>0.1 else
                    "BEARISH" if avg<-0.1 else "NEUTRAL")
        sent_col = UP if avg>0.1 else (DOWN if avg<-0.1 else MUTED)
        bull_n = sum(1 for i in items if i["label"]=="BULLISH")
        bear_n = sum(1 for i in items if i["label"]=="BEARISH")

        st.markdown(f"""
        <div style="display:flex;gap:12px;margin-bottom:10px;
                    padding:8px 12px;background:{BG3};
                    border-radius:6px;align-items:center;">
          <div style="font-size:11px;color:{MUTED};">
              Market Sentiment
          </div>
          <div style="font-size:14px;font-weight:700;
                      color:{sent_col};">{mkt_sent}</div>
          <div style="font-size:11px;color:{MUTED};margin-left:auto;">
            <span style="color:{UP};">▲ {bull_n} bullish</span>
            &nbsp;&nbsp;
            <span style="color:{DOWN};">▼ {bear_n} bearish</span>
          </div>
        </div>""", unsafe_allow_html=True)

        for item in items[:16]:
            scol = (UP   if item["label"]=="BULLISH" else
                    DOWN if item["label"]=="BEARISH" else MUTED)
            slbl = (f'<span class="nbull">BULL</span>'
                    if item["label"]=="BULLISH" else
                    f'<span class="nbear">BEAR</span>'
                    if item["label"]=="BEARISH" else
                    f'<span class="nneut">NEUT</span>')
            event_tag = ""
            if item["event"]:
                ev_data = SCENARIOS.get(item["event"],{})
                icon    = ev_data.get("icon","📌")
                event_tag = (f'<span style="font-size:9px;'
                             f'color:{ACCENT};background:{ACCENT}22;'
                             f'padding:1px 5px;border-radius:3px;'
                             f'white-space:nowrap;margin-top:2px;">'
                             f'{icon} {item["event"][:20]}</span>')

            st.markdown(
                f'<div class="nrow">'
                f'<span class="ntime">{item["time"]}</span>'
                f'<span class="nsrc">{item["src"]}</span>'
                f'{slbl}'
                f'<span class="ntxt">'
                f'<a href="{item["url"]}" target="_blank" '
                f'style="color:{TEXT};text-decoration:none;">'
                f'{item["txt"]}</a>'
                f'</span>'
                f'{event_tag}'
                f'</div>', unsafe_allow_html=True)
    else:
        # Fallback placeholder news
        fallback = [
            ("ET","Markets open higher on positive global cues","BULL"),
            ("MC","FII net buyers; DII also adds positions","BULL"),
            ("BS","RBI policy expected to hold; liquidity adequate","NEUT"),
            ("ET","IT sector outperforms on strong US earnings","BULL"),
            ("MC","Auto dispatches strong; sector sees upgrades","BULL"),
            ("BS","Metals under pressure on China demand concerns","BEAR"),
            ("ET","Banking NPAs continue to moderate","BULL"),
            ("MC","Pharma exports hit record; margin outlook positive","BULL"),
        ]
        for src, txt, lbl in fallback:
            slbl = (f'<span class="nbull">BULL</span>' if lbl=="BULL"
                    else f'<span class="nbear">BEAR</span>'
                    if lbl=="BEAR"
                    else f'<span class="nneut">NEUT</span>')
            st.markdown(
                f'<div class="nrow">'
                f'<span class="ntime">--:--</span>'
                f'<span class="nsrc">{src}</span>'
                f'{slbl}'
                f'<span class="ntxt">{txt}</span>'
                f'</div>', unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)


# ── FOOTER ────────────────────────────────────────────────────
st.markdown(f"""
<div style="text-align:center;color:{MUTED};font-size:10px;
            padding:10px 0 2px;border-top:1px solid {BORDER};
            margin-top:8px;">
  India Terminal v2 &nbsp;·&nbsp;
  NSE via yfinance &nbsp;·&nbsp; RBI · MOSPI · GST Council &nbsp;·&nbsp;
  Auto-refreshes every 30s &nbsp;·&nbsp; {ist_s}
</div>""", unsafe_allow_html=True)
