# pages/4_Portfolio_Alerts.py
# Portfolio Tracker · Price Alerts · Risk Metrics
import warnings; warnings.filterwarnings("ignore")
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from streamlit_autorefresh import st_autorefresh
from utils import (inject_css, dark_chart, q, ohlcv, bulk_quotes,
                   beta, NIFTY500,
                   BG, BG2, BG3, BORDER, TEXT, MUTED,
                   ACCENT, UP, DOWN, BLUE, PURPLE)

st.set_page_config(page_title="Portfolio · India Terminal",
                   page_icon="💼", layout="wide",
                   initial_sidebar_state="collapsed")
inject_css()
st_autorefresh(interval=60_000, key="port_refresh")

st.markdown(f"""
<div style="background:{BG2};border:1px solid {BORDER};border-radius:8px;
            padding:8px 18px;margin-bottom:12px;display:flex;
            align-items:center;gap:16px;">
  <span style="font-size:17px;font-weight:700;color:{TEXT};">
      💼 Portfolio & Alerts
  </span>
  <span style="font-size:11px;color:{MUTED};">
      Live P&amp;L · Risk metrics · Price alerts
  </span>
</div>""", unsafe_allow_html=True)


# ── Portfolio input ───────────────────────────────────────────
st.markdown(f'<div class="sh">My Portfolio</div>',
            unsafe_allow_html=True)
st.markdown(
    f'<div style="font-size:11px;color:{MUTED};margin-bottom:8px;">'
    f'Enter your holdings below. All data is stored only in your '
    f'browser session — nothing is saved to any server.</div>',
    unsafe_allow_html=True)

# Editable portfolio table
default_portfolio = pd.DataFrame({
    "Ticker (NSE)": ["RELIANCE","TCS","HDFCBANK","INFY","BAJFINANCE"],
    "Qty":          [10, 5, 15, 20, 8],
    "Avg Cost (₹)": [2750.0, 3800.0, 1580.0, 1420.0, 6800.0],
    "Target (₹)":   [3200.0, 4200.0, 1900.0, 1650.0, 8000.0],
    "Stop Loss (₹)": [2400.0, 3400.0, 1400.0, 1200.0, 5800.0],
})

portfolio = st.data_editor(
    default_portfolio,
    num_rows="dynamic",
    use_container_width=True,
    key="portfolio_editor",
    column_config={
        "Ticker (NSE)":  st.column_config.TextColumn("Ticker (NSE)", width="small"),
        "Qty":           st.column_config.NumberColumn("Qty", min_value=0),
        "Avg Cost (₹)":  st.column_config.NumberColumn("Avg Cost ₹", format="₹%.2f"),
        "Target (₹)":    st.column_config.NumberColumn("Target ₹", format="₹%.2f"),
        "Stop Loss (₹)": st.column_config.NumberColumn("Stop Loss ₹", format="₹%.2f"),
    },
)

if portfolio.empty:
    st.info("Add holdings using the table above.")
    st.stop()

# Fetch live prices for all holdings
tickers = [f"{t.strip().upper()}.NS"
           for t in portfolio["Ticker (NSE)"].dropna()
           if t.strip()]
quotes  = {t: q(t) for t in tickers}

# Build enriched portfolio
rows = []
for _, row in portfolio.iterrows():
    raw = str(row["Ticker (NSE)"]).strip().upper()
    if not raw: continue
    t    = f"{raw}.NS"
    dq   = quotes.get(t, {})
    lp   = dq.get("p") or float(row["Avg Cost (₹)"])
    qty  = float(row["Qty"] or 0)
    cost = float(row["Avg Cost (₹)"] or 0)
    tgt  = float(row["Target (₹)"]    or 0)
    sl   = float(row["Stop Loss (₹)"] or 0)
    cur_val  = qty * lp
    cost_val = qty * cost
    pnl      = cur_val - cost_val
    pnl_pct  = (lp/cost - 1)*100 if cost > 0 else 0
    tgt_pct  = (tgt/lp  - 1)*100 if lp  > 0 else 0
    sl_pct   = (sl/lp   - 1)*100 if lp  > 0 else 0
    hit_tgt  = lp >= tgt and tgt > 0
    hit_sl   = lp <= sl  and sl  > 0

    rows.append({
        "Symbol":     raw,
        "Price":      lp,
        "Qty":        int(qty),
        "Cost":       cost,
        "Cur Value":  cur_val,
        "Cost Value": cost_val,
        "P&L ₹":      pnl,
        "P&L %":      pnl_pct,
        "Target":     tgt,
        "Tgt %":      tgt_pct,
        "Stop Loss":  sl,
        "SL %":       sl_pct,
        "Alert":      ("🎯 TARGET HIT" if hit_tgt else
                       "🛑 STOP HIT"   if hit_sl  else ""),
    })

pf_df = pd.DataFrame(rows)
if pf_df.empty:
    st.info("No valid holdings found.")
    st.stop()


# ── Portfolio summary metrics ─────────────────────────────────
tot_cur  = pf_df["Cur Value"].sum()
tot_cost = pf_df["Cost Value"].sum()
tot_pnl  = tot_cur - tot_cost
tot_pct  = (tot_cur/tot_cost - 1)*100 if tot_cost > 0 else 0
pnl_col  = UP if tot_pnl >= 0 else DOWN
arr      = "▲" if tot_pnl >= 0 else "▼"

m1,m2,m3,m4 = st.columns(4)
for col, (lbl, val, sub, clr) in zip(
        [m1,m2,m3,m4],[
    ("Portfolio Value", f"₹{tot_cur:,.0f}",
     f"Invested: ₹{tot_cost:,.0f}", TEXT),
    ("Total P&L",
     f"{arr} ₹{abs(tot_pnl):,.0f}",
     f"{tot_pct:+.2f}% overall", pnl_col),
    ("Holdings", str(len(pf_df)),
     f"{pf_df[pf_df['P&L %']>=0].shape[0]} profitable", UP),
    ("Alerts",
     str(pf_df[pf_df["Alert"]!=""].shape[0]),
     "targets or stops hit", ACCENT),
]):
    with col:
        st.markdown(f"""
        <div class="badge">
          <div class="badge-lbl">{lbl}</div>
          <div class="badge-val" style="color:{clr};">{val}</div>
          <div class="badge-sub">{sub}</div>
        </div>""", unsafe_allow_html=True)


# ── Alert banner ──────────────────────────────────────────────
alerts = pf_df[pf_df["Alert"] != ""]
if not alerts.empty:
    for _, r in alerts.iterrows():
        clr = UP if "TARGET" in r["Alert"] else DOWN
        st.markdown(f"""
        <div style="background:{clr}22;border:1px solid {clr}55;
                    border-radius:7px;padding:10px 16px;
                    margin:6px 0;display:flex;align-items:center;gap:12px;">
          <span style="font-size:18px;">{r["Alert"].split()[0]}</span>
          <div>
            <span style="font-size:14px;font-weight:700;color:{clr};">
                {r["Alert"]}
            </span>
            &nbsp;&nbsp;
            <span style="font-size:13px;color:{TEXT};">
                {r["Symbol"]} — Current ₹{r["Price"]:,.2f}
            </span>
          </div>
        </div>""", unsafe_allow_html=True)


# ── P&L chart ─────────────────────────────────────────────────
chart_col, table_col = st.columns([1.6, 1.4])

with chart_col:
    st.markdown(f'<div class="card"><div class="card-title">'
                f'P&L by Stock</div>', unsafe_allow_html=True)

    fig_pnl = go.Figure(go.Bar(
        x=pf_df["Symbol"],
        y=pf_df["P&L %"],
        marker=dict(
            color=pf_df["P&L %"],
            colorscale=[[0,DOWN],[0.5,"#333"],[1,UP]],
            cmin=-20, cmax=20,
        ),
        text=[f"{v:+.2f}%" for v in pf_df["P&L %"]],
        textposition="outside",
        textfont=dict(color=TEXT, size=11),
        hovertemplate=(
            "<b>%{x}</b><br>"
            "P&L: %{y:+.2f}%<br>"
            "₹%{customdata:,.0f}<extra></extra>"),
        customdata=pf_df["P&L ₹"],
    ))
    dc = dark_chart()
    fig_pnl.update_layout(
        **dc, height=280, showlegend=False,
        title=dict(text="Unrealised P&L %", font_size=12),
        margin=dict(l=4,r=4,t=32,b=4),
        xaxis=dict(showgrid=False,color=TEXT),
        yaxis=dict(ticksuffix="%",showgrid=True,
                   gridcolor=BORDER,color=MUTED),
    )
    st.plotly_chart(fig_pnl, use_container_width=True,
                    config={"displayModeBar":False})

    # Portfolio allocation pie
    fig_pie = go.Figure(go.Pie(
        labels=pf_df["Symbol"],
        values=pf_df["Cur Value"],
        hole=0.5,
        marker=dict(
            line=dict(color=BG2,width=2)),
        textfont=dict(color=TEXT,size=11),
        hovertemplate="<b>%{label}</b><br>₹%{value:,.0f} (%{percent})<extra></extra>",
    ))
    dc2 = dark_chart()
    fig_pie.update_layout(
        **dc2, height=220,
        title=dict(text="Portfolio allocation",font_size=11),
        showlegend=True,
        legend=dict(orientation="h",y=-0.08,
                    font_color=TEXT,bgcolor=BG2),
        margin=dict(l=4,r=4,t=32,b=4),
    )
    st.plotly_chart(fig_pie, use_container_width=True,
                    config={"displayModeBar":False})
    st.markdown("</div>", unsafe_allow_html=True)


with table_col:
    st.markdown(f'<div class="card"><div class="card-title">'
                f'Holdings Detail</div>', unsafe_allow_html=True)

    def style_pnl(val):
        if isinstance(val, (int, float)):
            if val > 0: return f"color:{UP};font-weight:600"
            if val < 0: return f"color:{DOWN};font-weight:600"
        return ""

    display_df = pf_df[[
        "Symbol","Price","Qty","P&L %","P&L ₹","Tgt %","Alert"
    ]].copy()
    st.dataframe(
        display_df.style
        .applymap(style_pnl, subset=["P&L %","P&L ₹","Tgt %"])
        .format({
            "Price":  "₹{:,.2f}",
            "P&L %":  "{:+.2f}%",
            "P&L ₹":  "₹{:+,.0f}",
            "Tgt %":  "{:+.1f}%",
        }),
        use_container_width=True, height=280,
    )

    # Target vs stop visualisation
    st.markdown(f'<div class="sh" style="margin-top:8px;">'
                f'Price vs Target vs Stop Loss</div>',
                unsafe_allow_html=True)
    for _, r in pf_df.iterrows():
        if r["Target"] > 0 and r["Stop Loss"] > 0:
            rng   = r["Target"] - r["Stop Loss"]
            if rng <= 0: continue
            pos_pct = min(100, max(0,
                (r["Price"]-r["Stop Loss"])/rng*100))
            bar_col = (UP if r["Price"] >= r["Cost"]
                       else ACCENT if pos_pct > 50 else DOWN)
            st.markdown(f"""
            <div style="margin:6px 0;">
              <div style="display:flex;justify-content:space-between;
                          font-size:10px;color:{MUTED};margin-bottom:2px;">
                <span>{r["Symbol"]}</span>
                <span style="color:{bar_col};">₹{r["Price"]:,.1f}</span>
              </div>
              <div style="background:{BORDER};border-radius:4px;
                          height:8px;position:relative;">
                <div style="background:{bar_col};width:{pos_pct:.0f}%;
                            height:100%;border-radius:4px;"></div>
              </div>
              <div style="display:flex;justify-content:space-between;
                          font-size:9px;color:{MUTED};">
                <span>SL ₹{r["Stop Loss"]:,.0f}</span>
                <span style="color:{UP};">Tgt ₹{r["Target"]:,.0f}</span>
              </div>
            </div>""", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)


# ── Price Alerts ──────────────────────────────────────────────
st.markdown(f'<div class="sh" style="margin-top:24px;">'
            f'Custom Price Alerts</div>', unsafe_allow_html=True)

default_alerts = pd.DataFrame({
    "Ticker":    ["RELIANCE","TCS","HDFCBANK"],
    "Alert when price is":["above","below","above"],
    "Target Price (₹)":   [3000.0, 3500.0, 1800.0],
    "Note":               ["Breakout","Pullback buy","Resistance"],
})
alert_df = st.data_editor(
    default_alerts, num_rows="dynamic",
    use_container_width=True, key="alert_editor",
    column_config={
        "Alert when price is": st.column_config.SelectboxColumn(
            options=["above","below"]),
        "Target Price (₹)": st.column_config.NumberColumn(
            format="₹%.2f"),
    },
)

if not alert_df.empty:
    st.markdown(
        f'<div class="sh">Alert Status</div>',
        unsafe_allow_html=True)
    for _, r in alert_df.iterrows():
        if not r["Ticker"]: continue
        t  = f"{str(r['Ticker']).strip().upper()}.NS"
        dq = q(t)
        if not dq["p"]: continue
        lp     = dq["p"]
        tgt    = float(r["Target Price (₹)"] or 0)
        cond   = str(r["Alert when price is"])
        fired  = (lp > tgt if cond=="above" else lp < tgt)
        flr    = UP if fired else MUTED
        ftext  = "🔔 TRIGGERED" if fired else "⏳ Waiting"
        st.markdown(f"""
        <div style="background:{flr}18;border:1px solid {flr}33;
                    border-radius:7px;padding:8px 14px;margin:4px 0;
                    display:flex;align-items:center;gap:16px;">
          <span style="font-weight:700;color:{flr};font-size:13px;">
              {ftext}
          </span>
          <span style="color:{TEXT};font-size:12px;">
              {r["Ticker"]} — {cond} ₹{tgt:,.2f}
          </span>
          <span style="color:{flr};font-size:12px;margin-left:auto;">
              Current: ₹{lp:,.2f}
          </span>
          <span style="color:{MUTED};font-size:11px;">
              {r["Note"]}
          </span>
        </div>""", unsafe_allow_html=True)
