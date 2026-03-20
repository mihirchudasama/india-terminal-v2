# pages/2_Stock_Analyser.py
# Stock Deep Dive · Momentum Predictor · Correlation Explorer
import warnings; warnings.filterwarnings("ignore")
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from streamlit_autorefresh import st_autorefresh
from utils import (inject_css, dark_chart, q, ohlcv, bulk_quotes,
                   indicators, beta, momentum_score,
                   NIFTY500, INDICES, SECTOR_INDICES,
                   BG, BG2, BG3, BORDER, TEXT, MUTED,
                   ACCENT, UP, DOWN, BLUE, PURPLE)

st.set_page_config(page_title="Stock Analyser · India Terminal",
                   page_icon="📈", layout="wide",
                   initial_sidebar_state="collapsed")
inject_css()
st_autorefresh(interval=60_000, key="stock_refresh")


# ── Header ────────────────────────────────────────────────────
st.markdown(f"""
<div style="background:{BG2};border:1px solid {BORDER};border-radius:8px;
            padding:8px 18px;margin-bottom:12px;display:flex;
            align-items:center;gap:16px;">
  <span style="font-size:17px;font-weight:700;color:{TEXT};">
      📈 Stock Analyser
  </span>
  <span style="font-size:11px;color:{MUTED};">
      Deep dive · Technicals · Momentum · Correlations
  </span>
</div>""", unsafe_allow_html=True)


# ── Ticker input ──────────────────────────────────────────────
col_in, col_per = st.columns([3, 2])

with col_in:
    ticker_input = st.text_input(
        "Type any NSE ticker (e.g. RELIANCE, TCS, HDFCBANK)",
        value="RELIANCE",
        placeholder="RELIANCE, TCS, INFY, SBIN ...",
        label_visibility="collapsed",
    ).upper().strip()
    ticker = f"{ticker_input}.NS"

with col_per:
    p_map = {"1M":"1mo","3M":"3mo","6M":"6mo","1Y":"1y","3Y":"3y","5Y":"5y"}
    p_sel = st.radio("Period", list(p_map), horizontal=True, index=3,
                     label_visibility="collapsed")
    pk = p_map[p_sel]


# ── Load data ─────────────────────────────────────────────────
df = ohlcv(ticker, pk, "1d")
dq = q(ticker)
nifty_df = ohlcv("^NSEI", pk, "1d")

if df.empty or not dq["p"]:
    st.warning(f"No data found for **{ticker_input}**. "
               "Check the ticker symbol (e.g. RELIANCE not RELIANCE.NS)")
    st.stop()


# ── Compute all indicators ────────────────────────────────────
ind = indicators(df)
b   = beta(df, nifty_df) if not nifty_df.empty else 1.0
ms  = momentum_score(df)
price = dq["p"]
pct   = dq["pct"]
chg   = dq["chg"]
name  = NIFTY500.get(ticker, (ticker_input, "—"))[0]
sector= NIFTY500.get(ticker, ("—","Unknown"))[1]
p_clr = UP if pct >= 0 else DOWN
arr   = "▲" if pct >= 0 else "▼"

# Support / resistance (simple pivot)
hi_20 = float(df["High"].tail(20).max())
lo_20 = float(df["Low"].tail(20).min())
pivot = (hi_20 + lo_20 + price) / 3
r1    = 2*pivot - lo_20
s1    = 2*pivot - hi_20


# ── ROW 1: Key metrics ────────────────────────────────────────
m1,m2,m3,m4,m5,m6,m7 = st.columns(7)
metrics = [
    (name[:18], f"₹{price:,.2f}", f"{arr} {pct:+.2f}%", p_clr),
    ("Beta",    f"{b:.2f}",
     "High risk" if b>1.3 else "Low risk" if b<0.7 else "Market",
     DOWN if b>1.3 else UP if b<0.7 else MUTED),
    ("RSI",     f"{ind['rsi']}",
     "Overbought" if ind['rsi']>70 else "Oversold" if ind['rsi']<30 else "Neutral",
     DOWN if ind['rsi']>70 else UP if ind['rsi']<30 else MUTED),
    ("MACD",    f"{ind['macd']:+.2f}",
     "Bullish" if ind['macd']>ind['macd_sig'] else "Bearish",
     UP if ind['macd']>ind['macd_sig'] else DOWN),
    ("Momentum",f"{ms:+.1f}",
     "Strong" if ms>10 else "Weak" if ms<-5 else "Neutral",
     UP if ms>10 else DOWN if ms<-5 else MUTED),
    ("Support", f"₹{s1:,.1f}", "20-day pivot", BLUE),
    ("Resist.", f"₹{r1:,.1f}", "20-day pivot", ACCENT),
]
for col, (lbl, val, sub, clr) in zip(
        [m1,m2,m3,m4,m5,m6,m7], metrics):
    with col:
        st.markdown(f"""
        <div class="badge">
          <div class="badge-lbl">{lbl}</div>
          <div class="badge-val" style="color:{clr};">{val}</div>
          <div class="badge-sub">{sub}</div>
        </div>""", unsafe_allow_html=True)


# ── ROW 2: Main chart + Technical panels ─────────────────────
c1, c2 = st.columns([2.8, 1.2])

with c1:
    st.markdown(f'<div class="card"><div class="card-title">'
                f'{name} — Price + Technicals</div>',
                unsafe_allow_html=True)

    show_bb  = st.checkbox("Bollinger Bands", value=True)
    show_mas = st.checkbox("Moving Averages (20/50)", value=True)

    fig = make_subplots(
        rows=3, cols=1, shared_xaxes=True,
        row_heights=[0.58, 0.22, 0.20],
        vertical_spacing=0.02,
        subplot_titles=["", "RSI", "MACD"],
    )

    # Candlestick
    fig.add_trace(go.Candlestick(
        x=df.index, open=df["Open"], high=df["High"],
        low=df["Low"], close=df["Close"],
        increasing=dict(line_color=UP,  fillcolor=UP+"55"),
        decreasing=dict(line_color=DOWN,fillcolor=DOWN+"55"),
        name=ticker_input,
    ), row=1, col=1)

    # BB
    if show_bb:
        fig.add_trace(go.Scatter(
            x=df.index, y=ind["bb_up_s"],
            line=dict(color=PURPLE,width=0.8,dash="dot"),
            name="BB Upper", showlegend=False,
        ), row=1, col=1)
        fig.add_trace(go.Scatter(
            x=df.index, y=ind["bb_lo_s"],
            fill="tonexty", fillcolor=PURPLE+"18",
            line=dict(color=PURPLE,width=0.8,dash="dot"),
            name="BB Lower", showlegend=False,
        ), row=1, col=1)

    # MAs
    if show_mas:
        fig.add_trace(go.Scatter(
            x=df.index, y=ind["ma20_s"],
            line=dict(color=ACCENT,width=1.2),
            name="MA20",
        ), row=1, col=1)
        if ind["ma50"] and not ind["ma50_s"].isna().all():
            fig.add_trace(go.Scatter(
                x=df.index, y=ind["ma50_s"],
                line=dict(color=BLUE,width=1.2),
                name="MA50",
            ), row=1, col=1)

    # Support / Resistance lines
    fig.add_hline(y=s1, line_dash="dot",
                  line_color=UP,  line_width=1,
                  annotation_text=f"S1 {s1:,.0f}",
                  annotation_font_color=UP,
                  annotation_font_size=9, row=1, col=1)
    fig.add_hline(y=r1, line_dash="dot",
                  line_color=DOWN, line_width=1,
                  annotation_text=f"R1 {r1:,.0f}",
                  annotation_font_color=DOWN,
                  annotation_font_size=9, row=1, col=1)

    # RSI
    fig.add_trace(go.Scatter(
        x=df.index, y=ind["rsi_s"],
        line=dict(color=ACCENT,width=1.5),
        name="RSI", showlegend=False,
    ), row=2, col=1)
    fig.add_hline(y=70, line_dash="dot", line_color=DOWN+"88",
                  line_width=1, row=2, col=1)
    fig.add_hline(y=30, line_dash="dot", line_color=UP+"88",
                  line_width=1, row=2, col=1)

    # MACD
    fig.add_trace(go.Scatter(
        x=df.index, y=ind["macd_s"],
        line=dict(color=BLUE,width=1.5),
        name="MACD", showlegend=False,
    ), row=3, col=1)
    fig.add_trace(go.Scatter(
        x=df.index, y=ind["sig_s"],
        line=dict(color=ACCENT,width=1,dash="dot"),
        name="Signal", showlegend=False,
    ), row=3, col=1)
    hist_c = [UP if v>=0 else DOWN for v in ind["hist_s"]]
    fig.add_trace(go.Bar(
        x=df.index, y=ind["hist_s"],
        marker_color=hist_c, opacity=0.7,
        showlegend=False,
    ), row=3, col=1)

    dc = dark_chart(); dc.pop("xaxis",None); dc.pop("yaxis",None)
    fig.update_layout(
        **dc, height=500,
        title=dict(
            text=(f"{name}  "
                  f"<span style='color:{p_clr};'>"
                  f"₹{price:,.2f}  {arr} {pct:+.2f}%</span>"),
            font_size=13),
        xaxis_rangeslider_visible=False,
        showlegend=True,
        xaxis3=dict(showgrid=True,gridcolor=BORDER,color=MUTED),
        yaxis=dict(showgrid=True,gridcolor=BORDER,color=MUTED,
                   tickformat=",.0f"),
        yaxis2=dict(showgrid=True,gridcolor=BORDER,color=MUTED,
                    range=[0,100]),
        yaxis3=dict(showgrid=True,gridcolor=BORDER,color=MUTED),
    )
    st.plotly_chart(fig, use_container_width=True,
                    config={"displayModeBar":False})
    st.markdown("</div>", unsafe_allow_html=True)


with c2:
    # AI Signal Summary
    st.markdown(f'<div class="card"><div class="card-title">'
                f'AI Signal Summary</div>', unsafe_allow_html=True)

    signals = []
    if ind["rsi"] > 65:
        signals.append(("RSI Overbought","Caution — possible reversal",DOWN))
    elif ind["rsi"] < 35:
        signals.append(("RSI Oversold","Potential bounce zone",UP))
    else:
        signals.append(("RSI Neutral","No extreme signal",MUTED))

    if ind["macd"] > ind["macd_sig"] and ind["macd"] > 0:
        signals.append(("MACD Bullish","Upward momentum confirmed",UP))
    elif ind["macd"] < ind["macd_sig"] and ind["macd"] < 0:
        signals.append(("MACD Bearish","Downward momentum confirmed",DOWN))
    else:
        signals.append(("MACD Crossover","Watch for direction",ACCENT))

    if price > ind["ma20"]:
        signals.append(("Above MA20","Short-term bullish",UP))
    else:
        signals.append(("Below MA20","Short-term bearish",DOWN))

    if ind["ma50"] and price > ind["ma50"]:
        signals.append(("Above MA50","Medium-term bullish",UP))
    elif ind["ma50"]:
        signals.append(("Below MA50","Medium-term bearish",DOWN))

    if price > ind["bb_up"]:
        signals.append(("BB Breakout","Strong upside momentum",UP))
    elif price < ind["bb_lo"]:
        signals.append(("BB Breakdown","Possible oversold bounce",ACCENT))
    else:
        signals.append(("Inside BB","Normal range",MUTED))

    bull_sig = sum(1 for _, _, c in signals if c == UP)
    bear_sig = sum(1 for _, _, c in signals if c == DOWN)
    overall  = ("BUY" if bull_sig >= 3 else
                "SELL" if bear_sig >= 3 else "HOLD")
    ov_col   = UP if overall=="BUY" else DOWN if overall=="SELL" else MUTED

    st.markdown(f"""
    <div style="background:{ov_col}22;border:1px solid {ov_col}55;
                border-radius:8px;padding:10px;text-align:center;
                margin-bottom:12px;">
      <div style="font-size:10px;color:{ov_col};font-weight:700;
                  text-transform:uppercase;">Overall Signal</div>
      <div style="font-size:26px;font-weight:700;
                  color:{ov_col};">{overall}</div>
      <div style="font-size:11px;color:{MUTED};">
          {bull_sig} bullish · {bear_sig} bearish
      </div>
    </div>""", unsafe_allow_html=True)

    for title, detail, clr in signals:
        st.markdown(
            f'<div style="padding:6px 0;border-bottom:1px solid {BORDER};">'
            f'<div style="font-size:12px;color:{clr};font-weight:600;">'
            f'{title}</div>'
            f'<div style="font-size:11px;color:{MUTED};">{detail}</div>'
            f'</div>', unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)

    # Return summary
    st.markdown(f'<div class="card"><div class="card-title">'
                f'Returns</div>', unsafe_allow_html=True)
    ret_periods = [("1W",5),("1M",21),("3M",63),("6M",126),("1Y",252)]
    for lbl, days in ret_periods:
        if len(df) > days:
            r = (float(df["Close"].iloc[-1]) /
                 float(df["Close"].iloc[-days])-1)*100
            clr = UP if r >= 0 else DOWN
            arr2 = "▲" if r >= 0 else "▼"
            st.markdown(
                f'<div class="trow">'
                f'<span style="color:{MUTED};">{lbl}</span>'
                f'<span style="color:{clr};font-weight:600;">'
                f'{arr2} {r:+.2f}%</span>'
                f'</div>', unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)


# ── ROW 3: Momentum Screener ──────────────────────────────────
st.markdown(f'<div class="sh">Momentum Screener — Stocks About to Move</div>',
            unsafe_allow_html=True)
st.markdown(
    f'<div style="font-size:11px;color:{MUTED};margin-bottom:8px;">'
    f'Composite score = momentum (60%) + low-volatility (40%). '
    f'Positive = potential breakout. Negative = potential breakdown.</div>',
    unsafe_allow_html=True)

all_q = bulk_quotes(tuple(NIFTY500.keys()))

if not all_q.empty:
    c_filt1, c_filt2 = st.columns([2, 2])
    with c_filt1:
        sectors_all = ["All"] + sorted(all_q["Sector"].unique().tolist())
        sel_sec = st.selectbox("Filter sector", sectors_all,
                               label_visibility="collapsed")
    with c_filt2:
        sort_by = st.selectbox(
            "Sort",
            ["Momentum Score","1D Change %","Price"],
            label_visibility="collapsed")

    filt_df = all_q.copy()
    if sel_sec != "All":
        filt_df = filt_df[filt_df["Sector"]==sel_sec]

    sort_map = {
        "Momentum Score":"Pct",
        "1D Change %":   "Pct",
        "Price":         "Price",
    }
    filt_df = (filt_df.sort_values(sort_map[sort_by], ascending=False)
               .reset_index(drop=True))
    filt_df.index += 1

    # Bar chart of top 15
    top15 = filt_df.head(15)
    fig_m = go.Figure(go.Bar(
        x=top15["Symbol"],
        y=top15["Pct"],
        marker=dict(
            color=top15["Pct"],
            colorscale=[[0,DOWN],[0.5,"#333"],[1,UP]],
            cmin=-5, cmax=5,
        ),
        text=[f"{v:+.2f}%" for v in top15["Pct"]],
        textposition="outside",
        textfont=dict(color=TEXT, size=10),
        hovertemplate="<b>%{x}</b><br>₹%{customdata:,.1f}<extra></extra>",
        customdata=top15["Price"],
    ))
    dc = dark_chart()
    fig_m.update_layout(
        **dc, height=260, showlegend=False,
        title=dict(text="Top 15 by momentum (today's change %)",
                   font_size=11),
        margin=dict(l=4,r=4,t=32,b=4),
        xaxis=dict(showgrid=False,color=TEXT),
        yaxis=dict(ticksuffix="%",showgrid=True,
                   gridcolor=BORDER,color=MUTED),
    )
    st.plotly_chart(fig_m, use_container_width=True,
                    config={"displayModeBar":False})

    # Full table
    def colour_cell(val):
        if isinstance(val, float):
            if val > 0: return f"color: {UP}; font-weight:600"
            if val < 0: return f"color: {DOWN}; font-weight:600"
        return ""

    st.dataframe(
        filt_df[["Symbol","Name","Sector","Price","Pct","Chg"]]
        .rename(columns={"Pct":"Chg %","Chg":"Chg ₹"})
        .style
        .applymap(colour_cell, subset=["Chg %","Chg ₹"])
        .format({"Price":"₹{:,.2f}","Chg %":"{:+.2f}%",
                 "Chg ₹":"₹{:+.2f}"}),
        use_container_width=True, height=280,
    )


# ── ROW 4: Correlation Explorer ───────────────────────────────
st.markdown(f'<div class="sh">Correlation Explorer</div>',
            unsafe_allow_html=True)
st.markdown(
    f'<div style="font-size:11px;color:{MUTED};margin-bottom:8px;">'
    f'Compare any two instruments or macro indicators. '
    f'Correlation: +1 = move together, -1 = move opposite, 0 = no link.</div>',
    unsafe_allow_html=True)

all_instruments = (
    list(NIFTY500.keys()) +
    list(INDICES.values()) +
    list({"USD/INR":"INR=X","Gold":"GC=F","Crude":"CL=F"}.values())
)
all_labels = (
    [f"{v[0]} ({k.replace('.NS','')})" for k,v in NIFTY500.items()] +
    [f"Index: {k}" for k in INDICES.keys()] +
    ["FX: USD/INR","Commod: Gold","Commod: Crude"]
)
label_to_ticker = dict(zip(all_labels, all_instruments))

cc1, cc2, cc3 = st.columns([2, 2, 1])
with cc1:
    sel_a = st.selectbox("Instrument A", all_labels,
                         index=0, label_visibility="collapsed")
with cc2:
    sel_b = st.selectbox("Instrument B", all_labels,
                         index=2, label_visibility="collapsed")
with cc3:
    window = st.select_slider(
        "Rolling window",
        options=[30, 60, 90, 180, 252],
        value=60, label_visibility="collapsed")

t_a = label_to_ticker[sel_a]
t_b = label_to_ticker[sel_b]

df_a = ohlcv(t_a, "2y", "1d")
df_b = ohlcv(t_b, "2y", "1d")

if not df_a.empty and not df_b.empty:
    ret_a = df_a["Close"].pct_change().rename("A")
    ret_b = df_b["Close"].pct_change().rename("B")
    both  = pd.concat([ret_a, ret_b], axis=1).dropna()
    roll_corr = both["A"].rolling(window).corr(both["B"])
    overall_corr = round(float(both.corr().iloc[0,1]), 3)
    corr_col = (UP    if overall_corr > 0.3 else
                DOWN  if overall_corr < -0.3 else MUTED)

    cca, ccb = st.columns([2.5, 1.5])
    with cca:
        fig_c = make_subplots(rows=2, cols=1, shared_xaxes=True,
                              row_heights=[0.55, 0.45],
                              vertical_spacing=0.04)
        # Normalised price comparison
        norm_a = df_a["Close"] / float(df_a["Close"].iloc[0]) * 100
        norm_b = df_b["Close"] / float(df_b["Close"].iloc[0]) * 100
        fig_c.add_trace(go.Scatter(
            x=df_a.index, y=norm_a,
            mode="lines",line=dict(color=BLUE,width=1.5),
            name=sel_a[:25],
        ), row=1, col=1)
        fig_c.add_trace(go.Scatter(
            x=df_b.index, y=norm_b,
            mode="lines",line=dict(color=ACCENT,width=1.5),
            name=sel_b[:25],
        ), row=1, col=1)
        # Rolling correlation
        fig_c.add_trace(go.Scatter(
            x=roll_corr.index, y=roll_corr,
            mode="lines",
            line=dict(color=PURPLE,width=1.5),
            name=f"{window}d rolling corr",
            fill="tozeroy",
            fillcolor=PURPLE+"22",
        ), row=2, col=1)
        fig_c.add_hline(y=0, line_color=MUTED, line_width=0.8,
                        row=2, col=1)
        fig_c.add_hline(y=0.7,  line_dash="dot",
                        line_color=UP+"66",  line_width=1, row=2, col=1)
        fig_c.add_hline(y=-0.7, line_dash="dot",
                        line_color=DOWN+"66",line_width=1, row=2, col=1)
        dc2 = dark_chart(); dc2.pop("xaxis",None); dc2.pop("yaxis",None)
        fig_c.update_layout(
            **dc2, height=340,
            title=dict(
                text=f"Correlation: <span style='color:{corr_col};'>"
                     f"{overall_corr:+.3f}</span>",
                font_size=12),
            xaxis2=dict(showgrid=True,gridcolor=BORDER,color=MUTED),
            yaxis=dict(showgrid=True,gridcolor=BORDER,color=MUTED),
            yaxis2=dict(range=[-1.1,1.1],showgrid=True,
                        gridcolor=BORDER,color=MUTED),
        )
        st.plotly_chart(fig_c, use_container_width=True,
                        config={"displayModeBar":False})

    with ccb:
        st.markdown(f'<div class="card"><div class="card-title">'
                    f'Correlation Analysis</div>', unsafe_allow_html=True)
        st.markdown(f"""
        <div style="text-align:center;padding:12px;
                    background:{corr_col}18;border-radius:8px;
                    border:1px solid {corr_col}44;margin-bottom:12px;">
          <div style="font-size:11px;color:{MUTED};">
              Overall correlation
          </div>
          <div style="font-size:30px;font-weight:700;
                      color:{corr_col};">{overall_corr:+.3f}</div>
          <div style="font-size:11px;color:{corr_col};">
            {'Strong positive' if overall_corr>0.7 else
             'Moderate positive' if overall_corr>0.3 else
             'Strong negative' if overall_corr<-0.7 else
             'Moderate negative' if overall_corr<-0.3 else
             'Weak / no correlation'}
          </div>
        </div>
        """, unsafe_allow_html=True)

        # Interpretation
        if abs(overall_corr) > 0.7:
            interp = (f"These two move <b>strongly together</b>. "
                      f"When one moves, expect the other to follow.")
        elif abs(overall_corr) > 0.3:
            interp = (f"<b>Moderate</b> relationship. "
                      f"Partial co-movement — not always in sync.")
        else:
            interp = (f"<b>Weak relationship</b>. "
                      f"These instruments move independently.")

        st.markdown(
            f'<div style="font-size:12px;color:{TEXT};'
            f'line-height:1.6;margin-bottom:12px;">{interp}</div>',
            unsafe_allow_html=True)

        # Scatter plot
        fig_sc = go.Figure(go.Scatter(
            x=both["A"], y=both["B"],
            mode="markers",
            marker=dict(color=BLUE,size=3,opacity=0.5),
            hovertemplate=f"A: %{{x:.3%}}<br>B: %{{y:.3%}}<extra></extra>",
        ))
        dc3 = dark_chart()
        fig_sc.update_layout(
            **dc3, height=200,
            title=dict(text="Daily return scatter",font_size=10),
            margin=dict(l=4,r=4,t=28,b=4),
            xaxis=dict(tickformat=".1%",showgrid=True,
                       gridcolor=BORDER,color=MUTED),
            yaxis=dict(tickformat=".1%",showgrid=True,
                       gridcolor=BORDER,color=MUTED),
            showlegend=False,
        )
        st.plotly_chart(fig_sc, use_container_width=True,
                        config={"displayModeBar":False})
        st.markdown("</div>", unsafe_allow_html=True)
else:
    st.info("Loading correlation data...")
