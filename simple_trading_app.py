import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.graph_objects as go
from plotly.subplots import make_subplots

st.set_page_config(page_title="量價分析", page_icon="📊", layout="wide")
st.title("📊 量價分析")
st.caption("看價格和成交量的搭配，判斷是真漲還是假漲")


def fix_symbol(s):
    s = s.strip().upper().replace(" ", "")
    if s.endswith(".HK"):
        num = s[:-3].lstrip("0")
        if num.isdigit():
            s = num.zfill(4) + ".HK"
    return s


@st.cache_data(ttl=600, show_spinner=False)
def get_data(symbol, period="1y"):
    try:
        df = yf.download(symbol, interval="1d", period=period,
                         progress=False, auto_adjust=False)
        if df is None or df.empty:
            return None
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        df = df.dropna(subset=["Close"])
        return df if len(df) > 60 else None
    except Exception:
        return None


def analyze_pv(df, days=5):
    """判斷量價關係"""
    c = df["Close"]
    v = df["Volume"]

    price_now = float(c.iloc[-1])
    price_before = float(c.iloc[-1 - days])
    price_chg = (price_now - price_before) / price_before * 100

    vol_recent = float(v.tail(days).mean())
    vol_base = float(v.tail(60).mean())
    vol_ratio = vol_recent / vol_base if vol_base > 0 else 1.0

    # 分類門檻
    if price_chg > 2:
        p_state = "漲"
    elif price_chg < -2:
        p_state = "跌"
    else:
        p_state = "平"

    if vol_ratio > 1.3:
        v_state = "增"
    elif vol_ratio < 0.75:
        v_state = "縮"
    else:
        v_state = "平"

    return {
        "price_chg": price_chg,
        "vol_ratio": vol_ratio,
        "vol_recent": vol_recent,
        "vol_base": vol_base,
        "p_state": p_state,
        "v_state": v_state,
        "combo": p_state + v_state,
    }


# 六種組合的白話解讀
PV_MEANING = {
    "漲增": {
        "icon": "🟢", "title": "價漲量增　健康的上漲",
        "level": "good",
        "what": "價格在漲，而且成交量放大。",
        "why": "有人真的拿錢出來買，不是幾張小單把價格墊上去的。"
                "這是最健康的上漲型態。",
        "watch": "如果量放大到平常的三倍以上，要留意是不是「天量」—— "
                 "有時候爆大量之後反而見高點。",
    },
    "漲縮": {
        "icon": "🟡", "title": "價漲量縮　上漲沒力",
        "level": "warn",
        "what": "價格在漲，但成交量反而萎縮。",
        "why": "沒什麼人在買，價格是被少數幾張單推上去的。"
                "這種漲勢通常撐不久。",
        "watch": "常出現在一波漲勢的末端。追進去容易買在高點。",
    },
    "跌增": {
        "icon": "🔴", "title": "價跌量增　有人在出貨",
        "level": "bad",
        "what": "價格在跌，成交量卻放大。",
        "why": "有人急著賣，而且賣得很兇。通常是持有大量股票的人在倒貨，"
                "或是壞消息出來大家搶著逃。",
        "watch": "這是最危險的組合。手上有的話要考慮先走，"
                 "沒有的話千萬別急著接。",
    },
    "跌縮": {
        "icon": "🟡", "title": "價跌量縮　賣壓在減輕",
        "level": "neutral",
        "what": "價格還在跌，但成交量縮小了。",
        "why": "想賣的人差不多賣完了，賣壓在減輕。"
                "這是「可能」快跌完的訊號。",
        "watch": "注意是「可能」。跌完不代表馬上會漲，"
                 "常常會在底部橫盤很久。不要因為這樣就搶進。",
    },
    "平增": {
        "icon": "🔵", "title": "價平量增　有人在換手",
        "level": "neutral",
        "what": "價格幾乎沒動，但成交量放大。",
        "why": "有人在大量買、也有人在大量賣，籌碼在換手。"
                "可能是大戶在默默收集，也可能是在慢慢出貨 —— "
                "從量價本身分不出來。",
        "watch": "這是需要搭配籌碼資料（大戶持股）才能判斷的情況。"
                 "單看量價會誤判。",
    },
    "平縮": {
        "icon": "⚪", "title": "價平量縮　沒人理它",
        "level": "neutral",
        "what": "價格沒動，成交量也小。",
        "why": "市場對這檔沒興趣，多空都在觀望。",
        "watch": "這種時候進場會被時間磨掉，通常要等有量才會動。",
    },
    "漲平": {
        "icon": "🟢", "title": "價漲量持平　溫和上漲",
        "level": "neutral",
        "what": "價格在漲，成交量跟平常差不多。",
        "why": "沒有特別放量也沒有萎縮，是比較溫和的上漲。",
        "watch": "不算強也不算弱。要看接下來量能會不會跟上來。",
    },
    "跌平": {
        "icon": "🟡", "title": "價跌量持平　溫和下跌",
        "level": "neutral",
        "what": "價格在跌，成交量跟平常差不多。",
        "why": "沒有恐慌性賣壓，但也沒人急著進來接。",
        "watch": "溫水煮青蛙型的下跌，容易讓人一路凹單。",
    },
    "平平": {
        "icon": "⚪", "title": "價平量平　橫盤整理",
        "level": "neutral",
        "what": "價格和成交量都沒什麼變化。",
        "why": "市場在等消息，多空都不想先出手。",
        "watch": "這種時候沒什麼好做的，等它表態再說。",
    },
}


c1, c2 = st.columns([3, 1])
with c1:
    sym_in = st.text_input("股票代號", "2330.TW",
                           help="台股 2330.TW｜港股 0700.HK｜美股 AAPL")
with c2:
    days = st.selectbox("看最近幾天", [3, 5, 10, 20], index=1)

if st.button("分析", use_container_width=True, type="primary"):
    sym = fix_symbol(sym_in)
    if sym != sym_in.strip().upper():
        st.info(f"代號已修正：{sym_in} → **{sym}**")

    with st.spinner("查資料中..."):
        df = get_data(sym)

    if df is None:
        st.error(f"查不到 **{sym}**。台股加 .TW，港股湊四位數加 .HK")
        st.stop()

    r = analyze_pv(df, days)
    m = PV_MEANING.get(r["combo"], {
        "icon": "⚪", "title": f"{r['combo']}　狀況不明顯",
        "level": "neutral",
        "what": "價格和成交量都沒有明顯特徵。",
        "why": "現在沒有清楚的方向。",
        "watch": "再觀察一陣子。",
    })

    # ===== 主結論 =====
    st.markdown("---")
    if m["level"] == "good":
        st.success(f"## {m['icon']} {m['title']}")
    elif m["level"] == "bad":
        st.error(f"## {m['icon']} {m['title']}")
    elif m["level"] == "warn":
        st.warning(f"## {m['icon']} {m['title']}")
    else:
        st.info(f"## {m['icon']} {m['title']}")

    a, b, c = st.columns(3)
    a.metric(f"最近 {days} 天漲跌", f"{r['price_chg']:+.2f}%")
    b.metric("成交量", f"平常的 {r['vol_ratio']:.2f} 倍")
    c.metric("現價", f"{float(df['Close'].iloc[-1]):,.2f}")

    st.write("")
    st.write(f"**發生什麼事：** {m['what']}")
    st.write(f"**代表什麼：** {m['why']}")
    st.write(f"**要注意：** {m['watch']}")

    # ===== 爆量偵測 =====
    st.markdown("---")
    st.write("### 有沒有異常爆量？")

    v = df["Volume"]
    v_base = float(v.tail(60).mean())
    recent20 = v.tail(20)
    spikes = []
    for i in range(len(recent20)):
        val = float(recent20.iloc[i])
        if v_base > 0 and val / v_base >= 2.5:
            d = recent20.index[i]
            pc = df["Close"].iloc[-20 + i]
            po = df["Open"].iloc[-20 + i]
            direction = "紅K（收漲）" if pc > po else "黑K（收跌）"
            spikes.append((d.strftime("%m/%d"), val / v_base, direction))

    if spikes:
        st.write(f"最近 20 天有 **{len(spikes)}** 天爆量（超過平常 2.5 倍）：")
        for d, ratio, direction in spikes[-5:]:
            st.write(f"　• **{d}**　量是平常的 {ratio:.1f} 倍，{direction}")
        st.write("")
        st.info("**爆量在紅K** → 通常是有人大買，或利多消息\n\n"
                "**爆量在黑K** → 通常是有人大賣，要小心")
    else:
        st.write("最近 20 天沒有明顯爆量，量能算平穩。")

    # ===== 量能趨勢 =====
    st.markdown("---")
    st.write("### 量能是在放大還是萎縮？")

    v5 = float(v.tail(5).mean())
    v20 = float(v.tail(20).mean())
    v60 = float(v.tail(60).mean())

    unit = 1000 if sym.endswith(".TW") or sym.endswith(".TWO") else 1
    ulabel = "張" if unit == 1000 else "股"

    st.write(f"近5日均量 **{v5/unit:,.0f} {ulabel}**　"
             f"近20日 {v20/unit:,.0f} {ulabel}　"
             f"近60日 {v60/unit:,.0f} {ulabel}")

    if v5 > v20 > v60:
        st.success("### 📈 量能持續放大")
        st.write("越來越多人在交易這檔，市場注意力在增加。")
    elif v5 < v20 < v60:
        st.warning("### 📉 量能持續萎縮")
        st.write("越來越少人交易，市場興趣在退燒。這時候行情通常走不動。")
    else:
        st.info("### ▫️ 量能起伏，沒有明顯方向")

    # ===== 圖 =====
    st.markdown("---")
    st.write("### 走勢圖")

    show = df.tail(120)
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                        row_heights=[0.7, 0.3], vertical_spacing=0.05,
                        subplot_titles=("價格", "成交量"))

    fig.add_trace(go.Candlestick(
        x=show.index, open=show["Open"], high=show["High"],
        low=show["Low"], close=show["Close"], name="價格"), row=1, col=1)

    fig.add_trace(go.Scatter(x=show.index,
                             y=show["Close"].rolling(20).mean(),
                             name="20日平均價",
                             line=dict(color="orange", width=1.5)), row=1, col=1)

    colors = ["red" if float(show["Close"].iloc[i]) >= float(show["Open"].iloc[i])
              else "green" for i in range(len(show))]
    fig.add_trace(go.Bar(x=show.index, y=show["Volume"], name="成交量",
                         marker_color=colors, opacity=0.7), row=2, col=1)
    fig.add_trace(go.Scatter(x=show.index,
                             y=show["Volume"].rolling(20).mean(),
                             name="20日均量",
                             line=dict(color="blue", width=1.5)), row=2, col=1)

    fig.update_layout(height=560, xaxis_rangeslider_visible=False,
                      margin=dict(t=40, b=20), hovermode="x unified",
                      showlegend=True)
    st.plotly_chart(fig, use_container_width=True)
    st.caption("下面那排柱子是成交量。紅色=當天收漲、綠色=當天收跌。"
               "藍線是20日平均量，柱子超過藍線很多就是爆量。")

    # ===== 六種對照表 =====
    st.markdown("---")
    with st.expander("📖 六種量價組合對照表"):
        rows = []
        for k, val in PV_MEANING.items():
            rows.append({
                "組合": f"{val['icon']} {k}",
                "意思": val["what"],
                "代表": val["why"].split("。")[0] + "。",
            })
        st.dataframe(pd.DataFrame(rows), use_container_width=True,
                     hide_index=True)

        st.write("")
        st.write("**最重要的兩個：**")
        st.write("　🟢 **價漲量增** = 真的有人在買，健康")
        st.write("　🔴 **價跌量增** = 有人在倒貨，危險")

st.markdown("---")
st.warning("**要說清楚的事：** 量價關係是流傳很久的經驗法則，"
           "但**沒有經過回測驗證**。它提供的是一個看待市場的角度，"
           "不是可以直接照做的買賣訊號。")
st.caption("資料來源：Yahoo Finance。僅供研究，不是投資建議。")
