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


def add_ind(df):
    d = df.copy()
    c, h, l = d["Close"], d["High"], d["Low"]
    d["MA"] = c.rolling(60).mean()
    lo, hi = l.rolling(60).min(), h.rolling(60).max()
    d["K"] = (100 * (c - lo) / (hi - lo + 1e-10)).rolling(3).mean()
    e12 = c.ewm(span=12, adjust=False).mean()
    e26 = c.ewm(span=26, adjust=False).mean()
    d["MACD"] = e12 - e26
    d["SIG"] = d["MACD"].ewm(span=9, adjust=False).mean()
    return d


def trend_check(d):
    ma = d["MA"].values
    valid = ma[~np.isnan(ma)]
    if len(valid) < 40:
        return "不明", 0.0
    now, before = valid[-1], valid[-40]
    slope = (now - before) / before * 100
    if slope > 3:
        return "往上", slope
    if slope < -3:
        return "往下", slope
    return "橫著走", slope


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

# 量價對買賣的加減分（僅供輔助，未經回測）
PV_SCORE = {
    "漲增": +2, "漲平": +1, "漲縮": -1,
    "平增":  0, "平平":  0, "平縮":  0,
    "跌縮": -1, "跌平": -2, "跌增": -3,
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

    d = add_ind(df)
    last, prev = d.iloc[-1], d.iloc[-2]
    price = float(last["Close"])
    ma_v = float(last["MA"])
    k_v, k_p = float(last["K"]), float(prev["K"])
    gap = float(last["MACD"]) - float(last["SIG"])
    trend, slope = trend_check(d)

    q1 = price > ma_v
    q2 = (k_v > 50) and (k_v > k_p)
    q3 = gap > 0
    hit = sum([q1, q2, q3])

    r = analyze_pv(df, days)
    # ============ 整合判斷 ============
    pv_score = PV_SCORE.get(r["combo"], 0)

    st.markdown("---")
    st.write("# 📌 結論")

    if trend == "往下":
        verdict = "避開"
        color = "error"
        headline = "🚫 避開　不要碰這檔"
        reason = [
            f"**這檔整體在跌**（近期平均價下滑 {abs(slope):.1f}%）。",
            "你的回測已經證明：股票本身在跌的時候，"
            "照這套規則做會賠得比「買了放著不動」更慘。",
        ]
    elif hit == 3 and pv_score >= 1:
        verdict = "買進"
        color = "success"
        headline = "🟢 買進　條件齊了"
        reason = [
            f"**趨勢往上**（平均價上升 {slope:.1f}%）",
            "**三條規則全過**：站上平均價、K值>50且上升、MACD轉強",
            f"**量價配合**：{r['combo']}（{PV_MEANING[r['combo']]['title'].split()[-1]}）",
        ]
    elif hit == 3 and pv_score < 0:
        verdict = "小心"
        color = "warning"
        headline = "🟡 小心　訊號到了但量價不對"
        reason = [
            "**三條規則全過**，時間點是對的",
            f"**但量價是「{r['combo']}」**，"
            f"{PV_MEANING[r['combo']]['why'].split('。')[0]}。",
            "價格漲但沒人跟著買，這種漲勢通常撐不久。",
        ]
    elif hit == 3:
        verdict = "可以看"
        color = "info"
        headline = "🟢 可以看　規則過了，量能普通"
        reason = [
            "**三條規則全過**",
            "量價沒有特別好也沒有特別差",
        ]
    elif pv_score <= -3:
        verdict = "賣出"
        color = "error"
        headline = "🔴 賣出　有人在倒貨"
        reason = [
            f"**量價是「{r['combo']}」** — 價格在跌，成交量卻放大。",
            "通常是持有大量股票的人在出貨。",
            f"技術面也只過 {hit}/3 條規則。",
            "**手上有的話考慮先走，沒有的話千萬別接。**",
        ]
    elif hit <= 1 and pv_score < 0:
        verdict = "賣出"
        color = "error"
        headline = "🔴 偏空　技術面和量價都不好"
        reason = [
            f"三條規則只過 {hit} 條",
            f"量價是「{r['combo']}」，偏弱",
        ]
    else:
        verdict = "等"
        color = "warning"
        headline = "🟡 等　條件還不夠"
        missing = []
        if not q1:
            missing.append("價格還沒站上平均價")
        if not q2:
            missing.append("K值不到50或在下降")
        if not q3:
            missing.append("MACD還沒轉強")
        reason = [
            f"**三條規則過了 {hit}/3**",
            "缺：" + "、".join(missing) if missing else "",
            f"量價：{r['combo']}",
        ]

    box = {"success": st.success, "error": st.error,
           "warning": st.warning, "info": st.info}[color]
    box(f"## {headline}")

    for line in reason:
        if line:
            st.write(f"　• {line}")

    # 操作建議
    if verdict == "買進":
        st.write("")
        sl = price * 0.85
        a, b, c = st.columns(3)
        a.metric("買進價（現在）", f"{price:,.2f}")
        b.metric("認賠出場價", f"{sl:,.2f}", "-15%")
        c.metric("平均價（跌破也走）", f"{ma_v:,.2f}")
        st.error("**部位大小：這套系統最多放你資金的 10-20%。**　"
                 "回測裡曾連續賠 14 次、資金縮水 48%。")
    elif verdict in ("賣出", "避開"):
        st.write("")
        st.write(f"**如果你手上有這檔：** 現價 {price:,.2f}，"
                 f"平均價 {ma_v:,.2f}。"
                 f"{'已經跌破平均價了。' if price < ma_v else '還在平均價上，但要盯緊。'}")

    st.caption("判斷依據：趨勢和三條規則是回測過的（每筆平均 +6.37% / +3.69%）；"
               "量價關係沒有回測過，只當輔助參考。")

    st.markdown("---")
    st.write("## 詳細數字")

    e1, e2, e3, e4 = st.columns(4)
    e1.metric("現價", f"{price:,.2f}")
    e2.metric("60日平均價", f"{ma_v:,.2f}", f"{(price/ma_v-1)*100:+.1f}%")
    e3.metric("K值", f"{k_v:.1f}", f"{k_v-k_p:+.1f}")
    e4.metric("三條規則", f"{hit}/3")

    st.markdown("---")
    st.write("## 量價細節")

    m = PV_MEANING.get(r["combo"], {
        "icon": "⚪", "title": f"{r['combo']}　狀況不明顯",
        "level": "neutral",
        "what": "價格和成交量都沒有明顯特徵。",
        "why": "現在沒有清楚的方向。",
        "watch": "再觀察一陣子。",
    })

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

    show = df.tail(120).copy()

    # 逐日標出量價訊號
    vma20 = df["Volume"].rolling(20).mean()
    buy_x, buy_y, buy_t = [], [], []
    sell_x, sell_y, sell_t = [], [], []

    for i in range(len(show)):
        idx = show.index[i]
        cl = float(show["Close"].iloc[i])
        op = float(show["Open"].iloc[i])
        hi = float(show["High"].iloc[i])
        lo = float(show["Low"].iloc[i])
        vol = float(show["Volume"].iloc[i])
        base = float(vma20.loc[idx]) if idx in vma20.index else np.nan
        if np.isnan(base) or base <= 0:
            continue
        ratio = vol / base
        pct = (cl - op) / op * 100

        if pct > 1.0 and ratio > 1.5:
            buy_x.append(idx); buy_y.append(lo * 0.985)
            buy_t.append(f"價漲量增<br>漲{pct:.1f}%　量{ratio:.1f}倍")
        elif pct < -1.0 and ratio > 1.5:
            sell_x.append(idx); sell_y.append(hi * 1.015)
            sell_t.append(f"價跌量增<br>跌{abs(pct):.1f}%　量{ratio:.1f}倍")

    fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                        row_heights=[0.72, 0.28], vertical_spacing=0.04,
                        subplot_titles=("價格（紅漲綠跌）", "成交量"))

    # 台股習慣：紅色漲、綠色跌
    fig.add_trace(go.Candlestick(
        x=show.index, open=show["Open"], high=show["High"],
        low=show["Low"], close=show["Close"], name="價格",
        increasing_line_color="#d62728", increasing_fillcolor="#d62728",
        decreasing_line_color="#2ca02c", decreasing_fillcolor="#2ca02c",
        ), row=1, col=1)

    fig.add_trace(go.Scatter(x=show.index,
                             y=show["Close"].rolling(20).mean(),
                             name="20日平均價",
                             line=dict(color="orange", width=1.5)), row=1, col=1)

    if buy_x:
        fig.add_trace(go.Scatter(
            x=buy_x, y=buy_y, mode="markers", name="🟢 價漲量增",
            marker=dict(symbol="triangle-up", size=13, color="#00A000",
                        line=dict(width=1, color="white")),
            text=buy_t, hoverinfo="text"), row=1, col=1)

    if sell_x:
        fig.add_trace(go.Scatter(
            x=sell_x, y=sell_y, mode="markers", name="🔴 價跌量增",
            marker=dict(symbol="triangle-down", size=13, color="#000000",
                        line=dict(width=1, color="white")),
            text=sell_t, hoverinfo="text"), row=1, col=1)

    vcolors = ["#d62728" if float(show["Close"].iloc[i]) >= float(show["Open"].iloc[i])
               else "#2ca02c" for i in range(len(show))]
    fig.add_trace(go.Bar(x=show.index, y=show["Volume"], name="成交量",
                         marker_color=vcolors, opacity=0.75), row=2, col=1)
    fig.add_trace(go.Scatter(x=show.index,
                             y=show["Volume"].rolling(20).mean(),
                             name="20日均量",
                             line=dict(color="blue", width=1.5)), row=2, col=1)

    fig.update_layout(height=600, xaxis_rangeslider_visible=False,
                      margin=dict(t=50, b=20), hovermode="x unified",
                      showlegend=True,
                      legend=dict(orientation="h", y=1.08))
    st.plotly_chart(fig, use_container_width=True)

    st.write(f"**圖上標記：** 🟢 綠色朝上三角 = 價漲量增（{len(buy_x)} 天）　"
             f"⚫ 黑色朝下三角 = 價跌量增（{len(sell_x)} 天）")
    st.caption("K線紅色=當天收漲、綠色=當天收跌（台股習慣）。"
               "下方柱子是成交量，藍線是20日均量，柱子明顯超過藍線就是放量。"
               "滑鼠移到三角形上可以看當天的漲跌幅和量倍數。")

    # 最近的訊號
    if buy_x or sell_x:
        st.write("")
        recent = []
        for x, t in zip(buy_x, buy_t):
            recent.append((x, "🟢 價漲量增", t.replace("<br>", "　").split("　", 1)[1]))
        for x, t in zip(sell_x, sell_t):
            recent.append((x, "⚫ 價跌量增", t.replace("<br>", "　").split("　", 1)[1]))
        recent.sort(key=lambda z: z[0], reverse=True)

        st.write("**最近 5 次訊號：**")
        for x, lab, detail in recent[:5]:
            st.write(f"　• {x.strftime('%Y/%m/%d')}　{lab}　{detail}")

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
