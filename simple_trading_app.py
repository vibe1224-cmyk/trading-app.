import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.graph_objects as go

st.set_page_config(page_title="股票紅綠燈", page_icon="🚦", layout="wide")
st.title("🚦 股票紅綠燈")
st.caption("有防呆設計，會自動擋掉危險的用法")

# ==================== 資料 ====================

@st.cache_data(ttl=600, show_spinner=False)
def get_data(symbol, interval, period):
    try:
        df = yf.download(symbol, interval=interval, period=period,
                         progress=False, auto_adjust=False)
        if df is None or df.empty:
            return None
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        df = df.dropna(subset=["Close"])
        return df if len(df) > 80 else None
    except Exception:
        return None


def add_ind(df):
    d = df.copy()
    c, h, l = d["Close"], d["High"], d["Low"]
    d["MA"] = c.rolling(60).mean()
    d["MA120"] = c.rolling(120).mean()
    lo, hi = l.rolling(60).min(), h.rolling(60).max()
    d["K"] = (100 * (c - lo) / (hi - lo + 1e-10)).rolling(3).mean()
    e12 = c.ewm(span=12, adjust=False).mean()
    e26 = c.ewm(span=26, adjust=False).mean()
    d["MACD"] = e12 - e26
    d["SIG"] = d["MACD"].ewm(span=9, adjust=False).mean()
    return d


def trend_check(d):
    """判斷這檔是往上還往下"""
    ma = d["MA"].values
    valid = ma[~np.isnan(ma)]
    if len(valid) < 40:
        return "資料不足", None
    now, before = valid[-1], valid[-40]
    slope = (now - before) / before * 100
    if slope > 3:
        return "往上", slope
    elif slope < -3:
        return "往下", slope
    else:
        return "橫著走", slope


MARKETS = {
    "台股": {
        "大盤/ETF": "^TWII, 0050.TW, 0056.TW",
        "權值股": "2330.TW, 2317.TW, 2454.TW, 2308.TW",
        "原油ETF": "00715L.TW, 00642U.TW, 00673R.TW",
    },
    "港股": {
        "大盤/ETF": "^HSI, 2800.HK, 2828.HK",
        "權值股": "0700.HK, 9988.HK, 1211.HK, 1810.HK",
    },
}

tab1, tab2, tab3 = st.tabs(["🚦 看一檔", "🩺 一次測很多檔", "📖 怎麼用"])


# ==================== 分頁一：看一檔 ====================
with tab1:
    c1, c2 = st.columns([3, 2])
    with c1:
        sym = st.text_input("股票代號", "2330.TW",
                            help="台股 2330.TW｜港股 0700.HK（四位數）｜美股 AAPL")
    with c2:
        tf_pick = st.radio("你想抱多久", ["幾週到幾個月（日線）", "幾個月以上（周線）"],
                           horizontal=False)

    # 防呆一：沒有 60 分鐘選項
    st.caption("💡 這裡沒有「60分鐘」選項。回測顯示短線進出幾乎每筆都在賠，所以拿掉了。")

    if st.button("看現在狀況", use_container_width=True, type="primary"):
        iv, per = ("1d", "5y") if "日線" in tf_pick else ("1wk", "10y")

        with st.spinner("查資料中..."):
            df = get_data(sym, iv, per)

        if df is None:
            st.error("查不到這檔。台股要加 .TW，港股要加 .HK（騰訊是 0700.HK 不是 700.HK）")
        else:
            d = add_ind(df)
            last, prev = d.iloc[-1], d.iloc[-2]
            price = float(last["Close"])
            ma_v = float(last["MA"])
            k_v, k_p = float(last["K"]), float(prev["K"])
            gap = float(last["MACD"]) - float(last["SIG"])

            trend, slope = trend_check(d)

            # ===== 防呆二：趨勢過濾 =====
            st.markdown("---")
            st.write("### 第一關：這檔股票整體在漲還是在跌？")

            if trend == "往下":
                st.error(f"## ⛔ 這檔在跌（近期平均價下滑 {abs(slope):.1f}%）")
                st.write("**建議不要用這套規則做這檔。**")
                st.write("你自己的回測已經證明過：股票本身在跌的時候，"
                         "照這套規則做會賠得比「買了放著不動」更慘。")
                st.write("")
                st.info("如果你還是想看它的訊號，可以往下捲。但請先想清楚為什麼。")
                show_detail = st.checkbox("我知道風險，還是要看", value=False)
            elif trend == "橫著走":
                st.warning(f"## ⚠️ 這檔在橫著走（平均價幾乎沒動，{slope:+.1f}%）")
                st.write("盤整的時候訊號會忽真忽假，做幾次就被手續費磨光。建議先觀望。")
                show_detail = True
            else:
                st.success(f"## ✅ 這檔在漲（近期平均價上升 {slope:.1f}%）")
                st.write("這是這套規則比較有機會的環境。")
                show_detail = True

            if show_detail:
                st.markdown("---")
                st.write("### 第二關：現在這個時間點可以嗎？")

                q1 = price > ma_v
                q2 = (k_v > 50) and (k_v > k_p)
                q3 = gap > 0
                hit = sum([q1, q2, q3])

                if hit == 3 and trend == "往上":
                    st.success("## 🟢 綠燈")
                    st.write("**三個都對，而且大方向是往上的。**這是規則說可以看的時候。")
                elif hit == 3:
                    st.warning("## 🟡 黃燈（三個都對，但大方向不好）")
                    st.write("時間點對，但這檔整體不是往上的。這種訊號成功率會低很多。")
                elif hit == 2:
                    st.warning("## 🟡 黃燈")
                    st.write("**對了兩個，還差一個。**你的規則要三個都對，差一個就是還沒到。")
                else:
                    st.error(f"## 🔴 紅燈（只對 {hit} 個）")
                    st.write("現在不是時候。")

                st.write("")
                pct = (price / ma_v - 1) * 100
                st.write(f"**① 現在的價錢，比最近的平均價高嗎？**")
                if q1:
                    st.write(f"　✅ 是。現在 {price:,.0f}，平均價 {ma_v:,.0f}，高出 {pct:.1f}%")
                    if pct < 2:
                        st.write("　⚠️ 只高一點點，很容易掉下去")
                else:
                    st.write(f"　❌ 不是。現在 {price:,.0f}，平均價 {ma_v:,.0f}，低了 {abs(pct):.1f}%")

                st.write("")
                st.write(f"**② 有沒有力氣，而且力氣在變大？**")
                dirn = "變大" if k_v > k_p else "變小"
                if q2:
                    st.write(f"　✅ 是。力氣 {k_v:.0f} 分（滿分100），上次 {k_p:.0f} 分，正在{dirn}")
                else:
                    st.write(f"　❌ 不是。力氣 {k_v:.0f} 分，上次 {k_p:.0f} 分，正在{dirn}")
                if k_v > 80:
                    st.write(f"　⚠️ {k_v:.0f} 分太高了，這時候買是追高")

                st.write("")
                st.write(f"**③ 漲的力道還在加強嗎？**")
                st.write(f"　{'✅ 是' if q3 else '❌ 不是'}。力道{'還在往上' if q3 else '在減弱'}"
                         f"（{gap:+.2f}）")

                # ===== 防呆三：自動算好停損和部位 =====
                if hit == 3 and trend == "往上":
                    st.markdown("---")
                    st.write("### 第三關：如果要買，這樣設定")

                    sl_price = price * 0.85
                    ma_price = ma_v

                    b1, b2, b3 = st.columns(3)
                    b1.metric("買進價（現在）", f"{price:,.0f}")
                    b2.metric("認賠出場價", f"{sl_price:,.0f}", "-15%")
                    b3.metric("平均價（也要看）", f"{ma_price:,.0f}")

                    st.write(f"**什麼時候該賣？** 以下任一個發生就走：")
                    st.write(f"　• 跌到 **{sl_price:,.0f}**（賠 15%）")
                    st.write(f"　• 跌破平均價 **{ma_price:,.0f}** 而且沒有很快站回去")
                    st.write("")
                    st.info("**為什麼是 15% 不是 8%？**　"
                            "你的回測顯示，賺錢的都是抱三個月以上那幾筆。"
                            "停損設太緊，股票稍微震一下就被洗出場，"
                            "永遠等不到那一波大的。")

                    st.write("")
                    st.error("**部位大小：這套系統最多只放你資金的 10-20%。**\n\n"
                             "理由：你的回測顯示曾經連續賠 14 次、資金縮水 48%。"
                             "如果全押，那 48% 你可能撐不過去。")

                # 圖
                st.markdown("---")
                fig = go.Figure(data=[go.Candlestick(
                    x=d.index, open=d["Open"], high=d["High"],
                    low=d["Low"], close=d["Close"], name="價格")])
                fig.add_trace(go.Scatter(x=d.index, y=d["MA"], name="平均價",
                                         line=dict(color="red", width=2)))
                fig.update_layout(height=380, xaxis_rangeslider_visible=False,
                                  margin=dict(t=20, b=20))
                st.plotly_chart(fig, use_container_width=True)
                st.caption("紅線是最近的平均價。價格在紅線上=偏強，在紅線下=偏弱。")


# ==================== 分頁二：批次 ====================
with tab2:
    st.write("### 一次測很多檔，直接給結論")

    mkt = st.selectbox("市場", list(MARKETS.keys()))
    grp = st.selectbox("類型", list(MARKETS[mkt].keys()))
    txt = st.text_area("股票代號（可以自己改，逗號分開）",
                       MARKETS[mkt][grp], height=70)

    c1, c2 = st.columns(2)
    with c1:
        tfs = st.multiselect("週期", ["日線", "周線"], default=["日線"])
    with c2:
        # 防呆四：預設 15，不給選太緊的
        stops = st.multiselect("認賠出場設多少%", [8, 12, 15, 20], default=[15])
        st.caption("💡 建議用 15。8% 太緊，回測顯示會把賺錢的大波段砍掉。")

    if st.button("開始測", use_container_width=True, type="primary", key="b2"):
        syms = [s.strip().upper() for s in txt.replace("\n", ",").split(",") if s.strip()][:10]

        if not syms or not tfs or not stops:
            st.error("三個欄位都要至少選一個")
        else:
            rows, failed = [], []
            jobs = len(syms) * len(tfs) * len(stops)
            bar = st.progress(0.0, text="準備中")
            done = 0

            for s in syms:
                for t in tfs:
                    iv, per = ("1d", "5y") if t == "日線" else ("1wk", "10y")
                    df = get_data(s, iv, per)
                    if df is None:
                        failed.append(s)
                        done += len(stops)
                        bar.progress(min(done / jobs, 1.0))
                        continue
                    d = add_ind(df)
                    trend, slope = trend_check(d)

                    ma = d["MA"].values
                    k = d["K"].values
                    mc = d["MACD"].values
                    sg = d["SIG"].values
                    cl = d["Close"].values

                    for sp in stops:
                        rets, holds = [], []
                        pos = False
                        ep = ei = 0
                        for i in range(63, len(d)):
                            if np.isnan(ma[i]) or np.isnan(k[i]) or np.isnan(k[i-1]):
                                continue
                            p = cl[i]
                            if not pos:
                                if p > ma[i] and k[i] > 50 and k[i] > k[i-1] and mc[i] > sg[i]:
                                    pos, ep, ei = True, p, i
                            else:
                                r = (p - ep) / ep * 100
                                if r <= -sp or p < ma[i]:
                                    rets.append(r - 0.6)
                                    holds.append(i - ei)
                                    pos = False
                        done += 1
                        bar.progress(min(done / jobs, 1.0), text=f"{s} {t} {sp}%")

                        if len(rets) < 3:
                            continue
                        rr = np.array(rets)
                        hh = np.array(holds)
                        w = rr[rr > 0]
                        wr = len(w) / len(rr) * 100
                        eq = 100.0
                        curve = [100.0]
                        for x in rr:
                            eq *= (1 + x / 100)
                            curve.append(eq)
                        bh = (cl[-1] - cl[63]) / cl[63] * 100
                        peak, mdd = curve[0], 0.0
                        for v in curve:
                            peak = max(peak, v)
                            mdd = min(mdd, (v - peak) / peak * 100)
                        st_, mx = 0, 0
                        for x in rr:
                            if x <= 0:
                                st_ += 1; mx = max(mx, st_)
                            else:
                                st_ = 0
                        lr = rr[hh >= 30]
                        sr = rr[hh < 30]

                        rows.append({
                            "股票": s, "週期": t, "認賠%": sp,
                            "這檔在": trend,
                            "做幾次": len(rr),
                            "幾次賺": len(w),
                            "每次平均%": round(rr.mean(), 2),
                            "照做總共%": round(eq - 100, 1),
                            "放著不動%": round(bh, 1),
                            "最慘縮水%": round(mdd, 1),
                            "連賠最多": mx,
                            "_long": lr.mean() if len(lr) else np.nan,
                            "_short": sr.mean() if len(sr) else np.nan,
                        })
            bar.empty()

            if failed:
                st.warning("查不到，已跳過：" + "、".join(set(failed)))

            if not rows:
                st.error("沒有跑出足夠交易。換股票試試。")
            else:
                res = pd.DataFrame(rows)
                st.markdown("---")
                st.write("## 結論")

                n = len(res)
                good = res[res["每次平均%"] > 0]
                beat = res[res["照做總共%"] > res["放著不動%"]]

                pg = len(good) / n * 100
                if pg >= 70:
                    st.success(f"**{len(good)} 種賺錢 / 共 {n} 種（{pg:.0f}%）**　整體站得住腳。")
                elif pg >= 40:
                    st.warning(f"**{len(good)} 種賺錢 / 共 {n} 種（{pg:.0f}%）**　一半一半，很不穩。")
                else:
                    st.error(f"**只有 {len(good)} 種賺錢 / 共 {n} 種（{pg:.0f}%）**　大多數情況是虧的。")

                if len(beat) == 0:
                    st.error("**沒有一種贏過「買了放著不動」。**　進進出出只是白忙。")
                else:
                    st.info(f"**{len(beat)} / {n} 種贏過「買了放著不動」。**")

                # 漲跌對照
                up = res[res["這檔在"] == "往上"]
                dn = res[res["這檔在"] == "往下"]
                if len(up) and len(dn):
                    st.write("")
                    st.write(f"**這檔在漲的時候** → 平均每次 {up['每次平均%'].mean():+.2f}%")
                    st.write(f"**這檔在跌的時候** → 平均每次 {dn['每次平均%'].mean():+.2f}%")
                    st.warning("→ 選對股票比什麼都重要。規則不會幫你選股。")

                # 長短
                lm, sm = res["_long"].dropna(), res["_short"].dropna()
                if len(lm) and len(sm):
                    st.write("")
                    st.write(f"**抱久（30根K以上）** → 平均 {lm.mean():+.2f}%　"
                             f"**抱短（不到30根）** → 平均 {sm.mean():+.2f}%")
                    if lm.mean() > sm.mean():
                        st.info("→ 抱久明顯比較好。這套規則是抓波段的，不適合短進短出。")

                st.write("")
                st.write(f"**心理準備：** 最慘曾連賠 **{int(res['連賠最多'].max())} 次**，"
                         f"資金縮水過 **{abs(res['最慘縮水%'].min()):.0f}%**。")

                st.markdown("---")
                st.write("## 明細")
                cols = ["股票", "週期", "認賠%", "這檔在", "做幾次", "幾次賺",
                        "每次平均%", "照做總共%", "放著不動%", "最慘縮水%", "連賠最多"]
                show = res.sort_values("每次平均%", ascending=False)[cols].reset_index(drop=True)

                def col(v):
                    try:
                        return "color: green" if v > 0 else "color: red"
                    except Exception:
                        return ""

                st.dataframe(show.style.map(col, subset=["每次平均%", "照做總共%", "放著不動%"]),
                             use_container_width=True)


# ==================== 分頁三：說明 ====================
with tab3:
    st.write("## 這個工具幫你擋掉的四件事")

    st.write("### 1️⃣ 不給你選 60 分鐘")
    st.write("你的回測跑出來，42 筆交易裡面，抱不到 10 根 K 棒的**沒有一筆是賺的**。"
             "短線進出只會讓你做更多賠錢的交易，所以這個選項直接拿掉。")

    st.write("### 2️⃣ 認賠出場預設 15%，不是 8%")
    st.write("你的回測顯示，賺大錢的都是抱三到五個月那幾筆（+74%、+47%、+42%）。"
             "停損設 8% 太緊，股票中間震一下就把你洗出場，那波大的你就吃不到了。")

    st.write("### 3️⃣ 這檔在跌的話，會先擋你")
    st.write("你測的三檔已經證明：股票本身在漲，規則賺；股票本身在跌，"
             "規則賠得比「買了放著不動」還慘（-48% vs -17%）。"
             "所以工具會先看這檔整體是漲是跌，在跌就先警告你。")

    st.write("### 4️⃣ 提醒你部位不要放太大")
    st.write("回測裡曾經連續賠 14 次、資金縮水 48%。"
             "如果你全押，這 48% 大概率會讓你在最低點放棄。所以建議這套系統只放 10-20%。")

    st.markdown("---")
    st.write("## 資金該怎麼分配（僅供參考）")
    st.write("""
    這不是建議，只是把你回測看到的數字整理一下：

    - 你測的每一檔，**照規則做都輸給「買了放著不動」**
    - 台積電：規則 174% vs 放著不動 297%
    - 鴻海：規則 72% vs 放著不動 141%

    所以比較合理的想法是：**大部分資金放大盤ETF長期擺著，
    這套系統只用小部位玩。**

    要不要這樣做，還是你自己決定。
    """)

    st.markdown("---")
    st.write("## 三個問題在講什麼")
    st.write("""
    **① 現在的價錢，比最近的平均價高嗎？**　
    專業說法叫「站上60日均線」。就是拿最近60天的收盤價平均起來，
    看現在的價格比它高還低。比平均高，代表最近買的人大多是賺的，
    比較不會急著賣。

    **② 有沒有力氣，而且力氣在變大？**　
    專業說法叫「KD的K值」。滿分100分，分數高代表最近漲得比較猛。
    但超過80分就太高了，代表已經漲一段，這時候進場是追高。

    **③ 漲的力道還在加強嗎？**　
    專業說法叫「MACD」。就算股票還在漲，如果一次比一次沒力，
    通常快要停了。
    """)

st.markdown("---")
st.caption("這是自己研究用的工具，不是投資建議。所有數字都來自公開資料回測，"
           "已扣手續費估算但未計滑價。過去會賺不代表以後也會。")
