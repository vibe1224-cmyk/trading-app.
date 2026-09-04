import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.graph_objects as go

st.set_page_config(page_title="股票紅綠燈", page_icon="🚦", layout="wide")
st.title("🚦 股票紅綠燈")
st.caption("五道關卡，一關一關幫你檢查")

# ==================== 工具 ====================

def fix_symbol(s):
    """自動修正常見的代號打錯"""
    s = s.strip().upper().replace(" ", "")
    if s.endswith(".HK"):
        num = s[:-3].lstrip("0")
        if num.isdigit():
            s = num.zfill(4) + ".HK"
    return s


def which_market(sym):
    """判斷這檔屬於哪個市場，回傳(市場名, 大盤代號, 幣別)"""
    s = sym.upper()
    if s.endswith(".TW") or s.endswith(".TWO") or s == "^TWII":
        return "台股", "^TWII", "台幣"
    if s.endswith(".HK") or s == "^HSI":
        return "港股", "^HSI", "港幣"
    return "美股", "^GSPC", "美元"


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


@st.cache_data(ttl=1800, show_spinner=False)
def get_info(symbol):
    """抓公司基本資料，失敗就回傳空的"""
    try:
        t = yf.Ticker(symbol)
        info = t.info
        return info if isinstance(info, dict) and len(info) > 3 else {}
    except Exception:
        return {}


def add_ind(df):
    d = df.copy()
    c, h, l = d["Close"], d["High"], d["Low"]
    d["MA"] = c.rolling(60).mean()
    lo, hi = l.rolling(60).min(), h.rolling(60).max()
    d["K"] = (100 * (c - lo) / (hi - lo + 1e-10)).rolling(3).mean()
    d["D"] = d["K"].rolling(3).mean()
    e12 = c.ewm(span=12, adjust=False).mean()
    e26 = c.ewm(span=26, adjust=False).mean()
    d["MACD"] = e12 - e26
    d["SIG"] = d["MACD"].ewm(span=9, adjust=False).mean()
    return d


def kd_cross(d, lookback=10):
    """判斷最近有沒有黃金交叉或死亡交叉，回傳(狀態, 幾根K棒前)"""
    k = d["K"].values
    dd = d["D"].values
    n = len(k)
    for back in range(1, min(lookback, n - 1) + 1):
        i = n - back
        if np.isnan(k[i]) or np.isnan(dd[i]) or np.isnan(k[i-1]) or np.isnan(dd[i-1]):
            continue
        if k[i-1] <= dd[i-1] and k[i] > dd[i]:
            return "黃金交叉", back - 1
        if k[i-1] >= dd[i-1] and k[i] < dd[i]:
            return "死亡交叉", back - 1
    return "沒有交叉", None


def trend_check(d):
    ma = d["MA"].values
    valid = ma[~np.isnan(ma)]
    if len(valid) < 40:
        return "資料不足", 0.0
    now, before = valid[-1], valid[-40]
    slope = (now - before) / before * 100
    if slope > 3:
        return "往上", slope
    if slope < -3:
        return "往下", slope
    return "橫著走", slope


def money_str(v, cur):
    """把金額變成好讀的字"""
    if v is None or np.isnan(v):
        return "查不到"
    if cur == "美元":
        if v >= 1e9:
            return f"{v/1e9:.1f} 十億{cur}"
        if v >= 1e6:
            return f"{v/1e6:.0f} 百萬{cur}"
        return f"{v:,.0f} {cur}"
    if v >= 1e8:
        return f"{v/1e8:.1f} 億{cur}"
    if v >= 1e4:
        return f"{v/1e4:.0f} 萬{cur}"
    return f"{v:,.0f} {cur}"


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


# ==================== 分頁一 ====================
with tab1:
    c1, c2 = st.columns([3, 2])
    with c1:
        sym_in = st.text_input("股票代號", "2330.TW",
                               help="台股 2330.TW｜港股湊四位數 0700.HK｜美股 AAPL")
    with c2:
        tf_pick = st.radio("你想抱多久",
                           ["幾週到幾個月（日線）", "幾個月以上（周線）"])

    st.caption("💡 沒有「60分鐘」選項。回測顯示短線進出幾乎每筆都在賠，所以拿掉了。")

    if st.button("開始檢查", use_container_width=True, type="primary"):
        sym = fix_symbol(sym_in)
        if sym != sym_in.strip().upper():
            st.info(f"代號已自動修正：{sym_in} → **{sym}**")

        mkt, idx_sym, cur = which_market(sym)
        iv, per = ("1d", "5y") if "日線" in tf_pick else ("1wk", "10y")

        with st.spinner("查資料中，約需10-20秒..."):
            df = get_data(sym, iv, per)
            df_day = get_data(sym, "1d", "1y")      # 成交量固定用日線
            df_idx = get_data(idx_sym, "1d", "2y")  # 大盤固定用日線
            info = get_info(sym)

        if df is None:
            st.error(f"查不到 **{sym}**")
            st.write("**代號格式：** 台股 2330.TW｜港股湊四位數 0700.HK｜"
                     "美股 AAPL｜大盤 ^TWII、^HSI")
            st.stop()

        d = add_ind(df)
        pass_count = 0

        # ========== 第一關：大盤環境 ==========
        st.markdown("---")
        st.write("## 第一關　大盤現在是什麼環境？")

        market_ok = None
        if df_idx is None:
            st.warning("⚠️ 查不到大盤資料，這關跳過。")
            market_ok = True
        else:
            di = add_ind(df_idx)
            i_price = float(di["Close"].iloc[-1])
            i_ma = float(di["MA"].iloc[-1])
            i_trend, i_slope = trend_check(di)

            above = i_price > i_ma
            if above and i_trend == "往上":
                st.success(f"### ✅ {mkt}大盤在漲")
                st.write(f"大盤 {i_price:,.0f}，在平均價 {i_ma:,.0f} 之上，"
                         f"近期上升 {i_slope:.1f}%")
                st.write("**大環境是順的。個股表現有大盤幫忙。**")
                market_ok = True
                pass_count += 1
            elif not above:
                st.error(f"### ⛔ {mkt}大盤在跌")
                st.write(f"大盤 {i_price:,.0f}，掉到平均價 {i_ma:,.0f} 之下")
                st.write("**大盤在跌的時候，八成的個股都會跟著跌。**"
                         "這時候買個股，等於是逆著水流游泳。")
                market_ok = False
            else:
                st.warning(f"### ⚠️ {mkt}大盤在整理")
                st.write(f"大盤 {i_price:,.0f}，剛好在平均價 {i_ma:,.0f} 附近，方向不明")
                st.write("**這種時候訊號容易忽真忽假。**")
                market_ok = True

        if market_ok is False:
            st.write("")
            if not st.checkbox("我知道大盤在跌，還是要繼續看", value=False):
                st.stop()

        # ========== 第二關：這檔的趨勢 ==========
        st.markdown("---")
        st.write("## 第二關　這檔股票自己在漲還是在跌？")

        trend, slope = trend_check(d)
        if trend == "往上":
            st.success(f"### ✅ 在漲（近期平均價上升 {slope:.1f}%）")
            st.write("這是這套規則比較有機會的環境。")
            pass_count += 1
        elif trend == "往下":
            st.error(f"### ⛔ 在跌（近期平均價下滑 {abs(slope):.1f}%）")
            st.write("**你的回測已經證明：**股票本身在跌的時候，"
                     "照這套規則做會賠得比「買了放著不動」更慘。")
            st.write("")
            if not st.checkbox("我知道這檔在跌，還是要繼續看", value=False):
                st.stop()
        else:
            st.warning(f"### ⚠️ 橫著走（{slope:+.1f}%，幾乎沒動）")
            st.write("盤整的時候做幾次就被手續費磨光。建議先觀望。")

        # ========== 第三關：公司體質 ==========
        st.markdown("---")
        st.write("## 第三關　這間公司本身還可以嗎？")

        name = info.get("longName") or info.get("shortName") or sym
        pe = info.get("trailingPE")
        eps = info.get("trailingEps")
        margin = info.get("profitMargins")
        roe = info.get("returnOnEquity")
        d2e = info.get("debtToEquity")
        rev_g = info.get("revenueGrowth")
        cap = info.get("marketCap")
        qtype = (info.get("quoteType") or "").upper()

        st.write(f"**{name}**")

        if qtype in ("ETF", "MUTUALFUND") or sym.startswith("^"):
            st.info("這是指數或 ETF，不是單一公司，沒有財報可以看。跳過這關。")
            pass_count += 1
        elif not info or (pe is None and eps is None and margin is None):
            st.warning("⚠️ Yahoo 查不到這檔的財務資料，這關沒辦法檢查。\n\n"
                       "**建議你自己去看一下：**這間公司有沒有賺錢？"
                       "負債會不會太重？營收在成長還是衰退？")
        else:
            issues = []
            goods = []

            # 賺不賺錢
            if eps is not None:
                if eps > 0:
                    goods.append(f"**有賺錢**（每股賺 {eps:.2f} {cur}）")
                else:
                    issues.append(f"**在虧錢**（每股虧 {abs(eps):.2f} {cur}）")

            if margin is not None:
                m = margin * 100
                if m > 10:
                    goods.append(f"**賺得不錯**（每 100 元營收賺 {m:.0f} 元）")
                elif m > 0:
                    goods.append(f"賺得普通（每 100 元營收賺 {m:.1f} 元）")
                else:
                    issues.append(f"**做一單賠一單**（利潤率 {m:.1f}%）")

            # 貴不貴
            if pe is not None and pe > 0:
                if pe < 15:
                    goods.append(f"**價格不算貴**（本益比 {pe:.0f} 倍）")
                elif pe < 30:
                    goods.append(f"價格合理偏高（本益比 {pe:.0f} 倍）")
                else:
                    issues.append(f"**價格偏貴**（本益比 {pe:.0f} 倍，"
                                  f"代表要 {pe:.0f} 年的獲利才回本）")

            # 負債
            if d2e is not None:
                if d2e > 200:
                    issues.append(f"**負債偏重**（負債是股東權益的 {d2e/100:.1f} 倍）")
                elif d2e < 100:
                    goods.append(f"**負債健康**（負債是股東權益的 {d2e/100:.1f} 倍）")

            # 成長
            if rev_g is not None:
                g = rev_g * 100
                if g > 10:
                    goods.append(f"**營收在成長**（比去年多 {g:.0f}%）")
                elif g < -10:
                    issues.append(f"**營收在衰退**（比去年少 {abs(g):.0f}%）")

            # ROE
            if roe is not None:
                r = roe * 100
                if r > 15:
                    goods.append(f"**股東的錢用得有效率**（ROE {r:.0f}%）")
                elif r < 0:
                    issues.append(f"**在虧股東的錢**（ROE {r:.0f}%）")

            if cap:
                st.caption(f"公司總市值：{money_str(cap, cur)}")

            if goods:
                st.write("**還不錯的地方：**")
                for g in goods:
                    st.write(f"　✅ {g}")
            if issues:
                st.write("**要注意的地方：**")
                for x in issues:
                    st.write(f"　⚠️ {x}")

            if not issues:
                st.success("### ✅ 體質沒看到明顯問題")
                pass_count += 1
            elif len(issues) >= 3:
                st.error(f"### ⛔ 有 {len(issues)} 個地方要注意")
                st.write("問題有點多。線圖再好看，公司本身有狀況還是有風險。")
            else:
                st.warning(f"### ⚠️ 有 {len(issues)} 個地方要注意")

            st.caption("財務數字來自 Yahoo，可能有落後或不準。重要決定請自己再查一次。")

        # ========== 第四關：成交量 ==========
        st.markdown("---")
        st.write("## 第四關　買得進、賣得出嗎？")

        if df_day is None or "Volume" not in df_day.columns:
            st.warning("⚠️ 查不到成交量資料，這關跳過。")
        else:
            vol = df_day["Volume"].tail(20)
            cls = df_day["Close"].tail(20)
            turnover = float((vol * cls).mean())
            avg_vol = float(vol.mean())
            recent = float(df_day["Volume"].tail(5).mean())
            base = float(df_day["Volume"].tail(60).mean())

            st.write(f"最近 20 天，平均每天成交 **{money_str(turnover, cur)}**")

            if cur == "美元":
                hot, ok_lv = 5e7, 5e6
            else:
                hot, ok_lv = 1e8, 2e7

            if turnover >= hot:
                st.success("### ✅ 很好賣")
                st.write("成交熱絡，你想買想賣隨時都有人接。")
                pass_count += 1
            elif turnover >= ok_lv:
                st.warning("### ⚠️ 普通")
                st.write("量還可以，但如果你部位比較大，賣的時候可能要分批。")
            else:
                st.error("### ⛔ 太冷門")
                st.write("**這種量很危險。** 買的時候可能買不到好價，"
                         "真的想賣的時候可能砍到跌停才有人接。")
                st.write("回測看不出這個問題，但實際交易會要命。")

            if base > 0:
                ratio = recent / base
                st.write("")
                if ratio > 2:
                    st.info(f"📢 最近 5 天的量是平常的 **{ratio:.1f} 倍**，"
                            f"有人在大量進出，可能有事情在發生。")
                elif ratio < 0.5:
                    st.info(f"😴 最近 5 天的量只有平常的 **{ratio:.1f} 倍**，"
                            f"沒什麼人在交易，行情容易走不動。")

        # ========== 第五關：時間點 ==========
        st.markdown("---")
        st.write("## 第五關　現在這個時間點可以嗎？")

        last, prev = d.iloc[-1], d.iloc[-2]
        price = float(last["Close"])
        ma_v = float(last["MA"])
        k_v, k_p = float(last["K"]), float(prev["K"])
        d_v = float(last["D"])
        cross, cross_ago = kd_cross(d)
        gap = float(last["MACD"]) - float(last["SIG"])

        q1 = price > ma_v
        q2 = (k_v > 50) and (k_v > k_p)
        q3 = gap > 0
        hit = sum([q1, q2, q3])
        if hit == 3:
            pass_count += 1

        if hit == 3:
            st.success("### 🟢 三個都對")
        elif hit == 2:
            st.warning("### 🟡 對了兩個，還差一個")
        else:
            st.error(f"### 🔴 只對了 {hit} 個")

        pct = (price / ma_v - 1) * 100
        st.write("")
        st.write("**① 現在的價錢，比最近的平均價高嗎？**")
        if q1:
            st.write(f"　✅ 是。現在 {price:,.2f}，平均價 {ma_v:,.2f}，高出 {pct:.1f}%")
            if pct < 2:
                st.write("　⚠️ 只高一點點，很容易掉下去")
        else:
            st.write(f"　❌ 不是。現在 {price:,.2f}，平均價 {ma_v:,.2f}，低了 {abs(pct):.1f}%")

        st.write("")
        st.write("**② KD 指標怎麼說？**")
        dirn = "上升" if k_v > k_p else "下降"
        st.write(f"　K值 **{k_v:.1f}**（上一根 {k_p:.1f}，{dirn}中）　"
                 f"D值 **{d_v:.1f}**")

        if cross == "黃金交叉":
            when = "這一根剛發生" if cross_ago == 0 else f"{cross_ago} 根K棒前發生"
            st.write(f"　🟡 **黃金交叉**（K往上穿過D，{when}）")
            if k_v < 30:
                st.write("　　→ 在低檔黃金交叉，這是比較好的位置")
            elif k_v > 80:
                st.write("　　→ 但已經在 80 以上的高檔，追高風險大")
        elif cross == "死亡交叉":
            when = "這一根剛發生" if cross_ago == 0 else f"{cross_ago} 根K棒前發生"
            st.write(f"　⚫ **死亡交叉**（K往下穿過D，{when}）")
            if k_v > 70:
                st.write("　　→ 在高檔死亡交叉，通常是要回檔了")
        else:
            pos_txt = "K在D上面（偏強）" if k_v > d_v else "K在D下面（偏弱）"
            st.write(f"　▫️ 最近 10 根沒有交叉，目前 {pos_txt}")

        if k_v > 80:
            st.write(f"　⚠️ K值 {k_v:.0f} 已在超買區（80以上），這時候買是追高")
        elif k_v < 20:
            st.write(f"　⚠️ K值 {k_v:.0f} 在超賣區（20以下），跌深但不代表會馬上反彈")

        st.write("")
        st.write(f"　**你的規則要求：K值>50 且上升** → "
                 f"{'✅ 符合' if q2 else '❌ 不符合'}")
        st.caption("　　註：你的666戰法用的是「K值站上50」，不是黃金交叉。"
                   "交叉資訊只是多給你參考，沒有納入判斷。")

        st.write("")
        st.write("**③ 漲的力道還在加強嗎？**")
        st.write(f"　{'✅ 是' if q3 else '❌ 不是'}。"
                 f"力道{'還在往上' if q3 else '在減弱'}（{gap:+.2f}）")

        # ========== 總結 ==========
        st.markdown("---")
        st.write("## 📋 五關總結")

        st.write(f"### 通過 {pass_count} / 5 關")

        if pass_count == 5:
            st.success("**五關全過。**這是這套規則能給你的最好狀況。")
        elif pass_count >= 4:
            st.warning("**過了大部分，但有一關沒過。**"
                       "沒過的那關就是你要承擔的風險。")
        elif pass_count >= 2:
            st.warning("**只過了一半。**條件不夠好，建議再等等。")
        else:
            st.error("**大部分都沒過。**現在不是時候。")

        if pass_count == 5:
            st.markdown("---")
            st.write("### 如果要買，這樣設定")
            sl = price * 0.85
            a, b, c = st.columns(3)
            a.metric("買進價（現在）", f"{price:,.2f}")
            b.metric("認賠出場價", f"{sl:,.2f}", "-15%")
            c.metric("平均價（跌破也要走）", f"{ma_v:,.2f}")

            st.info("**為什麼認賠設 15% 不是 8%？**　"
                    "你的回測顯示賺錢的都是抱三個月以上那幾筆。"
                    "設太緊，股票震一下就把你洗出場，那波大的就吃不到了。")

            st.error("**部位大小：這套系統最多放你資金的 10-20%。**\n\n"
                     "回測裡曾經連續賠 14 次、資金縮水 48%。全押的話你很難撐過去。")

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


# ==================== 分頁二 ====================
with tab2:
    st.write("### 一次測很多檔，直接給結論")

    mk = st.selectbox("市場", list(MARKETS.keys()))
    gp = st.selectbox("類型", list(MARKETS[mk].keys()))
    txt = st.text_area("股票代號（可自己改，逗號分開）", MARKETS[mk][gp], height=70)

    c1, c2 = st.columns(2)
    with c1:
        tfs = st.multiselect("週期", ["日線", "周線"], default=["日線"])
    with c2:
        stops = st.multiselect("認賠出場設多少%", [8, 12, 15, 20], default=[15])
        st.caption("💡 建議 15。8% 太緊會把賺錢的大波段砍掉。")

    if st.button("開始測", use_container_width=True, type="primary", key="b2"):
        syms = [fix_symbol(s) for s in txt.replace("\n", ",").split(",") if s.strip()][:10]

        if not syms or not tfs or not stops:
            st.error("三個欄位都要至少選一個")
        else:
            rows, failed = [], []
            jobs = len(syms) * len(tfs) * len(stops)
            bar = st.progress(0.0)
            done = 0

            for s in syms:
                for t in tfs:
                    iv, per = ("1d", "5y") if t == "日線" else ("1wk", "10y")
                    dfx = get_data(s, iv, per)
                    if dfx is None:
                        failed.append(s)
                        done += len(stops)
                        bar.progress(min(done / jobs, 1.0))
                        continue
                    dd = add_ind(dfx)
                    tr_name, _ = trend_check(dd)

                    ma = dd["MA"].values; k = dd["K"].values
                    mc = dd["MACD"].values; sg = dd["SIG"].values
                    cl = dd["Close"].values

                    for sp in stops:
                        rets, holds = [], []
                        pos = False; ep = ei = 0
                        for i in range(63, len(dd)):
                            if np.isnan(ma[i]) or np.isnan(k[i]) or np.isnan(k[i-1]):
                                continue
                            p = cl[i]
                            if not pos:
                                if p > ma[i] and k[i] > 50 and k[i] > k[i-1] and mc[i] > sg[i]:
                                    pos, ep, ei = True, p, i
                            else:
                                r = (p - ep) / ep * 100
                                if r <= -sp or p < ma[i]:
                                    rets.append(r - 0.6); holds.append(i - ei); pos = False
                        done += 1
                        bar.progress(min(done / jobs, 1.0), text=f"{s} {t} {sp}%")

                        if len(rets) < 3:
                            continue
                        rr = np.array(rets); hh = np.array(holds)
                        w = rr[rr > 0]
                        eq = 100.0; curve = [100.0]
                        for x in rr:
                            eq *= (1 + x / 100); curve.append(eq)
                        bh = (cl[-1] - cl[63]) / cl[63] * 100
                        peak = curve[0]; mdd = 0.0
                        for v in curve:
                            peak = max(peak, v); mdd = min(mdd, (v - peak) / peak * 100)
                        stk = mx = 0
                        for x in rr:
                            if x <= 0:
                                stk += 1; mx = max(mx, stk)
                            else:
                                stk = 0
                        lr = rr[hh >= 30]; sr = rr[hh < 30]

                        rows.append({
                            "股票": s, "週期": t, "認賠%": sp, "這檔在": tr_name,
                            "做幾次": len(rr), "幾次賺": len(w),
                            "每次平均%": round(rr.mean(), 2),
                            "照做總共%": round(eq - 100, 1),
                            "放著不動%": round(bh, 1),
                            "最慘縮水%": round(mdd, 1), "連賠最多": mx,
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
                    st.warning(f"**{len(good)} 種賺錢 / 共 {n} 種（{pg:.0f}%）**　一半一半，不穩。")
                else:
                    st.error(f"**只有 {len(good)} 種賺錢 / 共 {n} 種（{pg:.0f}%）**　多數是虧的。")

                if len(beat) == 0:
                    st.error("**沒有一種贏過「買了放著不動」。**")
                else:
                    st.info(f"**{len(beat)} / {n} 種贏過「買了放著不動」。**")

                up = res[res["這檔在"] == "往上"]; dn = res[res["這檔在"] == "往下"]
                if len(up) and len(dn):
                    st.write("")
                    st.write(f"**這檔在漲的時候** → 平均每次 {up['每次平均%'].mean():+.2f}%")
                    st.write(f"**這檔在跌的時候** → 平均每次 {dn['每次平均%'].mean():+.2f}%")
                    st.warning("→ 選對股票比什麼都重要。規則不會幫你選股。")

                lm, sm = res["_long"].dropna(), res["_short"].dropna()
                if len(lm) and len(sm):
                    st.write("")
                    st.write(f"**抱久（30根K以上）** → {lm.mean():+.2f}%　"
                             f"**抱短（不到30根）** → {sm.mean():+.2f}%")

                st.write("")
                st.write(f"**心理準備：** 最慘曾連賠 **{int(res['連賠最多'].max())} 次**，"
                         f"資金縮水過 **{abs(res['最慘縮水%'].min()):.0f}%**。")

                st.markdown("---")
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


# ==================== 分頁三 ====================
with tab3:
    st.write("## 五道關卡在檢查什麼")

    st.write("### 第一關　大盤環境")
    st.write("大盤在跌的時候，八成的個股都會跟著跌。"
             "個股再好也對抗不了大環境。所以先看大盤，"
             "大盤在平均價之下就先警告你。")

    st.write("### 第二關　這檔自己的方向")
    st.write("你的回測已經證明：股票在漲，規則賺；股票在跌，"
             "規則賠得比「買了放著不動」還慘（-48% vs -17%）。"
             "所以會先看這檔整體是漲是跌。")

    st.write("### 第三關　公司體質")
    st.write("線圖不會告訴你這間公司有沒有賺錢。"
             "一間虧損、負債重的公司，線圖再漂亮也可能突然出事。"
             "這關看：有沒有賺錢、貴不貴、負債重不重、營收有沒有成長。")

    st.write("### 第四關　成交量")
    st.write("**這關最容易被忽略，但實際交易最要命。**"
             "冷門股買得進、賣不出。回測看不出這個問題，"
             "因為回測假設你想賣就賣得掉，實際上不是這樣。")

    st.write("### 第五關　現在的時間點")
    st.write("這才是原本的三個技術問題。"
             "前面四關是在問「該不該碰這檔」，"
             "第五關才是問「現在是不是好時機」。")

    st.markdown("---")
    st.write("## 為什麼不多加幾個技術指標？")
    st.write("""
    加 RSI、布林通道、黃金交叉之類的，回測數字**一定會變好看**。

    但那是假的。指標加越多，越容易「剛好符合過去五年」，
    實際上路反而更差。這個坑很多人踩。

    真正缺的不是指標，是**大盤、公司、成交量**這些線圖以外的東西。
    所以這次加的是這三塊。
    """)

    st.markdown("---")
    st.write("## 三個技術問題在講什麼")
    st.write("""
    **① 現在的價錢，比最近的平均價高嗎？**　
    專業叫「站上60日均線」。拿最近60天收盤價平均，
    看現在比它高還低。比平均高，代表最近買的人大多是賺的。

    **② KD 指標**　
    KD 有兩條線：K線（快）和 D線（慢），都是 0~100 分。

    - **K往上穿過D = 黃金交叉**，一般視為轉強
    - **K往下穿過D = 死亡交叉**，一般視為轉弱
    - **K > 80 = 超買區**，已經漲一段，這時候買是追高
    - **K < 20 = 超賣區**，跌深了，但不代表馬上會反彈

    注意：你的666戰法用的是「K值站上50且上升」，**不是黃金交叉**。
    畫面上的交叉資訊只是給你參考，沒有算進判斷裡。

    **③ 漲的力道還在加強嗎？**　
    專業叫「MACD」。就算還在漲，如果一次比一次沒力，通常快停了。
    """)

st.markdown("---")
st.caption("研究工具，不是投資建議。財務數字來自 Yahoo，可能落後或不準。回測已扣手續費估算但未計滑價。過去會賺不代表以後也會。")
