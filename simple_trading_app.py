import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.graph_objects as go

st.set_page_config(page_title="666戰法系統", page_icon="📈", layout="wide")
st.title("📈 666戰法系統")

# ==================== 共用函式 ====================

@st.cache_data(ttl=300)
def get_data(symbol, interval, period):
    """下載資料並整理欄位格式"""
    df = yf.download(symbol, interval=interval, period=period,
                     progress=False, auto_adjust=False)
    if df is None or df.empty:
        return None
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df = df.dropna(subset=["Close"])
    return df if len(df) > 0 else None


def add_indicators(df, ma_len=60, kd_len=60):
    """計算 MA / KD / MACD"""
    d = df.copy()
    close, high, low = d["Close"], d["High"], d["Low"]

    d["MA"] = close.rolling(ma_len).mean()

    lowest = low.rolling(kd_len).min()
    highest = high.rolling(kd_len).max()
    rsv = 100 * (close - lowest) / (highest - lowest + 1e-10)
    d["K"] = rsv.rolling(3).mean()
    d["D"] = d["K"].rolling(3).mean()

    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    d["MACD"] = ema12 - ema26
    d["SIG"] = d["MACD"].ewm(span=9, adjust=False).mean()

    return d


TF_MAP = {
    "60分鐘": ("60m", ["1mo", "3mo", "6mo"]),
    "日線":   ("1d",  ["6mo", "1y", "2y", "5y"]),
    "周線":   ("1wk", ["2y", "5y", "10y"]),
}

tab1, tab2 = st.tabs(["📍 即時分析", "📊 歷史回測"])


# ---------- 分頁一：即時分析 ----------
with tab1:
    st.subheader("目前這檔符合幾條規則")

    c1, c2, c3 = st.columns(3)
    with c1:
        sym1 = st.text_input("股票代號", "2330.TW", key="s1")
    with c2:
        tf1 = st.selectbox("週期", list(TF_MAP.keys()), index=1, key="t1")
    with c3:
        ma_len1 = st.selectbox("均線長度", [20, 60], index=1, key="m1")

    if st.button("分析", use_container_width=True, key="b1"):
        interval, periods = TF_MAP[tf1]
        period = periods[-1]

        with st.spinner("下載資料中..."):
            df = get_data(sym1, interval, period)

        if df is None or len(df) < ma_len1 + 5:
            st.error("抓不到足夠資料。代號格式：台股 2330.TW、上櫃 6488.TWO、美股 AAPL")
        else:
            d = add_indicators(df, ma_len1, 60)
            last = d.iloc[-1]
            prev = d.iloc[-2]

            price = float(last["Close"])
            ma_v = float(last["MA"])
            k_v = float(last["K"])
            k_p = float(prev["K"])
            macd_v = float(last["MACD"])
            sig_v = float(last["SIG"])

            r1 = price > ma_v
            r2 = (k_v > 50) and (k_v > k_p)
            r3 = macd_v > sig_v
            hit = sum([r1, r2, r3])

            st.markdown("---")

            if hit == 3:
                st.success(f"### 三條規則全部符合（{hit}/3）")
            elif hit == 2:
                st.warning(f"### 符合 2 條，訊號不完整（{hit}/3）")
            elif hit == 1:
                st.info(f"### 只符合 1 條，不算訊號（{hit}/3）")
            else:
                st.error(f"### 一條都沒符合（{hit}/3）")

            m1, m2, m3, m4 = st.columns(4)
            m1.metric("現價", f"{price:,.2f}")
            m2.metric(f"{ma_len1}MA", f"{ma_v:,.2f}", f"{(price/ma_v-1)*100:+.2f}%")
            m3.metric("K值", f"{k_v:.1f}", f"{k_v-k_p:+.1f}")
            m4.metric("MACD-Signal", f"{macd_v-sig_v:+.3f}")

            st.markdown("---")
            st.write("**逐條檢查**")
            st.write(f"{'✅' if r1 else '❌'} 規則1　股價站上{ma_len1}MA　（{price:,.2f} vs {ma_v:,.2f}）")
            st.write(f"{'✅' if r2 else '❌'} 規則2　K值>50且上升　（K={k_v:.1f}，前一根{k_p:.1f}）")
            st.write(f"{'✅' if r3 else '❌'} 規則3　MACD>Signal　（差{macd_v-sig_v:+.3f}）")

            st.markdown("---")

            fig = go.Figure(data=[go.Candlestick(
                x=d.index, open=d["Open"], high=d["High"],
                low=d["Low"], close=d["Close"], name="K線")])
            fig.add_trace(go.Scatter(x=d.index, y=d["MA"], name=f"{ma_len1}MA",
                                     line=dict(color="red", width=2)))
            fig.update_layout(height=400, xaxis_rangeslider_visible=False,
                              margin=dict(t=30, b=20))
            st.plotly_chart(fig, use_container_width=True)

            fig2 = go.Figure()
            fig2.add_trace(go.Scatter(x=d.index, y=d["K"], name="K", line=dict(color="blue")))
            fig2.add_trace(go.Scatter(x=d.index, y=d["D"], name="D", line=dict(color="orange")))
            fig2.add_hline(y=50, line_dash="dash", line_color="gray")
            fig2.update_layout(height=250, margin=dict(t=30, b=20), title="KD")
            st.plotly_chart(fig2, use_container_width=True)

            st.caption("三條規則全中才是完整訊號。這裡只顯示規則符合狀況，不是買賣建議。")


# ---------- 分頁二：歷史回測 ----------
with tab2:
    st.subheader("這套規則過去真的賺錢嗎")

    c1, c2, c3 = st.columns(3)
    with c1:
        sym2 = st.text_input("股票代號", "2330.TW", key="s2")
    with c2:
        tf2 = st.selectbox("週期", list(TF_MAP.keys()), index=1, key="t2")
    with c3:
        per2 = st.selectbox("回測長度", TF_MAP[tf2][1],
                            index=len(TF_MAP[tf2][1]) - 1, key="p2")

    c4, c5, c6 = st.columns(3)
    with c4:
        ma_len2 = st.selectbox("均線長度", [20, 60], index=1, key="m2")
    with c5:
        stop_pct = st.slider("停損 %", 2, 20, 8, key="sl")
    with c6:
        use_macd = st.checkbox("要求MACD也符合", value=True, key="mc")

    fee = st.slider("單趟成本 %（台股來回約0.6）", 0.0, 1.0, 0.3, 0.05)

    if st.button("開始回測", use_container_width=True, key="b2"):
        interval = TF_MAP[tf2][0]

        with st.spinner("回測中..."):
            df = get_data(sym2, interval, per2)

        if df is None or len(df) < ma_len2 + 30:
            st.error("資料不足，拉長回測長度或換代號")
        else:
            d = add_indicators(df, ma_len2, 60)

            trades = []
            pos = False
            entry_p = 0.0
            entry_i = 0
            start = max(ma_len2, 60) + 3

            for i in range(start, len(d)):
                row = d.iloc[i]
                prv = d.iloc[i - 1]
                if pd.isna(row["MA"]) or pd.isna(row["K"]) or pd.isna(prv["K"]):
                    continue

                p = float(row["Close"])

                if not pos:
                    c_ma = p > float(row["MA"])
                    c_kd = (float(row["K"]) > 50) and (float(row["K"]) > float(prv["K"]))
                    c_md = float(row["MACD"]) > float(row["SIG"]) if use_macd else True
                    if c_ma and c_kd and c_md:
                        pos, entry_p, entry_i = True, p, i
                else:
                    ret = (p - entry_p) / entry_p * 100
                    why = None
                    if ret <= -stop_pct:
                        why = "停損"
                    elif p < float(row["MA"]):
                        why = "跌破MA"
                    if why:
                        trades.append({
                            "進場": d.index[entry_i].strftime("%Y-%m-%d"),
                            "出場": d.index[i].strftime("%Y-%m-%d"),
                            "進場價": round(entry_p, 2),
                            "出場價": round(p, 2),
                            "淨報酬%": round(ret - fee * 2, 2),
                            "原因": why,
                            "持有K棒": i - entry_i,
                        })
                        pos = False

            tr = pd.DataFrame(trades)

            if tr.empty or len(tr) < 3:
                st.warning(f"交易次數太少（{len(tr)}筆），樣本不足以判斷。拉長回測期間試試。")
                if not tr.empty:
                    st.dataframe(tr, use_container_width=True)
            else:
                wins = tr[tr["淨報酬%"] > 0]
                loss = tr[tr["淨報酬%"] <= 0]
                n = len(tr)
                wr = len(wins) / n * 100
                aw = wins["淨報酬%"].mean() if len(wins) else 0.0
                al = loss["淨報酬%"].mean() if len(loss) else 0.0
                exp = wr / 100 * aw + (1 - wr / 100) * al

                cum = 1.0
                eq = [100.0]
                for r in tr["淨報酬%"]:
                    cum *= (1 + r / 100)
                    eq.append(eq[-1] * (1 + r / 100))
                total_ret = (cum - 1) * 100

                bh_start = float(d["Close"].iloc[start])
                bh_end = float(d["Close"].iloc[-1])
                bh = (bh_end - bh_start) / bh_start * 100

                peak = eq[0]
                mdd = 0.0
                for v in eq:
                    peak = max(peak, v)
                    mdd = min(mdd, (v - peak) / peak * 100)

                streak = mx = 0
                for r in tr["淨報酬%"]:
                    if r <= 0:
                        streak += 1
                        mx = max(mx, streak)
                    else:
                        streak = 0

                st.markdown("---")
                a1, a2, a3, a4 = st.columns(4)
                a1.metric("真實勝率", f"{wr:.1f}%")
                a2.metric("交易次數", f"{n} 筆")
                a3.metric("每筆期望值", f"{exp:+.2f}%")
                a4.metric("最大回撤", f"{mdd:.1f}%")

                b1, b2, b3, b4 = st.columns(4)
                b1.metric("平均賺", f"{aw:+.2f}%")
                b2.metric("平均賠", f"{al:+.2f}%")
                b3.metric("策略累積", f"{total_ret:+.1f}%")
                b4.metric("買進持有", f"{bh:+.1f}%")

                st.markdown("---")
                st.write("**白話結論**")

                if exp > 0:
                    st.success(f"每做一筆平均賺 {exp:.2f}%（已扣成本），長期是正的。")
                else:
                    st.error(f"每做一筆平均賠 {abs(exp):.2f}%（已扣成本），長期照做會虧錢。")

                if total_ret > bh:
                    st.info(f"策略 {total_ret:+.1f}% 打敗 買進持有 {bh:+.1f}%。")
                else:
                    st.warning(f"策略 {total_ret:+.1f}% 輸給 買進持有 {bh:+.1f}%，不如買了放著。")

                st.write(f"心理準備：曾連續賠 {mx} 次，資金最深縮水 {abs(mdd):.1f}%。")

                st.markdown("---")
                st.write("**資金曲線（起始100）**")
                f1 = go.Figure()
                f1.add_trace(go.Scatter(y=eq, line=dict(color="steelblue", width=2)))
                f1.add_hline(y=100, line_dash="dash", line_color="gray")
                f1.update_layout(height=300, margin=dict(t=20, b=20), xaxis_title="交易次數")
                st.plotly_chart(f1, use_container_width=True)

                st.write("**每筆報酬**")
                f2 = go.Figure()
                f2.add_trace(go.Bar(
                    y=tr["淨報酬%"],
                    marker_color=["green" if r > 0 else "red" for r in tr["淨報酬%"]]))
                f2.add_hline(y=0, line_color="black")
                f2.update_layout(height=250, margin=dict(t=20, b=20))
                st.plotly_chart(f2, use_container_width=True)

                st.write("**交易明細**")
                st.dataframe(tr, use_container_width=True)

    st.caption("已扣手續費估算，未計滑價。交易少於30筆的結果參考價值有限。")

st.markdown("---")
st.caption("本工具僅供技術分析研究，不構成投資建議。")
