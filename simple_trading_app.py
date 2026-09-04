import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.graph_objects as go
from plotly.subplots import make_subplots

st.set_page_config(page_title="趨勢結構", page_icon="📐", layout="wide")
st.title("📐 趨勢結構判斷")
st.caption("頭頭高底底高 / 頭頭低底底低 — 道氏理論的趨勢判斷")


# ==================== 基礎 ====================

def fix_symbol(s):
    s = s.strip().upper().replace(" ", "")
    if s.endswith(".HK"):
        n = s[:-3].lstrip("0")
        if n.isdigit():
            s = n.zfill(4) + ".HK"
    return s


@st.cache_data(ttl=900, show_spinner=False)
def get_data(sym, period="3y", interval="1d"):
    try:
        df = yf.download(sym, interval=interval, period=period,
                         progress=False, auto_adjust=False)
        if df is None or df.empty:
            return None
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        df = df.dropna(subset=["Close"])
        return df if len(df) > 100 else None
    except Exception:
        return None


def add_ind(df):
    d = df.copy()
    c, h, l, v = d["Close"], d["High"], d["Low"], d["Volume"]
    d["MA5"] = c.rolling(5).mean()
    d["MA10"] = c.rolling(10).mean()
    d["MA20"] = c.rolling(20).mean()
    d["MA60"] = c.rolling(60).mean()
    d["VMA20"] = v.rolling(20).mean()
    d["VR"] = v / d["VMA20"]
    return d


# ==================== 波段高低點 ====================

def find_pivots(df, n=5):
    """
    找波段高點(頭)和低點(底)。
    重點：一個高點要等右邊 n 根都比它低才能「確認」，
    所以確認時間會比實際高點晚 n 根。回測時必須用確認後的，不能偷看。
    """
    h = df["High"].values
    l = df["Low"].values
    N = len(df)

    highs = []   # (位置, 價格, 確認位置)
    lows = []

    for i in range(n, N - n):
        wl = h[i - n:i]
        wr = h[i + 1:i + n + 1]
        if len(wl) and len(wr) and h[i] > wl.max() and h[i] > wr.max():
            highs.append((i, h[i], i + n))

        wl = l[i - n:i]
        wr = l[i + 1:i + n + 1]
        if len(wl) and len(wr) and l[i] < wl.min() and l[i] < wr.min():
            lows.append((i, l[i], i + n))

    return highs, lows


def trend_structure(highs, lows, upto=None, look=3, tol=0.005):
    """
    判斷趨勢結構。
    - upto：只看確認位置在此之前的（避免偷看未來）
    - look：看最近幾個頭/底（只比兩個太敏感，一個小反彈就誤判）
    - tol：差距小於這個比例視為「差不多」，不算創新高/低
    """
    if upto is not None:
        H = [x for x in highs if x[2] <= upto]
        L = [x for x in lows if x[2] <= upto]
    else:
        H, L = highs, lows

    if len(H) < 2 or len(L) < 2:
        return "資料不足", None, None

    hs = [x[1] for x in H[-look:]]
    ls = [x[1] for x in L[-look:]]

    def direction(seq):
        """回傳 +1(一路墊高) / -1(一路走低) / 0(不一致)"""
        if len(seq) < 2:
            return 0
        ups = sum(1 for a, b in zip(seq, seq[1:]) if b > a * (1 + tol))
        dns = sum(1 for a, b in zip(seq, seq[1:]) if b < a * (1 - tol))
        n = len(seq) - 1
        if ups == n:
            return 1
        if dns == n:
            return -1
        return 0

    dh = direction(hs)
    dl = direction(ls)

    h1, h2 = H[-2][1], H[-1][1]
    l1, l2 = L[-2][1], L[-1][1]

    if dh > 0 and dl > 0:
        state = "多頭"
    elif dh < 0 and dl < 0:
        state = "空頭"
    else:
        state = "盤整"

    detail = {
        "前頭": h1, "新頭": h2,
        "頭頭高": dh > 0, "頭頭低": dh < 0,
        "前底": l1, "新底": l2,
        "底底高": dl > 0, "底底低": dl < 0,
        "頭序列": hs, "底序列": ls,
        "頭位置": H[-1][0], "底位置": L[-1][0],
    }
    return state, detail, (H, L)


# ==================== K線型態 ====================

def candle_signals(df, i):
    """判斷第 i 根 K 棒的型態"""
    o = float(df["Open"].iloc[i])
    h = float(df["High"].iloc[i])
    l = float(df["Low"].iloc[i])
    c = float(df["Close"].iloc[i])
    vr = float(df["VR"].iloc[i]) if not np.isnan(df["VR"].iloc[i]) else 1.0

    rng = h - l
    body = abs(c - o)
    if rng <= 0:
        return []

    upper = h - max(o, c)
    lower = min(o, c) - l
    sig = []

    # 長紅 / 長黑
    if body / rng > 0.6:
        pct = (c - o) / o * 100
        if c > o and pct > 2:
            sig.append(("長紅K", f"實體佔{body/rng*100:.0f}%，漲{pct:.1f}%"))
        elif c < o and pct < -2:
            sig.append(("長黑K", f"實體佔{body/rng*100:.0f}%，跌{abs(pct):.1f}%"))

    # 長上影線
    if upper / rng > 0.5 and body / rng < 0.3:
        sig.append(("長上影線", f"上影佔{upper/rng*100:.0f}%，衝高被打下來"))

    # 長下影線
    if lower / rng > 0.5 and body / rng < 0.3:
        sig.append(("長下影線", f"下影佔{lower/rng*100:.0f}%，探底有人接"))

    # 爆量
    if vr > 2.0:
        sig.append(("爆大量", f"是平常的{vr:.1f}倍"))
    elif vr > 1.5:
        sig.append(("放量", f"是平常的{vr:.1f}倍"))

    return sig


# ==================== 續抱 / 出場 ====================

def hold_or_exit(df, state, i=-1):
    """
    多頭：沒跌破昨低就續抱，跌破昨低賣出
    空頭：沒突破昨高就續抱(空單)，突破昨高回補
    """
    if len(df) < 3:
        return None

    low_y = float(df["Low"].iloc[i - 1])
    high_y = float(df["High"].iloc[i - 1])
    low_t = float(df["Low"].iloc[i])
    high_t = float(df["High"].iloc[i])
    close_t = float(df["Close"].iloc[i])

    if state == "多頭":
        broke = low_t < low_y
        return {
            "情境": "多頭持有中",
            "關鍵價": low_y,
            "觸發": broke,
            "動作": "跌破昨低 → 賣出" if broke else "沒跌破昨低 → 續抱",
            "說明": f"昨天最低 {low_y:,.2f}，今天最低 {low_t:,.2f}",
        }
    elif state == "空頭":
        broke = high_t > high_y
        return {
            "情境": "空頭（觀望或空單）",
            "關鍵價": high_y,
            "觸發": broke,
            "動作": "突破昨高 → 回補/轉多留意" if broke else "沒突破昨高 → 空頭續行",
            "說明": f"昨天最高 {high_y:,.2f}，今天最高 {high_t:,.2f}",
        }
    else:
        return {
            "情境": "盤整中",
            "關鍵價": None,
            "觸發": False,
            "動作": "等突破或跌破再說",
            "說明": "盤整時「昨高昨低」訊號會很雜，容易來回被巴",
        }


tab1, tab2 = st.tabs(["📐 現在的結構", "🔬 這套有沒有用（回測）"])


# ==================== 分頁一 ====================
with tab1:
    c1, c2 = st.columns([3, 1])
    with c1:
        sym_in = st.text_input("股票代號", "2330.TW",
                               help="台股 2330.TW｜港股 0700.HK｜美股 AAPL")
    with c2:
        piv_n = st.selectbox("波段敏感度", [3, 5, 8],
                             index=1,
                             help="數字小=抓小波段(訊號多)，大=抓大波段(訊號少)")

    if st.button("判斷", use_container_width=True, type="primary"):
        sym = fix_symbol(sym_in)
        with st.spinner("查資料中..."):
            df = get_data(sym)

        if df is None:
            st.error(f"查不到 **{sym}**")
            st.stop()

        d = add_ind(df)
        highs, lows = find_pivots(d, piv_n)
        state, det, HL = trend_structure(highs, lows)

        st.markdown("---")

        if state == "多頭":
            st.success("# 📈 多頭趨勢")
            st.write("**頭頭高、底底高** — 下一個高點比上一個高，下一個低點也比上一個高。")
        elif state == "空頭":
            st.error("# 📉 空頭趨勢")
            st.write("**頭頭低、底底低** — 下一個高點比上一個低，下一個低點也更低。")
        elif state == "盤整":
            st.warning("# ↔️ 盤整")
            st.write("**頭跟底沒有明顯方向** — 上上下下來回震。")
        else:
            st.info("# ❓ 資料不足")
            st.write("波段高低點太少，抓不出結構。把敏感度調小試試。")

        if det:
            st.write("")
            a, b = st.columns(2)
            with a:
                st.write("**頭（波段高點）**")
                st.write(f"　前一個頭：{det['前頭']:,.2f}")
                st.write(f"　最新的頭：{det['新頭']:,.2f}")
                seq = "　→　".join(f"{x:,.1f}" for x in det["頭序列"])
                st.write(f"　最近幾個頭：{seq}")
                if det["頭頭高"]:
                    st.write("　✅ **頭頭高**（一個比一個高）")
                elif det["頭頭低"]:
                    st.write("　❌ **頭頭低**（一個比一個低）")
                else:
                    st.write("　▫️ 頭沒有一致方向")
            with b:
                st.write("**底（波段低點）**")
                st.write(f"　前一個底：{det['前底']:,.2f}")
                st.write(f"　最新的底：{det['新底']:,.2f}")
                seq = "　→　".join(f"{x:,.1f}" for x in det["底序列"])
                st.write(f"　最近幾個底：{seq}")
                if det["底底高"]:
                    st.write("　✅ **底底高**（一個比一個高）")
                elif det["底底低"]:
                    st.write("　❌ **底底低**（一個比一個低）")
                else:
                    st.write("　▫️ 底沒有一致方向")

        # 續抱/出場
        st.markdown("---")
        st.write("## 今天該續抱還是出場")
        he = hold_or_exit(d, state)
        if he:
            if he["觸發"]:
                st.error(f"### ⚠️ {he['動作']}")
            else:
                st.success(f"### ✅ {he['動作']}")
            st.write(f"　{he['說明']}")
            if he["關鍵價"]:
                st.write(f"　**明天要盯的價位：{he['關鍵價']:,.2f}**")
            st.caption("這是短線的續抱判斷。多頭時只要沒跌破前一天最低就抱著，"
                       "跌破就走；空頭時反過來看昨高。")

        # K線型態
        st.markdown("---")
        st.write("## 最近幾根 K 棒的型態")
        found = False
        for k in range(1, 6):
            i = len(d) - k
            sigs = candle_signals(d, i)
            if sigs:
                found = True
                dt = d.index[i].strftime("%m/%d")
                names = "、".join(s[0] for s in sigs)
                st.write(f"**{dt}**　{names}")
                for nm, detail in sigs:
                    st.write(f"　• {nm}：{detail}")

                # 組合解讀
                nm_set = {s[0] for s in sigs}
                cl = float(d["Close"].iloc[i])
                ma20 = float(d["MA20"].iloc[i])
                high_pos = cl > ma20

                if "長上影線" in nm_set and "爆大量" in nm_set and high_pos:
                    st.error("　　→ **高檔長上影線＋爆量：轉弱訊號。**"
                             "衝高有人大量倒貨。隔天下跌就要考慮賣出。")
                elif "長黑K" in nm_set and "爆大量" in nm_set:
                    st.error("　　→ **長黑＋爆量：有人在大量出貨。**"
                             "如果是盤整末端出現，通常是要往下走。")
                elif "長紅K" in nm_set and "爆大量" in nm_set:
                    st.success("　　→ **長紅＋爆量：有人在大量買進。**"
                               "如果是突破盤整，是多頭確認訊號。")
                elif "長下影線" in nm_set and "爆大量" in nm_set:
                    st.info("　　→ **長下影＋爆量：低檔有人在接。**"
                            "可能是止跌訊號，但要等後續確認。")
        if not found:
            st.write("最近 5 根沒有明顯型態。")

        # 圖
        st.markdown("---")
        st.write("## 走勢圖（標出頭與底）")

        show = d.tail(180)
        off = len(d) - len(show)

        fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                            row_heights=[0.72, 0.28], vertical_spacing=0.04,
                            subplot_titles=("價格（紅漲綠跌）", "成交量"))

        fig.add_trace(go.Candlestick(
            x=show.index, open=show["Open"], high=show["High"],
            low=show["Low"], close=show["Close"], name="價格",
            increasing_line_color="#d62728", increasing_fillcolor="#d62728",
            decreasing_line_color="#2ca02c", decreasing_fillcolor="#2ca02c"),
            row=1, col=1)
        fig.add_trace(go.Scatter(x=show.index, y=show["MA20"], name="20日均",
                                 line=dict(color="orange", width=1.3)),
                      row=1, col=1)

        if HL:
            H, L = HL
            hx = [d.index[i] for i, p, cf in H if i >= off]
            hy = [p * 1.02 for i, p, cf in H if i >= off]
            lx = [d.index[i] for i, p, cf in L if i >= off]
            ly = [p * 0.98 for i, p, cf in L if i >= off]

            if hx:
                fig.add_trace(go.Scatter(
                    x=hx, y=hy, mode="markers+text", name="頭",
                    marker=dict(symbol="triangle-down", size=11, color="black"),
                    text=["頭"] * len(hx), textposition="top center",
                    textfont=dict(size=9)), row=1, col=1)
            if lx:
                fig.add_trace(go.Scatter(
                    x=lx, y=ly, mode="markers+text", name="底",
                    marker=dict(symbol="triangle-up", size=11, color="blue"),
                    text=["底"] * len(lx), textposition="bottom center",
                    textfont=dict(size=9)), row=1, col=1)

        vc = ["#d62728" if float(show["Close"].iloc[i]) >= float(show["Open"].iloc[i])
              else "#2ca02c" for i in range(len(show))]
        fig.add_trace(go.Bar(x=show.index, y=show["Volume"], name="量",
                             marker_color=vc, opacity=0.7), row=2, col=1)
        fig.add_trace(go.Scatter(x=show.index, y=show["VMA20"], name="20日均量",
                                 line=dict(color="blue", width=1.2)), row=2, col=1)

        fig.update_layout(height=620, xaxis_rangeslider_visible=False,
                          margin=dict(t=50, b=20), hovermode="x unified",
                          legend=dict(orientation="h", y=1.06))
        st.plotly_chart(fig, use_container_width=True)
        st.caption("黑色朝下三角=頭（波段高點）　藍色朝上三角=底（波段低點）。"
                   "注意：一個頭要等右邊幾根都比它低才能確認，所以標記會比實際晚出現。")


# ==================== 分頁二：回測 ====================
with tab2:
    st.write("### 加上趨勢結構判斷，有變好嗎？")
    st.caption("拿真實資料跑，比較「原本三條規則」和「加上頭頭高底底高」的差別。")

    c1, c2, c3 = st.columns(3)
    with c1:
        bsyms = st.text_input("股票代號（逗號分開）",
                              "2330.TW, 2317.TW, 2454.TW, 0050.TW", key="bs")
    with c2:
        bstop = st.slider("認賠出場 %", 5, 30, 15, key="bst")
    with c3:
        bcost = st.slider("單趟成本 %", 0.0, 1.0, 0.35, 0.05, key="bc")

    if st.button("開始回測", use_container_width=True, type="primary", key="bb"):
        syms = [fix_symbol(s) for s in bsyms.split(",") if s.strip()][:6]

        rows = []
        bar = st.progress(0.0)

        for n, s in enumerate(syms):
            df = get_data(s, period="10y")
            bar.progress((n + 1) / len(syms), text=f"回測 {s}")
            if df is None:
                continue
            d = add_ind(df)

            # 基本指標
            c = d["Close"]
            hh, ll = d["High"], d["Low"]
            d["MA"] = c.rolling(60).mean()
            lo, hi = ll.rolling(60).min(), hh.rolling(60).max()
            d["K"] = (100 * (c - lo) / (hi - lo + 1e-10)).rolling(3).mean()
            e12 = c.ewm(span=12, adjust=False).mean()
            e26 = c.ewm(span=26, adjust=False).mean()
            d["MACD"] = e12 - e26
            d["SIG"] = d["MACD"].ewm(span=9, adjust=False).mean()

            highs, lows = find_pivots(d, 5)

            # 逐日計算「當下已確認」的趨勢結構（不偷看未來）
            N = len(d)
            struct = np.array(["未知"] * N, dtype=object)
            hi_arr = [x for x in highs]
            lo_arr = [x for x in lows]
            for i in range(N):
                s_, _, _ = trend_structure(hi_arr, lo_arr, upto=i)
                struct[i] = s_ if s_ != "資料不足" else "未知"

            cl = d["Close"].values
            op = d["Open"].values
            ma = d["MA"].values
            K = d["K"].values
            MC = d["MACD"].values
            SG = d["SIG"].values
            lowv = d["Low"].values

            def bt(mode):
                rets = []
                pos = False
                ep = 0.0
                for i in range(63, N - 1):
                    if np.isnan(ma[i]) or np.isnan(K[i]) or np.isnan(K[i - 1]):
                        continue
                    base = (cl[i] > ma[i] and K[i] > 50 and K[i] > K[i - 1]
                            and MC[i] > SG[i])
                    if mode == "加結構":
                        ok = base and struct[i] == "多頭"
                    elif mode == "只用結構":
                        ok = struct[i] == "多頭" and cl[i] > ma[i]
                    else:
                        ok = base

                    if not pos:
                        if ok:
                            pos = True
                            ep = op[i + 1]
                            entry_i = i + 1
                    else:
                        r = (cl[i] - ep) / ep * 100
                        exit_now = False
                        if mode == "結構+昨低":
                            if lowv[i] < lowv[i - 1]:
                                exit_now = True
                        if r <= -bstop or cl[i] < ma[i]:
                            exit_now = True
                        if exit_now:
                            rets.append((op[i + 1] - ep) / ep * 100 - bcost * 2)
                            pos = False
                if len(rets) < 3:
                    return None
                a = np.array(rets)
                eq = 100.0
                for x in a:
                    eq *= (1 + x / 100)
                sd = a.std(ddof=1) if len(a) > 1 else 0
                tv = a.mean() / (sd / np.sqrt(len(a))) if sd > 0 else 0
                return {
                    "筆數": len(a),
                    "勝率%": round(len(a[a > 0]) / len(a) * 100, 1),
                    "每筆平均%": round(a.mean(), 2),
                    "t值": round(tv, 2),
                    "總報酬%": round(eq - 100, 1),
                }

            bh = (cl[-1] - cl[63]) / cl[63] * 100
            for mode in ["原本三條", "加結構", "只用結構", "結構+昨低"]:
                r = bt(mode)
                rows.append({
                    "股票": s, "方式": mode,
                    "筆數": r["筆數"] if r else 0,
                    "勝率%": r["勝率%"] if r else None,
                    "每筆平均%": r["每筆平均%"] if r else None,
                    "t值": r["t值"] if r else None,
                    "總報酬%": r["總報酬%"] if r else None,
                    "買著不動%": round(bh, 1),
                })
        bar.empty()

        if not rows:
            st.error("沒抓到資料")
            st.stop()

        res = pd.DataFrame(rows)
        st.markdown("---")
        st.write("### 明細")

        def col(v):
            try:
                return "color: green" if v > 0 else "color: red"
            except Exception:
                return ""

        st.dataframe(
            res.style.map(col, subset=["每筆平均%", "總報酬%", "買著不動%"]),
            use_container_width=True, hide_index=True)

        st.markdown("---")
        st.write("### 四種方式平均比較")
        g = res.dropna(subset=["每筆平均%"]).groupby("方式").agg(
            平均每筆=("每筆平均%", "mean"),
            平均筆數=("筆數", "mean"),
            平均勝率=("勝率%", "mean"),
            正報酬檔數=("每筆平均%", lambda x: (x > 0).sum()),
        ).round(2)
        st.dataframe(g, use_container_width=True)

        st.markdown("---")
        st.write("### 白話結論")

        if not g.empty:
            base = g.loc["原本三條", "平均每筆"] if "原本三條" in g.index else None
            best_name = g["平均每筆"].idxmax()
            best_val = g["平均每筆"].max()

            st.write(f"**最好的是「{best_name}」**，平均每筆 {best_val:+.2f}%")

            if base is not None:
                st.write(f"原本三條規則是 {base:+.2f}%")
                st.write("")
                if best_name == "原本三條":
                    st.warning("**加上趨勢結構沒有變好。**　"
                               "頭頭高底底高是很直觀的概念，但在這幾檔上，"
                               "加了它反而讓訊號變少、報酬變差。建議不要加。")
                else:
                    diff = best_val - base
                    st.success(f"**加上趨勢結構有幫助**，每筆多賺 {diff:.2f} 個百分點。")
                    n_best = g.loc[best_name, "平均筆數"]
                    if n_best < 10:
                        st.warning(f"⚠️ 但平均只做 {n_best:.0f} 次，"
                                   f"樣本太少，這個結果不可靠。")

            st.write("")
            st.write("**四種方式的意思：**")
            st.write("　• **原本三條** — 站上60均、K值>50且上升、MACD轉強")
            st.write("　• **加結構** — 上面三條，而且要「頭頭高底底高」")
            st.write("　• **只用結構** — 只看頭頭高底底高＋站上60均，不看KD和MACD")
            st.write("　• **結構+昨低** — 加結構，出場改成「跌破昨低就走」")

st.markdown("---")
st.caption("趨勢結構判斷源自道氏理論（Dow Theory），是流傳超過百年的經典技術分析概念。"
           "回測已扣成本、訊號隔日開盤成交、波段高低點使用確認後才採用（不偷看未來）。研究工具，不是投資建議。")
