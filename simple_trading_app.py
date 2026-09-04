import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import requests
from datetime import datetime, timedelta
import plotly.graph_objects as go
from plotly.subplots import make_subplots

st.set_page_config(page_title="股票分析系統", page_icon="📊", layout="wide")

# ============================================================
# 基礎工具
# ============================================================

def fix_symbol(s):
    s = s.strip().upper().replace(" ", "")
    if s.endswith(".HK"):
        n = s[:-3].lstrip("0")
        if n.isdigit():
            s = n.zfill(4) + ".HK"
    return s


def is_tw(sym):
    return sym.endswith(".TW") or sym.endswith(".TWO") or sym == "^TWII"


def market_of(sym):
    if is_tw(sym):
        return "台股", "^TWII", "元"
    if sym.endswith(".HK") or sym == "^HSI":
        return "港股", "^HSI", "港幣"
    return "美股", "^GSPC", "美元"


TF = {
    "日線": ("1d", "5y", 60),
    "週線": ("1wk", "10y", 60),
    "月線": ("1mo", "max", 36),
}


@st.cache_data(ttl=600, show_spinner=False)
def get_data(sym, interval="1d", period="5y"):
    try:
        df = yf.download(sym, interval=interval, period=period,
                         progress=False, auto_adjust=False)
        if df is None or df.empty:
            return None
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        df = df.dropna(subset=["Close"])
        return df if len(df) > 40 else None
    except Exception:
        return None


@st.cache_data(ttl=1800, show_spinner=False)
def get_info(sym):
    try:
        info = yf.Ticker(sym).info
        return info if isinstance(info, dict) and len(info) > 3 else {}
    except Exception:
        return {}


def add_ind(df, ma_long=60):
    d = df.copy()
    c, h, l, v = d["Close"], d["High"], d["Low"], d["Volume"]

    d["MA5"] = c.rolling(5).mean()
    d["MA10"] = c.rolling(10).mean()
    d["MA20"] = c.rolling(20).mean()
    d["MA60"] = c.rolling(ma_long).mean()

    # KD (參數 60,3,3)
    lo, hi = l.rolling(60).min(), h.rolling(60).max()
    rsv = 100 * (c - lo) / (hi - lo + 1e-10)
    d["K"] = rsv.rolling(3).mean()
    d["D"] = d["K"].rolling(3).mean()

    # KD (標準 9,3,3) 供對照
    lo9, hi9 = l.rolling(9).min(), h.rolling(9).max()
    rsv9 = 100 * (c - lo9) / (hi9 - lo9 + 1e-10)
    d["K9"] = rsv9.rolling(3).mean()
    d["D9"] = d["K9"].rolling(3).mean()

    # RSI 14
    delta = c.diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = (-delta.clip(upper=0)).rolling(14).mean()
    d["RSI"] = 100 - 100 / (1 + gain / (loss + 1e-10))

    # MACD
    e12 = c.ewm(span=12, adjust=False).mean()
    e26 = c.ewm(span=26, adjust=False).mean()
    d["MACD"] = e12 - e26
    d["SIG"] = d["MACD"].ewm(span=9, adjust=False).mean()
    d["HIST"] = d["MACD"] - d["SIG"]

    d["VMA20"] = v.rolling(20).mean()
    d["VR"] = v / d["VMA20"]

    return d


def trend_of(d):
    ma = d["MA60"].values
    ok = ma[~np.isnan(ma)]
    if len(ok) < 25:
        return "不明", 0.0
    look = min(40, len(ok) - 1)
    s = (ok[-1] - ok[-look]) / ok[-look] * 100
    if s > 3:
        return "往上", s
    if s < -3:
        return "往下", s
    return "橫盤", s


# ============================================================
# 趨勢結構（道氏理論）
# ============================================================

def find_pivots(df, n=5):
    h, l = df["High"].values, df["Low"].values
    N = len(df)
    highs, lows = [], []
    for i in range(n, N - n):
        if h[i] > h[i - n:i].max() and h[i] > h[i + 1:i + n + 1].max():
            highs.append((i, h[i], i + n))
        if l[i] < l[i - n:i].min() and l[i] < l[i + 1:i + n + 1].min():
            lows.append((i, l[i], i + n))
    return highs, lows


def trend_structure(highs, lows, upto=None, look=3, tol=0.005):
    H = [x for x in highs if upto is None or x[2] <= upto]
    L = [x for x in lows if upto is None or x[2] <= upto]
    if len(H) < 2 or len(L) < 2:
        return "資料不足", None, None

    hs = [x[1] for x in H[-look:]]
    ls = [x[1] for x in L[-look:]]

    def direction(seq):
        if len(seq) < 2:
            return 0
        n = len(seq) - 1
        ups = sum(1 for a, b in zip(seq, seq[1:]) if b > a * (1 + tol))
        dns = sum(1 for a, b in zip(seq, seq[1:]) if b < a * (1 - tol))
        return 1 if ups == n else (-1 if dns == n else 0)

    dh, dl = direction(hs), direction(ls)
    state = "多頭" if (dh > 0 and dl > 0) else ("空頭" if (dh < 0 and dl < 0) else "盤整")

    return state, {"頭序列": hs, "底序列": ls,
                   "頭頭高": dh > 0, "頭頭低": dh < 0,
                   "底底高": dl > 0, "底底低": dl < 0}, (H, L)


def candle_signals(df, i):
    o = float(df["Open"].iloc[i]); h = float(df["High"].iloc[i])
    l = float(df["Low"].iloc[i]);  c = float(df["Close"].iloc[i])
    vr = df["VR"].iloc[i]
    vr = float(vr) if not np.isnan(vr) else 1.0
    rng = h - l
    if rng <= 0:
        return []
    body = abs(c - o)
    upper = h - max(o, c)
    lower = min(o, c) - l
    sig = []
    pct = (c - o) / o * 100
    if body / rng > 0.6 and pct > 2:
        sig.append(("長紅K", f"實體佔{body/rng*100:.0f}%，漲{pct:.1f}%"))
    if body / rng > 0.6 and pct < -2:
        sig.append(("長黑K", f"實體佔{body/rng*100:.0f}%，跌{abs(pct):.1f}%"))
    if upper / rng > 0.5 and body / rng < 0.3:
        sig.append(("長上影線", f"上影佔{upper/rng*100:.0f}%，衝高被打下來"))
    if lower / rng > 0.5 and body / rng < 0.3:
        sig.append(("長下影線", f"下影佔{lower/rng*100:.0f}%，探底有人接"))
    if vr > 2.0:
        sig.append(("爆大量", f"是平常的{vr:.1f}倍"))
    elif vr > 1.5:
        sig.append(("放量", f"是平常的{vr:.1f}倍"))
    return sig


# ============================================================
# 法人資料（台股）
# ============================================================

@st.cache_data(ttl=1800, show_spinner=False)
def fetch_t86(date_str):
    url = ("https://www.twse.com.tw/rwd/zh/fund/T86"
           f"?date={date_str}&selectType=ALL&response=json")
    r = requests.get(url, timeout=20, headers={"User-Agent": "Mozilla/5.0"})
    r.raise_for_status()
    j = r.json()
    if j.get("stat") != "OK" or not j.get("data"):
        return None
    return pd.DataFrame(j["data"], columns=j["fields"])


@st.cache_data(ttl=1800, show_spinner=False)
def get_inst(code, want=10):
    out = []
    today = datetime.now()
    for i in range(20):
        if len(out) >= want:
            break
        ds = (today - timedelta(days=i)).strftime("%Y%m%d")
        try:
            t = fetch_t86(ds)
        except Exception:
            continue
        if t is None:
            continue
        ccol = next((x for x in t.columns if "代號" in x), None)
        if not ccol:
            continue
        row = t[t[ccol].astype(str).str.strip() == code]
        if row.empty:
            continue
        row = row.iloc[0]

        def pick(*kw):
            for k in row.index:
                if all(w in k for w in kw):
                    try:
                        return float(str(row[k]).replace(",", "")) / 1000
                    except Exception:
                        return np.nan
            return np.nan

        fo = pick("外陸資", "買賣超")
        if np.isnan(fo):
            fo = pick("外資", "買賣超")
        out.append({"日期": f"{ds[4:6]}/{ds[6:]}", "外資": fo,
                    "投信": pick("投信", "買賣超"),
                    "自營": pick("自營商", "買賣超")})
    return pd.DataFrame(out).iloc[::-1].reset_index(drop=True) if out else None


def streak_of(arr):
    a = [x for x in arr if not np.isnan(x)]
    if not a:
        return 0
    sign = 1 if a[-1] > 0 else (-1 if a[-1] < 0 else 0)
    if sign == 0:
        return 0
    n = 0
    for x in a[::-1]:
        if (x > 0 and sign > 0) or (x < 0 and sign < 0):
            n += 1
        else:
            break
    return n * sign


# 集保大戶
TDCC_URL = "https://www.tdcc.com.tw/portal/zh/smWeb/qryStock"
UA = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                    "AppleWebKit/537.36 Chrome/120.0 Safari/537.36",
      "Referer": TDCC_URL}


@st.cache_data(ttl=3600, show_spinner=False)
def tdcc_session():
    import re
    s = requests.Session()
    r = s.get(TDCC_URL, headers=UA, timeout=25)
    r.raise_for_status()
    tok = re.search(r'name="SYNCHRONIZER_TOKEN"[^>]*value="([^"]*)"', r.text)
    uri = re.search(r'name="SYNCHRONIZER_URI"[^>]*value="([^"]*)"', r.text)
    dates = sorted(set(re.findall(r'<option[^>]*value="(\d{8})"', r.text)),
                   reverse=True)
    return {"cookies": dict(s.cookies),
            "token": tok.group(1) if tok else "",
            "uri": uri.group(1) if uri else "/portal/zh/smWeb/qryStock",
            "dates": dates}


@st.cache_data(ttl=3600, show_spinner=False)
def tdcc_week(code, date, token, uri, cookies):
    from io import StringIO
    s = requests.Session()
    s.cookies.update(cookies)
    payload = {"SYNCHRONIZER_TOKEN": token, "SYNCHRONIZER_URI": uri,
               "method": "submit", "firDate": date, "scaDate": date,
               "sqlMethod": "StockNo", "stockNo": code, "StockNo": code,
               "stockName": "", "StockName": "", "radioStockNo": code}
    r = s.post(TDCC_URL, data=payload, headers=UA, timeout=25)
    r.raise_for_status()
    try:
        tables = pd.read_html(StringIO(r.text))
    except Exception:
        return None
    for t in tables:
        cols = "".join(str(x) for x in t.columns)
        if "持股分級" in cols and "人數" in cols:
            return t
    return None


def parse_tdcc(t):
    df = t.copy()
    df.columns = [str(c).strip() for c in df.columns]
    c_lv = next((c for c in df.columns if "分級" in c), None)
    c_pct = next((c for c in df.columns if "比例" in c), None)
    c_ppl = next((c for c in df.columns if "人數" in c), None)
    if not (c_lv and c_pct):
        return None
    df["_lv"] = pd.to_numeric(df[c_lv].astype(str).str.extract(r"(\d+)")[0],
                              errors="coerce")
    df["_pct"] = pd.to_numeric(df[c_pct].astype(str).str.replace(",", "")
                               .str.replace("%", ""), errors="coerce")
    df["_ppl"] = pd.to_numeric(df[c_ppl].astype(str).str.replace(",", ""),
                               errors="coerce") if c_ppl else np.nan
    big = df[(df["_lv"] >= 12) & (df["_lv"] <= 15)]
    sml = df[(df["_lv"] >= 1) & (df["_lv"] <= 3)]
    if big.empty:
        return None
    return {"大戶比例": float(big["_pct"].sum()),
            "散戶比例": float(sml["_pct"].sum()) if not sml.empty else np.nan,
            "大戶人數": int(big["_ppl"].sum()) if big["_ppl"].notna().any() else None}


# ============================================================
# 側邊欄
# ============================================================

st.sidebar.title("⚙️ 設定")

sym_in = st.sidebar.text_input("股票代號", "2330.TW",
                               help="台股 2330.TW｜港股 0700.HK｜美股 AAPL")
tf_name = st.sidebar.radio("K線週期", list(TF.keys()), index=0, horizontal=True)

st.sidebar.markdown("---")
st.sidebar.write("**要看什麼（可複選）**")

show_sum = st.sidebar.checkbox("📌 綜合結論", True)
show_ma = st.sidebar.checkbox("📈 K線與均線", True)
show_kd = st.sidebar.checkbox("📊 KD 指標", True)
show_rsi = st.sidebar.checkbox("📊 RSI 指標", True)
show_macd = st.sidebar.checkbox("📊 MACD 指標", True)
show_vol = st.sidebar.checkbox("📊 量價關係", True)
show_struct = st.sidebar.checkbox("📐 趨勢結構（頭頭高底底高）", True)
show_candle = st.sidebar.checkbox("🕯️ K線型態", True)
show_inst = st.sidebar.checkbox("🏦 法人買賣超（台股）", True)
show_big = st.sidebar.checkbox("🐋 大戶持股（台股，較慢）", False)
show_fund = st.sidebar.checkbox("🏢 公司基本面", True)
show_mkt = st.sidebar.checkbox("🌏 大盤環境", True)

st.sidebar.markdown("---")
st.sidebar.write("**持股（選填）**")
my_cost = st.sidebar.number_input("你的買進價", 0.0, step=1.0, format="%.2f")
my_qty = st.sidebar.number_input("張數／股數", 0.0, step=1.0)
my_stop = st.sidebar.slider("認賠出場 %", 5, 30, 15)

run = st.sidebar.button("開始分析", use_container_width=True, type="primary")

st.title("📊 股票分析系統")

tab1, tab2, tab3 = st.tabs(["📍 分析", "🔬 驗證回測", "📖 說明"])


# ============================================================
# 分頁一：分析
# ============================================================
with tab1:
    if not run:
        st.info("👈 左邊設定好之後，按「開始分析」")
        st.write("**這裡可以看什麼：**")
        st.write("　• K線（日／週／月）＋ 均線")
        st.write("　• KD、RSI、MACD 三個指標的圖和判讀")
        st.write("　• 量價關係、趨勢結構、K線型態")
        st.write("　• 法人買賣超、大戶持股、公司基本面")
        st.write("　• 最上面會給一個綜合結論")
        st.stop()

    sym = fix_symbol(sym_in)
    if sym != sym_in.strip().upper():
        st.info(f"代號已修正：{sym_in} → **{sym}**")

    mkt, idx_sym, cur = market_of(sym)
    interval, period, ma_long = TF[tf_name]
    code = sym.replace(".TWO", "").replace(".TW", "")

    with st.spinner("查資料中..."):
        df = get_data(sym, interval, period)
        dfi = get_data(idx_sym, "1d", "2y") if show_mkt else None
        info = get_info(sym) if show_fund else {}
        inst = (get_inst(code) if (show_inst and is_tw(sym)
                                   and not sym.startswith("^")) else None)

    if df is None:
        st.error(f"查不到 **{sym}**")
        st.write("台股 2330.TW（上櫃 .TWO）｜港股湊四位數 0700.HK｜美股 AAPL")
        st.stop()

    d = add_ind(df, ma_long)
    last, prev = d.iloc[-1], d.iloc[-2]

    price = float(last["Close"])
    ma60 = float(last["MA60"]); ma20 = float(last["MA20"])
    ma5 = float(last["MA5"]); ma10 = float(last["MA10"])
    k_v, k_p = float(last["K"]), float(prev["K"])
    d_v = float(last["D"])
    rsi = float(last["RSI"])
    macd, sigl = float(last["MACD"]), float(last["SIG"])
    hist, hist_p = float(last["HIST"]), float(prev["HIST"])
    vr5 = float(d["VR"].tail(5).mean())
    chg5 = (price / float(d["Close"].iloc[-6]) - 1) * 100 if len(d) > 6 else 0

    trend, slope = trend_of(d)

    # 量價
    if chg5 > 2 and vr5 > 1.3:
        pv = ("價漲量增", 2, "有人真的拿錢在買，健康")
    elif chg5 > 2 and vr5 < 0.75:
        pv = ("價漲量縮", -1, "沒人跟著買，漲勢撐不久")
    elif chg5 < -2 and vr5 > 1.3:
        pv = ("價跌量增", -3, "有人在倒貨，最危險")
    elif chg5 < -2 and vr5 < 0.75:
        pv = ("價跌量縮", -1, "賣壓在減輕")
    else:
        pv = ("量價平穩", 0, "沒有特別訊號")

    # 結構
    highs, lows = find_pivots(d, 5)
    st_state, st_det, HL = trend_structure(highs, lows)

    # 三條規則
    r1 = price > ma60
    r2 = (k_v > 50) and (k_v > k_p)
    r3 = macd > sigl
    hit = sum([r1, r2, r3])

    # ---------- 綜合結論 ----------
    if show_sum:
        score = 0
        plus, minus = [], []

        if trend == "往上":
            score += 3; plus.append(f"大方向往上（{ma_long}期均價升 {slope:.1f}%）")
        elif trend == "往下":
            score -= 3; minus.append(f"大方向往下（{ma_long}期均價跌 {abs(slope):.1f}%）")

        if price > ma5 > ma10 > ma20 > ma60:
            score += 2; plus.append("均線完美多頭排列")
        elif r1:
            score += 1; plus.append(f"站在{ma_long}期均價之上")
        else:
            score -= 2; minus.append(f"跌破{ma_long}期均價")

        if r2:
            score += 2; plus.append(f"K值 {k_v:.0f} 站上50且上升")
        elif k_v < 50 and k_v < k_p:
            score -= 2; minus.append(f"K值 {k_v:.0f} 不到50且下降")
        if k_v > 80:
            score -= 1; minus.append(f"K值 {k_v:.0f} 超買區，追高風險")

        if (float(prev["MACD"]) <= float(prev["SIG"])) and r3:
            score += 2; plus.append("MACD剛翻多")
        elif r3:
            score += 1; plus.append("MACD維持多方")
        else:
            score -= 1; minus.append("MACD在空方")

        if 50 < rsi < 75:
            score += 1; plus.append(f"RSI {rsi:.0f} 健康區間")
        elif rsi >= 80:
            score -= 1; minus.append(f"RSI {rsi:.0f} 過熱")

        score += pv[1]
        if pv[1] > 0:
            plus.append(f"{pv[0]}：{pv[2]}")
        elif pv[1] < 0:
            minus.append(f"{pv[0]}：{pv[2]}")

        if st_state == "多頭":
            score += 2; plus.append("趨勢結構：頭頭高、底底高")
        elif st_state == "空頭":
            score -= 2; minus.append("趨勢結構：頭頭低、底底低")

        if inst is not None and len(inst) >= 3:
            fs = streak_of(inst["外資"].tolist())
            iv = streak_of(inst["投信"].tolist())
            if fs >= 3 and iv >= 3:
                score += 2; plus.append(f"外資連買{fs}天、投信連買{iv}天")
            elif fs >= 3:
                score += 1; plus.append(f"外資連買{fs}天")
            elif iv >= 3:
                score += 1; plus.append(f"投信連買{iv}天")
            elif fs <= -3:
                score -= 1; minus.append(f"外資連賣{abs(fs)}天")

        if dfi is not None:
            di = add_ind(dfi)
            if float(di["Close"].iloc[-1]) > float(di["MA60"].iloc[-1]):
                score += 1; plus.append(f"{mkt}大盤在均價之上")
            else:
                score -= 2; minus.append(f"{mkt}大盤跌破均價")

        st.write("## 📌 綜合結論")

        if trend == "往下":
            st.error("# 🚫 建議避開")
            st.write("這檔整體在跌。回測顯示：股票本身在跌的時候，"
                     "照這套規則做會賠得比「買了放著不動」更慘。")
        elif score >= 9:
            st.success("# 🟢 條件很好")
        elif score >= 5:
            st.success("# 🟢 條件不錯")
            st.write("不是最完美的狀況，要進的話部位小一點。")
        elif score >= 2:
            st.warning("# 🟡 再等等")
        else:
            st.error("# 🔴 現在別進")

        st.write(f"**綜合分數 {score} 分**　"
                 f"（{tf_name} · 三條規則 {hit}/3 · 趨勢結構 {st_state}）")

        a, b = st.columns(2)
        with a:
            st.write("### ✅ 好的地方")
            for x in plus:
                st.write(f"　• {x}")
            if not plus:
                st.write("　（沒有）")
        with b:
            st.write("### ⚠️ 不好的地方")
            for x in minus:
                st.write(f"　• {x}")
            if not minus:
                st.write("　（沒有）")

        st.markdown("---")
        m1, m2, m3, m4, m5 = st.columns(5)
        m1.metric("現價", f"{price:,.2f}")
        m2.metric(f"{ma_long}期均價", f"{ma60:,.2f}", f"{(price/ma60-1)*100:+.1f}%")
        m3.metric("K值", f"{k_v:.0f}", f"{k_v-k_p:+.0f}")
        m4.metric("RSI", f"{rsi:.0f}")
        m5.metric("近5期量", f"{vr5:.2f} 倍")

        # 持股損益
        if my_cost > 0:
            st.markdown("---")
            st.write("### 💰 你的持股")
            pnl = (price - my_cost) / my_cost * 100
            sl = my_cost * (1 - my_stop / 100)
            unit = 1000 if (sym.endswith(".TW") or sym.endswith(".TWO")
                            or sym.endswith(".HK")) else 1
            money = (price - my_cost) * my_qty * unit if my_qty > 0 else None

            p1, p2, p3, p4 = st.columns(4)
            p1.metric("你的成本", f"{my_cost:,.2f}")
            p2.metric("現價", f"{price:,.2f}", f"{pnl:+.2f}%")
            if money is not None:
                p3.metric("目前損益", f"{money:+,.0f}")
            p4.metric("停損價", f"{sl:,.2f}")

            if price <= sl:
                st.error(f"🚨 **已跌破你的停損價 {sl:,.2f}。這是你自己設的線，"
                         f"跌破就走，不要凹。**")
            elif (price - sl) / price * 100 < 3:
                st.warning(f"⚠️ 快到停損價了，再跌 "
                           f"{(price-sl)/price*100:.1f}% 就到 {sl:,.2f}")
            else:
                st.info(f"距離停損價還有 {(price-sl)/price*100:.1f}%（{sl:,.2f}）")

            if price < ma60:
                st.error(f"⚠️ 已跌破{ma_long}期均價 {ma60:,.2f}，這也是出場條件。")
            else:
                st.write(f"距離跌破{ma_long}期均價還有 "
                         f"{(price-ma60)/price*100:.1f}%（{ma60:,.2f}）")

            if pnl > 20:
                st.success(f"目前賺 {pnl:.1f}%。回測顯示賺大錢的都是抱三到五個月那幾筆，"
                           f"只要沒跌破出場條件，提早了結反而少賺。")

        st.markdown("---")

    # ---------- K線與均線 ----------
    if show_ma:
        st.write(f"## 📈 {tf_name} K線與均線")

        n_show = {"日線": 180, "週線": 150, "月線": 120}[tf_name]
        show = d.tail(n_show)
        off = len(d) - len(show)

        fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                            row_heights=[0.75, 0.25], vertical_spacing=0.04,
                            subplot_titles=(f"{tf_name}（紅漲綠跌）", "成交量"))
        fig.add_trace(go.Candlestick(
            x=show.index, open=show["Open"], high=show["High"],
            low=show["Low"], close=show["Close"], name="價格",
            increasing_line_color="#d62728", increasing_fillcolor="#d62728",
            decreasing_line_color="#2ca02c", decreasing_fillcolor="#2ca02c"),
            row=1, col=1)
        for col, cname, w in [("MA5", "5期", 1), ("MA20", "20期", 1.3),
                              ("MA60", f"{ma_long}期", 2.2)]:
            fig.add_trace(go.Scatter(x=show.index, y=show[col], name=cname,
                                     line=dict(width=w)), row=1, col=1)

        if show_struct and HL:
            H, L = HL
            hx = [d.index[i] for i, p, cf in H if i >= off]
            hy = [p * 1.02 for i, p, cf in H if i >= off]
            lx = [d.index[i] for i, p, cf in L if i >= off]
            ly = [p * 0.98 for i, p, cf in L if i >= off]
            if hx:
                fig.add_trace(go.Scatter(x=hx, y=hy, mode="markers", name="頭",
                    marker=dict(symbol="triangle-down", size=10, color="black")),
                    row=1, col=1)
            if lx:
                fig.add_trace(go.Scatter(x=lx, y=ly, mode="markers", name="底",
                    marker=dict(symbol="triangle-up", size=10, color="blue")),
                    row=1, col=1)

        vc = ["#d62728" if float(show["Close"].iloc[i]) >= float(show["Open"].iloc[i])
              else "#2ca02c" for i in range(len(show))]
        fig.add_trace(go.Bar(x=show.index, y=show["Volume"], name="量",
                             marker_color=vc, opacity=0.7), row=2, col=1)
        fig.add_trace(go.Scatter(x=show.index, y=show["VMA20"], name="均量",
                                 line=dict(color="blue", width=1.2)), row=2, col=1)
        fig.update_layout(height=560, xaxis_rangeslider_visible=False,
                          margin=dict(t=50, b=20), hovermode="x unified",
                          legend=dict(orientation="h", y=1.05))
        st.plotly_chart(fig, use_container_width=True)

        st.write(f"**均線排列：** 現價 {price:,.2f}｜5期 {ma5:,.2f}｜"
                 f"10期 {ma10:,.2f}｜20期 {ma20:,.2f}｜{ma_long}期 {ma60:,.2f}")
        if price > ma5 > ma10 > ma20 > ma60:
            st.success("✅ **完美多頭排列** — 短天期的線在上、長天期在下，一層層往上疊。"
                       "代表不管什麼時候買的人現在都是賺的。")
        elif price < ma5 < ma10 < ma20 < ma60:
            st.error("❌ **完美空頭排列** — 一層層往下壓。反彈都會有解套賣壓。")
        elif price > ma60:
            st.info("▫️ 部分排列 — 大方向還在均線上，但中間有線交纏，還沒整理好。")
        else:
            st.warning("⚠️ 跌破長期均線 — 這是你設定的出場條件。")

        st.markdown("---")

    # ---------- KD ----------
    if show_kd:
        st.write("## 📊 KD 指標")

        kd_pick = st.radio("KD 參數", ["你的設定 (60,3,3)", "標準 (9,3,3)"],
                           horizontal=True, key="kdp")
        kcol, dcol = ("K", "D") if "60" in kd_pick else ("K9", "D9")
        kk = float(d[kcol].iloc[-1]); kkp = float(d[kcol].iloc[-2])
        dd = float(d[dcol].iloc[-1]); ddp = float(d[dcol].iloc[-2])

        a, b, c = st.columns(3)
        a.metric("K值", f"{kk:.1f}", f"{kk-kkp:+.1f}")
        b.metric("D值", f"{dd:.1f}", f"{dd-ddp:+.1f}")
        c.metric("K−D", f"{kk-dd:+.1f}")

        gold = (kkp <= ddp) and (kk > dd)
        dead = (kkp >= ddp) and (kk < dd)

        if gold:
            st.success(f"### 🟡 黃金交叉（K往上穿過D）")
            if kk < 30:
                st.write("在**低檔**交叉，這是比較好的位置。")
            elif kk > 80:
                st.write("但已在 **80 以上高檔**，追高風險大。")
        elif dead:
            st.error(f"### ⚫ 死亡交叉（K往下穿過D）")
            if kk > 70:
                st.write("在**高檔**交叉，通常是要回檔了。")
        else:
            st.info(f"### ▫️ 沒有交叉，目前 K 在 D "
                    f"{'上面（偏強）' if kk > dd else '下面（偏弱）'}")

        if kk > 80:
            st.warning(f"⚠️ K值 {kk:.0f} 在**超買區**（80以上）。已經漲一段了，這時候買是追高。")
        elif kk < 20:
            st.warning(f"⚠️ K值 {kk:.0f} 在**超賣區**（20以下）。跌深了，但不代表馬上會反彈。")

        st.write(f"**你的規則要求：K值>50 且上升** → "
                 f"{'✅ 符合' if r2 else '❌ 不符合'}")

        show = d.tail({"日線": 180, "週線": 150, "月線": 120}[tf_name])
        f = go.Figure()
        f.add_trace(go.Scatter(x=show.index, y=show[kcol], name="K",
                               line=dict(color="blue", width=2)))
        f.add_trace(go.Scatter(x=show.index, y=show[dcol], name="D",
                               line=dict(color="orange", width=2)))
        f.add_hline(y=80, line_dash="dash", line_color="red",
                    annotation_text="超買 80")
        f.add_hline(y=50, line_dash="dot", line_color="gray")
        f.add_hline(y=20, line_dash="dash", line_color="green",
                    annotation_text="超賣 20")
        f.update_layout(height=280, margin=dict(t=20, b=20),
                        hovermode="x unified", yaxis_range=[0, 100])
        st.plotly_chart(f, use_container_width=True)
        st.caption("K線（藍）比較快，D線（橘）比較慢。K往上穿過D=黃金交叉，"
                   "往下穿過=死亡交叉。80以上超買，20以下超賣。")
        st.markdown("---")

    # ---------- RSI ----------
    if show_rsi:
        st.write("## 📊 RSI 指標")
        rsi_p = float(d["RSI"].iloc[-2])
        a, b = st.columns(2)
        a.metric("RSI (14)", f"{rsi:.1f}", f"{rsi-rsi_p:+.1f}")
        b.metric("位置", "超買" if rsi >= 70 else ("超賣" if rsi <= 30 else "中性"))

        if rsi >= 80:
            st.error(f"### ⚠️ {rsi:.0f} — 嚴重超買")
            st.write("漲太快太多了。這種位置追進去，很容易買在轉折點。")
        elif rsi >= 70:
            st.warning(f"### 🟡 {rsi:.0f} — 超買區")
            st.write("偏熱。強勢股可以在超買區待很久，但要小心。")
        elif rsi >= 50:
            st.success(f"### ✅ {rsi:.0f} — 健康偏多")
            st.write("買方力道大於賣方，但還沒過熱。這是比較舒服的位置。")
        elif rsi >= 30:
            st.info(f"### ▫️ {rsi:.0f} — 偏弱")
            st.write("賣方比較有力。")
        else:
            st.warning(f"### ⚠️ {rsi:.0f} — 超賣區")
            st.write("跌深了。但**超賣不等於會漲**，弱勢股可以一路超賣一直跌。")

        show = d.tail({"日線": 180, "週線": 150, "月線": 120}[tf_name])
        f = go.Figure()
        f.add_trace(go.Scatter(x=show.index, y=show["RSI"], name="RSI",
                               line=dict(color="purple", width=2)))
        f.add_hline(y=70, line_dash="dash", line_color="red",
                    annotation_text="超買 70")
        f.add_hline(y=50, line_dash="dot", line_color="gray")
        f.add_hline(y=30, line_dash="dash", line_color="green",
                    annotation_text="超賣 30")
        f.update_layout(height=260, margin=dict(t=20, b=20),
                        hovermode="x unified", yaxis_range=[0, 100])
        st.plotly_chart(f, use_container_width=True)
        st.caption("RSI 是「最近漲的力道 vs 跌的力道」的比例。"
                   "70以上代表漲得又快又多，30以下代表跌得又快又多。")
        st.markdown("---")

    # ---------- MACD ----------
    if show_macd:
        st.write("## 📊 MACD 指標")
        a, b, c = st.columns(3)
        a.metric("MACD", f"{macd:+.3f}")
        b.metric("訊號線", f"{sigl:+.3f}")
        c.metric("柱狀體", f"{hist:+.3f}", f"{hist-hist_p:+.3f}")

        turn_up = (float(prev["MACD"]) <= float(prev["SIG"])) and (macd > sigl)
        turn_dn = (float(prev["MACD"]) >= float(prev["SIG"])) and (macd < sigl)

        if turn_up:
            st.success("### 🟢 MACD 剛翻多（黃金交叉）")
            st.write("動能剛轉向。這通常是波段的起點。")
        elif turn_dn:
            st.error("### 🔴 MACD 剛翻空（死亡交叉）")
            st.write("動能轉弱。手上有的要留意。")
        elif macd > sigl:
            if hist > hist_p:
                st.success("### ✅ 多方且力道還在加強")
                st.write("柱狀體變長，代表上漲力道越來越強。")
            else:
                st.info("### ▫️ 多方但力道在減弱")
                st.write("柱狀體變短。還在漲，但一次比一次沒力，通常快停了。")
        else:
            if hist < hist_p:
                st.error("### ❌ 空方且力道在加強")
            else:
                st.warning("### 🟡 空方但跌勢在收斂")
                st.write("柱狀體縮短，跌勢可能快緩了。")

        if macd > 0 and sigl > 0:
            st.write("兩條線都在 0 以上 → 中期偏多")
        elif macd < 0 and sigl < 0:
            st.write("兩條線都在 0 以下 → 中期偏空")

        show = d.tail({"日線": 180, "週線": 150, "月線": 120}[tf_name])
        f = go.Figure()
        hc = ["#d62728" if x > 0 else "#2ca02c" for x in show["HIST"]]
        f.add_trace(go.Bar(x=show.index, y=show["HIST"], name="柱狀體",
                           marker_color=hc, opacity=0.6))
        f.add_trace(go.Scatter(x=show.index, y=show["MACD"], name="MACD",
                               line=dict(color="blue", width=2)))
        f.add_trace(go.Scatter(x=show.index, y=show["SIG"], name="訊號線",
                               line=dict(color="orange", width=2)))
        f.add_hline(y=0, line_color="black")
        f.update_layout(height=280, margin=dict(t=20, b=20),
                        hovermode="x unified")
        st.plotly_chart(f, use_container_width=True)
        st.caption("藍線穿過橘線往上=翻多，往下=翻空。柱狀體越長代表力道越強。"
                   "柱狀體在縮短，代表趨勢在減弱。")
        st.markdown("---")

    # ---------- 量價 ----------
    if show_vol:
        st.write("## 📊 量價關係")
        a, b, c = st.columns(3)
        a.metric("近5期漲跌", f"{chg5:+.2f}%")
        b.metric("成交量", f"平常的 {vr5:.2f} 倍")
        c.metric("判定", pv[0])

        if pv[1] >= 2:
            st.success(f"### 🟢 {pv[0]}")
        elif pv[1] <= -2:
            st.error(f"### 🔴 {pv[0]}")
        elif pv[1] < 0:
            st.warning(f"### 🟡 {pv[0]}")
        else:
            st.info(f"### ▫️ {pv[0]}")
        st.write(f"**{pv[2]}**")

        v5 = float(d["Volume"].tail(5).mean())
        v20 = float(d["Volume"].tail(20).mean())
        v60 = float(d["Volume"].tail(60).mean()) if len(d) > 60 else v20
        unit = 1000 if is_tw(sym) else 1
        ul = "張" if unit == 1000 else "股"
        st.write(f"近5期均量 **{v5/unit:,.0f} {ul}**｜"
                 f"近20期 {v20/unit:,.0f} {ul}｜近60期 {v60/unit:,.0f} {ul}")
        if v5 > v20 > v60:
            st.success("📈 量能持續放大 — 越來越多人在交易，市場注意力增加。")
        elif v5 < v20 < v60:
            st.warning("📉 量能持續萎縮 — 越來越少人交易，行情容易走不動。")

        st.markdown("---")

    # ---------- 趨勢結構 ----------
    if show_struct:
        st.write("## 📐 趨勢結構（頭頭高底底高）")

        if st_state == "多頭":
            st.success("### 📈 多頭趨勢 — 頭頭高、底底高")
            st.write("下一個高點比上一個高，下一個低點也比上一個高。")
        elif st_state == "空頭":
            st.error("### 📉 空頭趨勢 — 頭頭低、底底低")
            st.write("下一個高點比上一個低，下一個低點也更低。")
        elif st_state == "盤整":
            st.warning("### ↔️ 盤整 — 頭跟底沒有明顯方向")
        else:
            st.info("### ❓ 波段高低點太少，抓不出結構")

        if st_det:
            a, b = st.columns(2)
            with a:
                st.write("**頭（波段高點）**")
                st.write("　" + "　→　".join(f"{x:,.1f}" for x in st_det["頭序列"]))
                st.write("　✅ **頭頭高**" if st_det["頭頭高"] else
                         ("　❌ **頭頭低**" if st_det["頭頭低"] else "　▫️ 沒一致方向"))
            with b:
                st.write("**底（波段低點）**")
                st.write("　" + "　→　".join(f"{x:,.1f}" for x in st_det["底序列"]))
                st.write("　✅ **底底高**" if st_det["底底高"] else
                         ("　❌ **底底低**" if st_det["底底低"] else "　▫️ 沒一致方向"))

        # 續抱出場
        if len(d) >= 3:
            low_y = float(d["Low"].iloc[-2]); low_t = float(d["Low"].iloc[-1])
            high_y = float(d["High"].iloc[-2]); high_t = float(d["High"].iloc[-1])
            st.write("")
            st.write("**今天該續抱還是出場**")
            if st_state == "多頭":
                if low_t < low_y:
                    st.error(f"⚠️ **跌破前一期最低 → 賣出**"
                             f"（前 {low_y:,.2f}，今 {low_t:,.2f}）")
                else:
                    st.success(f"✅ **沒跌破前一期最低 → 續抱**"
                               f"（前 {low_y:,.2f}，今 {low_t:,.2f}）")
                    st.write(f"　下一期要盯的價位：**{low_y:,.2f}**")
            elif st_state == "空頭":
                if high_t > high_y:
                    st.warning(f"突破前一期最高 → 空頭可能結束，留意轉多"
                               f"（前 {high_y:,.2f}，今 {high_t:,.2f}）")
                else:
                    st.error(f"沒突破前一期最高 → 空頭續行"
                             f"（前 {high_y:,.2f}，今 {high_t:,.2f}）")
            else:
                st.info("盤整時「前高前低」訊號很雜，容易來回被巴。等突破或跌破再說。")

        st.caption("這是道氏理論的趨勢判斷，流傳超過百年。"
                   "注意：一個「頭」要等右邊幾根都比它低才能確認，所以標記會比實際晚出現。")
        st.markdown("---")

    # ---------- K線型態 ----------
    if show_candle:
        st.write("## 🕯️ 最近的 K 線型態")
        found = False
        for k in range(1, 6):
            i = len(d) - k
            if i < 1:
                break
            sigs = candle_signals(d, i)
            if not sigs:
                continue
            found = True
            st.write(f"**{d.index[i].strftime('%Y/%m/%d')}**　"
                     + "、".join(s[0] for s in sigs))
            for nm, det_ in sigs:
                st.write(f"　• {nm}：{det_}")
            names = {s[0] for s in sigs}
            above = float(d["Close"].iloc[i]) > float(d["MA20"].iloc[i])
            if "長上影線" in names and "爆大量" in names and above:
                st.error("　→ **高檔長上影線＋爆量：轉弱訊號。**"
                         "衝高有人大量倒貨。下一期下跌就要考慮賣出。")
            elif "長黑K" in names and "爆大量" in names:
                st.error("　→ **長黑＋爆量：有人在大量出貨。**"
                         "盤整末端出現，通常是要往下走。")
            elif "長紅K" in names and "爆大量" in names:
                st.success("　→ **長紅＋爆量：有人在大量買進。**"
                           "如果是突破盤整，是多頭確認訊號。")
            elif "長下影線" in names and "爆大量" in names:
                st.info("　→ **長下影＋爆量：低檔有人在接。**可能止跌，但要等確認。")
            st.write("")
        if not found:
            st.write("最近 5 期沒有明顯型態。")
        st.markdown("---")

    # ---------- 法人 ----------
    if show_inst:
        st.write("## 🏦 法人買賣超")
        if not is_tw(sym) or sym.startswith("^"):
            st.info("只有台股個股有這項公開資料。港股、美股沒有對應來源。")
        elif inst is None or len(inst) < 2:
            st.warning("抓不到法人資料。可能是上櫃股（證交所API只涵蓋上市）、"
                       "ETF，或今天還沒公布（通常下午3點後）。")
        else:
            fs = streak_of(inst["外資"].tolist())
            iv = streak_of(inst["投信"].tolist())
            f5 = inst["外資"].tail(5).sum()
            i5 = inst["投信"].tail(5).sum()

            a, b, c = st.columns(3)
            a.metric("外資", f"{'連買' if fs>0 else '連賣'} {abs(fs)} 天",
                     f"近5日 {f5:+,.0f} 張")
            b.metric("投信", f"{'連買' if iv>0 else '連賣'} {abs(iv)} 天",
                     f"近5日 {i5:+,.0f} 張")
            c.metric("資料天數", f"{len(inst)} 天")

            if fs >= 3 and iv >= 3:
                st.success("✅ **外資投信同步連買** — 籌碼面偏正向。")
            elif fs <= -3 and iv <= -3:
                st.error("❌ **外資投信同步連賣** — 籌碼面偏負向。")
            elif fs >= 3:
                st.info("外資持續買進。")
            elif iv >= 3:
                st.info("投信持續買進。投信通常做中期，連買代表看好一段時間。")

            fg = go.Figure()
            fg.add_trace(go.Bar(x=inst["日期"], y=inst["外資"], name="外資"))
            fg.add_trace(go.Bar(x=inst["日期"], y=inst["投信"], name="投信"))
            fg.add_hline(y=0, line_color="black")
            fg.update_layout(height=280, barmode="group",
                             yaxis_title="買賣超（張）", margin=dict(t=20, b=20))
            st.plotly_chart(fg, use_container_width=True)
            st.caption("柱子在0以上=買超、以下=賣超。資料來自證交所，落後一個交易日。")
        st.markdown("---")

    # ---------- 大戶 ----------
    if show_big:
        st.write("## 🐋 大戶持股（集保）")
        if not is_tw(sym) or sym.startswith("^"):
            st.info("只有台股有集保公開資料。")
        else:
            try:
                with st.spinner("連線集保中心（較慢，約20-40秒）..."):
                    sess = tdcc_session()
                    recs = []
                    for dt in sess["dates"][:12]:
                        t = tdcc_week(code, dt, sess["token"], sess["uri"],
                                      sess["cookies"])
                        if t is None:
                            continue
                        r = parse_tdcc(t)
                        if r:
                            r["日期"] = f"{dt[:4]}/{dt[4:6]}/{dt[6:]}"
                            recs.append(r)
                if not recs:
                    st.warning("抓不到這檔的集保資料。可能代號不在資料裡，或網站改版。")
                else:
                    bd = pd.DataFrame(recs).iloc[::-1].reset_index(drop=True)
                    now_ = bd.iloc[-1]; first = bd.iloc[0]
                    chg = now_["大戶比例"] - first["大戶比例"]

                    a, b, c = st.columns(3)
                    a.metric("大戶持股比例", f"{now_['大戶比例']:.2f}%",
                             f"{chg:+.2f}% vs {len(bd)}週前")
                    if not np.isnan(now_["散戶比例"]):
                        b.metric("散戶持股比例", f"{now_['散戶比例']:.2f}%")
                    if now_.get("大戶人數"):
                        c.metric("大戶人數", f"{now_['大戶人數']:,} 人")

                    if chg > 1:
                        st.success("### 🐋 大戶在買")
                        st.write("大戶手上的股票變多了。")
                        if now_.get("大戶人數") and first.get("大戶人數"):
                            dp = now_["大戶人數"] - first["大戶人數"]
                            if dp < 0:
                                st.write(f"人數少了 {abs(dp)} 人但比例增加 → "
                                         f"**少數人在集中買進，籌碼更集中了。**")
                    elif chg < -1:
                        st.error("### 📉 大戶在賣")
                        st.write("大戶手上的股票變少了，籌碼流向散戶。")
                    else:
                        st.info("### ▫️ 大戶持股穩定，沒有明顯進出")

                    fg = go.Figure()
                    fg.add_trace(go.Scatter(x=bd["日期"], y=bd["大戶比例"],
                                            name="大戶（400張以上）",
                                            line=dict(color="crimson", width=3)))
                    if bd["散戶比例"].notna().any():
                        fg.add_trace(go.Scatter(x=bd["日期"], y=bd["散戶比例"],
                                                name="散戶（10張以下）",
                                                line=dict(color="steelblue", width=2)))
                    fg.update_layout(height=300, yaxis_title="佔總股數 %",
                                     margin=dict(t=20, b=40), hovermode="x unified")
                    st.plotly_chart(fg, use_container_width=True)
                    st.caption("紅線往上=大戶在買。集保每週五結算、至少落後一週。"
                               "「大戶」包含公司派、外資、券商自營，不一定是在做多。")
            except Exception as e:
                st.error(f"集保連線失敗：{type(e).__name__}")
                st.caption("集保網站隨時可能改版。這是實驗功能。")
        st.markdown("---")

    # ---------- 基本面 ----------
    if show_fund:
        st.write("## 🏢 公司基本面")
        name = info.get("longName") or info.get("shortName") or sym
        qtype = (info.get("quoteType") or "").upper()
        st.write(f"**{name}**")

        if qtype in ("ETF", "MUTUALFUND") or sym.startswith("^"):
            st.info("這是指數或 ETF，不是單一公司，沒有財報可看。")
        elif not info:
            st.warning("Yahoo 查不到這檔的財務資料。"
                       "建議自己確認：公司有沒有賺錢？負債重不重？營收成長還是衰退？")
        else:
            pe = info.get("trailingPE"); eps = info.get("trailingEps")
            margin = info.get("profitMargins"); roe = info.get("returnOnEquity")
            d2e = info.get("debtToEquity"); rev = info.get("revenueGrowth")
            cap = info.get("marketCap")

            goods, issues = [], []
            if eps is not None:
                (goods if eps > 0 else issues).append(
                    f"**{'有賺錢' if eps>0 else '在虧錢'}**"
                    f"（每股{'賺' if eps>0 else '虧'} {abs(eps):.2f} {cur}）")
            if margin is not None:
                m = margin * 100
                if m > 10:
                    goods.append(f"**賺得不錯**（每100元營收賺 {m:.0f} 元）")
                elif m > 0:
                    goods.append(f"賺得普通（每100元營收賺 {m:.1f} 元）")
                else:
                    issues.append(f"**做一單賠一單**（利潤率 {m:.1f}%）")
            if pe and pe > 0:
                if pe < 15:
                    goods.append(f"**價格不算貴**（本益比 {pe:.0f} 倍）")
                elif pe < 30:
                    goods.append(f"價格合理偏高（本益比 {pe:.0f} 倍）")
                else:
                    issues.append(f"**價格偏貴**（本益比 {pe:.0f} 倍，"
                                  f"要 {pe:.0f} 年獲利才回本）")
            if d2e is not None:
                if d2e > 200:
                    issues.append(f"**負債偏重**（負債是股東權益的 {d2e/100:.1f} 倍）")
                elif d2e < 100:
                    goods.append(f"**負債健康**（負債是股東權益的 {d2e/100:.1f} 倍）")
            if rev is not None:
                g_ = rev * 100
                if g_ > 10:
                    goods.append(f"**營收在成長**（比去年多 {g_:.0f}%）")
                elif g_ < -10:
                    issues.append(f"**營收在衰退**（比去年少 {abs(g_):.0f}%）")
            if roe is not None:
                r_ = roe * 100
                if r_ > 15:
                    goods.append(f"**股東的錢用得有效率**（ROE {r_:.0f}%）")
                elif r_ < 0:
                    issues.append(f"**在虧股東的錢**（ROE {r_:.0f}%）")

            if cap:
                v = cap
                if cur == "美元":
                    txt = f"{v/1e9:.1f} 十億美元" if v >= 1e9 else f"{v/1e6:.0f} 百萬美元"
                else:
                    txt = f"{v/1e8:.0f} 億{cur}" if v >= 1e8 else f"{v/1e4:.0f} 萬{cur}"
                st.caption(f"總市值：{txt}")

            if goods:
                st.write("**還不錯的地方：**")
                for g_ in goods:
                    st.write(f"　✅ {g_}")
            if issues:
                st.write("**要注意的地方：**")
                for x in issues:
                    st.write(f"　⚠️ {x}")
            if not issues:
                st.success("✅ 體質沒看到明顯問題")
            elif len(issues) >= 3:
                st.error(f"⛔ 有 {len(issues)} 個地方要注意。線圖再好看，"
                         f"公司本身有狀況還是有風險。")
            st.caption("財務數字來自 Yahoo，可能落後或不準。重要決定請自己再查一次。")
        st.markdown("---")

    # ---------- 大盤 ----------
    if show_mkt:
        st.write(f"## 🌏 {mkt}大盤環境")
        if dfi is None:
            st.warning("查不到大盤資料。")
        else:
            di = add_ind(dfi)
            ip = float(di["Close"].iloc[-1]); im = float(di["MA60"].iloc[-1])
            it, isl = trend_of(di)
            a, b = st.columns(2)
            a.metric("大盤指數", f"{ip:,.0f}", f"{(ip/im-1)*100:+.1f}% vs 均價")
            b.metric("60日均價", f"{im:,.0f}")

            if ip > im and it == "往上":
                st.success("### ✅ 大盤在漲")
                st.write("大環境是順的。個股表現有大盤幫忙。")
            elif ip < im:
                st.error("### ⛔ 大盤跌破均價")
                st.write("**大盤在跌的時候，八成的個股都會跟著跌。**"
                         "這時候買個股，等於逆著水流游泳。")
            else:
                st.warning("### ⚠️ 大盤在整理")
                st.write("方向不明，訊號容易忽真忽假。")

            show = di.tail(180)
            f = go.Figure()
            f.add_trace(go.Scatter(x=show.index, y=show["Close"], name="指數",
                                   line=dict(color="black", width=1.5)))
            f.add_trace(go.Scatter(x=show.index, y=show["MA60"], name="60日均",
                                   line=dict(color="red", width=2)))
            f.update_layout(height=260, margin=dict(t=20, b=20),
                            hovermode="x unified")
            st.plotly_chart(f, use_container_width=True)

    st.markdown("---")
    st.warning("以上都是照規則機械式計算的結果，不是買賣建議。"
               "要不要進出請自己判斷，並嚴守停損。")


# ============================================================
# 分頁二：驗證回測
# ============================================================
with tab2:
    st.write("### 這套規則到底有沒有用")
    st.caption("拿過去真實資料跑一遍。訊號隔期開盤才成交（不偷看未來），"
               "扣除完整成本。")

    c1, c2, c3 = st.columns(3)
    with c1:
        bsym = st.text_input("股票代號（逗號分開，最多5檔）",
                             "2330.TW, 2317.TW, 0050.TW", key="bs")
    with c2:
        bstop = st.slider("認賠出場 %", 5, 30, 15, key="bst")
    with c3:
        bcost = st.slider("單趟成本 %", 0.0, 1.0, 0.35, 0.05, key="bc")

    st.caption("💡 台股一買一賣約 0.585%，加滑價，單趟抓 0.35% 比較保守。")

    if st.button("開始回測", use_container_width=True, type="primary", key="bb"):
        syms = [fix_symbol(x) for x in bsym.split(",") if x.strip()][:5]
        rows = []
        bar = st.progress(0.0)

        for n, s in enumerate(syms):
            df = get_data(s, "1d", "10y")
            bar.progress((n + 1) / len(syms), text=f"回測 {s}")
            if df is None:
                continue
            dd_ = add_ind(df)
            H, L = find_pivots(dd_, 5)
            N = len(dd_)
            struct = np.array(["未知"] * N, dtype=object)
            for i in range(N):
                s_, _, _ = trend_structure(H, L, upto=i)
                struct[i] = s_ if s_ != "資料不足" else "未知"

            cl = dd_["Close"].values; op = dd_["Open"].values
            ma = dd_["MA60"].values; K = dd_["K"].values
            MC = dd_["MACD"].values; SG = dd_["SIG"].values
            VR = dd_["VR"].values

            def bt(mode):
                rets = []
                pos = False; ep = 0.0
                for i in range(63, N - 1):
                    if np.isnan(ma[i]) or np.isnan(K[i]) or np.isnan(K[i-1]):
                        continue
                    base = (cl[i] > ma[i] and K[i] > 50 and K[i] > K[i-1]
                            and MC[i] > SG[i])
                    if mode == "加趨勢結構":
                        ok = base and struct[i] == "多頭"
                    elif mode == "加量能放大":
                        ok = base and (not np.isnan(VR[i])) and VR[i] > 1.2
                    elif mode == "全部都要":
                        ok = (base and struct[i] == "多頭"
                              and (not np.isnan(VR[i])) and VR[i] > 1.2)
                    else:
                        ok = base
                    if not pos:
                        if ok:
                            pos = True; ep = op[i+1]
                    else:
                        r = (cl[i] - ep) / ep * 100
                        if r <= -bstop or cl[i] < ma[i]:
                            rets.append((op[i+1]-ep)/ep*100 - bcost*2)
                            pos = False
                if len(rets) < 3:
                    return None
                a = np.array(rets)
                eq = 100.0
                for x in a:
                    eq *= (1 + x/100)
                sd = a.std(ddof=1) if len(a) > 1 else 0
                tv = a.mean()/(sd/np.sqrt(len(a))) if sd > 0 else 0
                return {"筆數": len(a),
                        "勝率%": round(len(a[a>0])/len(a)*100, 1),
                        "每筆平均%": round(a.mean(), 2),
                        "t值": round(tv, 2),
                        "總報酬%": round(eq-100, 1)}

            bh = (cl[-1]-cl[63])/cl[63]*100
            for mode in ["原本三條", "加趨勢結構", "加量能放大", "全部都要"]:
                r = bt(mode)
                rows.append({"股票": s, "方式": mode,
                             "筆數": r["筆數"] if r else 0,
                             "勝率%": r["勝率%"] if r else None,
                             "每筆平均%": r["每筆平均%"] if r else None,
                             "t值": r["t值"] if r else None,
                             "總報酬%": r["總報酬%"] if r else None,
                             "買著不動%": round(bh, 1)})
        bar.empty()

        if not rows:
            st.error("沒抓到資料")
        else:
            res = pd.DataFrame(rows)

            def colf(v):
                try:
                    return "color: green" if v > 0 else "color: red"
                except Exception:
                    return ""

            st.markdown("---")
            st.write("### 明細")
            st.dataframe(res.style.map(colf,
                subset=["每筆平均%", "總報酬%", "買著不動%"]),
                use_container_width=True, hide_index=True)

            st.write("### 四種方式平均比較")
            g = res.dropna(subset=["每筆平均%"]).groupby("方式").agg(
                平均每筆=("每筆平均%", "mean"),
                平均筆數=("筆數", "mean"),
                平均勝率=("勝率%", "mean"),
                賺錢檔數=("每筆平均%", lambda x: (x > 0).sum()),
            ).round(2)
            st.dataframe(g, use_container_width=True)

            st.markdown("---")
            st.write("### 白話結論")
            if not g.empty:
                base = g.loc["原本三條", "平均每筆"] if "原本三條" in g.index else None
                bn = g["平均每筆"].idxmax(); bv = g["平均每筆"].max()
                st.write(f"**最好的是「{bn}」**，平均每筆 {bv:+.2f}%")
                if base is not None:
                    st.write(f"原本三條規則是 {base:+.2f}%")
                    if bn == "原本三條":
                        st.warning("**加條件沒有變好。**　多加的過濾反而讓訊號變少、"
                                   "報酬變差。建議維持簡單。")
                    else:
                        st.success(f"**加條件有幫助**，每筆多 {bv-base:.2f} 個百分點。")
                        if g.loc[bn, "平均筆數"] < 10:
                            st.warning(f"⚠️ 但平均只做 {g.loc[bn,'平均筆數']:.0f} 次，"
                                       f"樣本太少，結果不可靠。")

                vb = res.dropna(subset=["每筆平均%"])
                beat = (vb["總報酬%"] > vb["買著不動%"]).sum()
                st.write("")
                if beat == 0:
                    st.error(f"**沒有任何一種贏過「買了放著不動」。**　"
                             f"進進出出只是白忙，還多付手續費。")
                else:
                    st.info(f"{beat}/{len(vb)} 種贏過「買了放著不動」。")

                hi_t = (vb["t值"] > 2).sum()
                if hi_t == 0:
                    st.warning("**沒有一組 t值 > 2。**　"
                               "統計上無法排除「只是運氣好」的可能。")


# ============================================================
# 分頁三：說明
# ============================================================
with tab3:
    st.write("# 名詞白話解釋")

    st.write("## 均線")
    st.write("""
    把過去 N 期的收盤價平均起來畫成線。60日均線就是過去60天的平均價。

    **看它做什麼：** 價格在均線上，代表最近買的人大多是賺的，比較不會急著賣。
    跌破均線就是你的出場訊號。

    **完美多頭排列** = 現價 > 5期 > 10期 > 20期 > 60期，一層層往上疊。
    這代表不管什麼時候買的人現在都在賺。
    """)

    st.write("## KD")
    st.write("""
    有兩條線：**K線（快）** 和 **D線（慢）**，都是 0~100 分。

    - **K往上穿過D = 黃金交叉**，一般視為轉強
    - **K往下穿過D = 死亡交叉**，一般視為轉弱
    - **K > 80 = 超買**，已經漲一段，這時候買是追高
    - **K < 20 = 超賣**，跌深了，但不代表馬上會漲

    你的666戰法用的是「**K值站上50且上升**」，不是黃金交叉。
    畫面上的交叉資訊只是給你參考。
    """)

    st.write("## RSI")
    st.write("""
    「最近漲的力道 vs 跌的力道」的比例，0~100。

    - **70以上** = 漲得又快又多，偏熱
    - **50上下** = 多空平衡
    - **30以下** = 跌得又快又多

    注意：**超買不代表會跌，超賣也不代表會漲**。強勢股可以在超買區待很久。
    """)

    st.write("## MACD")
    st.write("""
    看「動能」的指標，兩條線加一排柱子。

    - **藍線穿過橘線往上** = 翻多（黃金交叉）
    - **藍線穿過橘線往下** = 翻空（死亡交叉）
    - **柱狀體越長** = 力道越強
    - **柱狀體在縮短** = 就算還在漲，力道一次比一次弱，通常快停了

    兩條線都在 0 以上 = 中期偏多，都在 0 以下 = 中期偏空。
    """)

    st.write("## 量價關係")
    st.write("""
    | 組合 | 意思 |
    |---|---|
    | 🟢 **價漲量增** | 有人真的拿錢在買，健康 |
    | 🟡 價漲量縮 | 少數單推上去的，撐不久 |
    | 🔴 **價跌量增** | 有人在倒貨，最危險 |
    | 🟡 價跌量縮 | 賣壓在減輕，可能快跌完 |
    | 🔵 價平量增 | 有人在換手，光看量價分不出好壞 |
    | ⚪ 價平量縮 | 沒人理它 |
    """)

    st.write("## 趨勢結構（頭頭高底底高）")
    st.write("""
    **道氏理論**，流傳一百多年的概念。

    - **多頭** = 頭頭高、底底高（下一個高點比上一個高，低點也比上一個高）
    - **空頭** = 頭頭低、底底低
    - **盤整** = 頭跟底沒有明顯方向

    **續抱判斷：** 多頭時只要沒跌破前一期最低就抱著，跌破就走。
    空頭時反過來看前一期最高。

    ⚠️ 一個「頭」要等右邊幾根都比它低才能確認，所以**訊號一定比實際晚出現**。
    這是這套方法的天生限制。
    """)

    st.write("## 法人與大戶")
    st.write("""
    **外資／投信／自營商** = 三大法人。他們資金大，動向有參考價值。
    投信通常做中期，連買代表看好一段時間。資料來自證交所，落後一個交易日。

    **大戶** = 持股 400 張以上的帳戶。集保每週公布一次，至少落後一週。
    注意：大戶裡面包含公司派、外資、券商自營，**不一定是在做多**。

    「人數變少但比例增加」= 少數人在集中買進，籌碼更集中了。
    """)

    st.write("## 基本面")
    st.write("""
    線圖不會告訴你這間公司有沒有賺錢。一間虧損、負債重的公司，
    線圖再漂亮也可能突然出事。

    - **每股盈餘（EPS）** — 一股賺多少錢，負的就是在虧
    - **本益比（PE）** — 幾年的獲利才回本。數字越大越貴
    - **利潤率** — 每 100 元營收能賺多少
    - **負債比** — 太高代表財務壓力大
    - **ROE** — 股東的錢用得有沒有效率
    """)

    st.markdown("---")
    st.write("# ⚠️ 重要提醒")
    st.error("""
    **1. 只有「三條規則」被驗證過，其他都沒有。**
    大盤過濾、趨勢結構、量價、籌碼、基本面 —— 這些是常識性的輔助判斷，
    不是從你的資料跑出來的結論。加了不保證變好。

    **2. 回測顯示，多數情況輸給「買了放著不動」。**
    台積電：規則 174% vs 放著不動 297%。鴻海：72% vs 141%。

    **3. 抱久才賺，短線都在賠。**
    42 筆交易裡，抱不到 10 天的沒有一筆是賺的。

    **4. 這套系統最多放你資金的 10-20%。**
    回測裡曾連續賠 14 次、資金縮水 48%。全押的話很難撐過去。
    """)

st.markdown("---")
st.caption("資料來源：Yahoo Finance、臺灣證券交易所、臺灣集中保管結算所。研究工具，不是投資建議。回測已扣手續費估算但未計滑價與流動性限制。過去表現不保證未來。")
