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


def describe_line(series, label, unit="", digits=2, n=None):
    """把一條線用文字描述出來，讓不看圖的人也懂"""
    s = series.dropna()
    if len(s) < 2:
        return f"　**{label}：** 資料不足"
    v0, v1 = float(s.iloc[0]), float(s.iloc[-1])
    hi, lo = float(s.max()), float(s.min())
    d_ = v1 - v0
    pct = (v1 / v0 - 1) * 100 if v0 != 0 else 0
    rng = hi - lo
    if rng > 0 and abs(d_) < rng * 0.15:
        shape = "上上下下，沒有明顯方向"
    elif d_ > 0:
        shape = "整體往上"
    else:
        shape = "整體往下"
    span = f"這 {len(s)} 期" if n is None else n
    return (f"　**{label}：** {shape}。{span}從 {v0:,.{digits}f}{unit} "
            f"到 {v1:,.{digits}f}{unit}（{pct:+.1f}%），"
            f"最高 {hi:,.{digits}f}{unit}、最低 {lo:,.{digits}f}{unit}。")


def describe_cross(fast, slow, fname, sname):
    """描述兩條線的交叉狀況"""
    f = fast.dropna(); s = slow.dropna()
    if len(f) < 3 or len(s) < 3:
        return ""
    n = min(len(f), len(s), 60)
    f = f.iloc[-n:]; s = s.iloc[-n:]
    crosses = []
    for i in range(1, len(f)):
        a0, a1 = float(f.iloc[i-1]), float(f.iloc[i])
        b0, b1 = float(s.iloc[i-1]), float(s.iloc[i])
        if a0 <= b0 and a1 > b1:
            crosses.append((f.index[i], "黃金交叉"))
        elif a0 >= b0 and a1 < b1:
            crosses.append((f.index[i], "死亡交叉"))
    if not crosses:
        pos = "上面" if float(f.iloc[-1]) > float(s.iloc[-1]) else "下面"
        return f"　最近 {n} 期沒有交叉，{fname}一直在{sname}{pos}。"
    last = crosses[-1]
    ago = len(f) - list(f.index).index(last[0]) - 1
    txt = (f"　最近 {n} 期共發生 {len(crosses)} 次交叉，"
           f"最後一次是 **{last[1]}**，在 {last[0].strftime('%Y/%m/%d')}"
           f"（{ago} 期前）。")
    return txt


def find_crosses(fast, slow):
    """找出所有黃金/死亡交叉的位置"""
    f = fast.values
    s = slow.values
    out = []
    for i in range(1, len(f)):
        if np.isnan(f[i]) or np.isnan(s[i]) or np.isnan(f[i-1]) or np.isnan(s[i-1]):
            continue
        if f[i-1] <= s[i-1] and f[i] > s[i]:
            out.append((i, "黃金交叉"))
        elif f[i-1] >= s[i-1] and f[i] < s[i]:
            out.append((i, "死亡交叉"))
    return out


def cross_accuracy(df, fast_col, slow_col, horizons=(5, 10, 20)):
    """
    統計歷史上每次交叉之後，過 N 期股價漲跌如何。
    這是回答「這個訊號準不準」的直接證據。
    """
    cl = df["Close"].values
    N = len(cl)
    crosses = find_crosses(df[fast_col], df[slow_col])
    res = {"黃金交叉": {}, "死亡交叉": {}}
    counts = {"黃金交叉": 0, "死亡交叉": 0}

    for kind in ["黃金交叉", "死亡交叉"]:
        pts = [i for i, k in crosses if k == kind]
        counts[kind] = len(pts)
        for h in horizons:
            rets = [(cl[i+h] - cl[i]) / cl[i] * 100
                    for i in pts if i + h < N]
            if len(rets) >= 3:
                a = np.array(rets)
                res[kind][h] = {
                    "次數": len(a),
                    "平均%": round(float(a.mean()), 2),
                    "上漲比例%": round(float((a > 0).mean() * 100), 1),
                    "最好%": round(float(a.max()), 1),
                    "最差%": round(float(a.min()), 1),
                }
    return res, counts, crosses


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
    vip = df[df["_lv"] == 15]
    body = df[(df["_lv"] >= 1) & (df["_lv"] <= 15)]
    if big.empty:
        return None

    big_pct = float(big["_pct"].sum())
    big_ppl = int(big["_ppl"].sum()) if big["_ppl"].notna().any() else None
    all_ppl = int(body["_ppl"].sum()) if body["_ppl"].notna().any() else None

    return {
        "大戶比例": big_pct,
        "散戶比例": float(sml["_pct"].sum()) if not sml.empty else np.nan,
        "千張比例": float(vip["_pct"].sum()) if not vip.empty else np.nan,
        "大戶人數": big_ppl,
        "總股東人數": all_ppl,
        # 平均每個大戶握有多少比例 → 越大代表籌碼越集中在少數人手上
        "每戶平均": (big_pct / big_ppl * 1000) if (big_ppl and big_ppl > 0) else np.nan,
    }


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
show_night = st.sidebar.checkbox("🌙 夜盤／隔日開盤預期", True)
show_macro = st.sidebar.checkbox("🌐 總體環境（油價/殖利率/匯率）", True)
st.sidebar.caption("💡 原油ETF 和 熱門類股 在上面的分頁，不用勾選")

st.sidebar.markdown("---")
st.sidebar.write("**持股（選填）**")
my_cost = st.sidebar.number_input("你的買進價", 0.0, step=1.0, format="%.2f")
my_qty = st.sidebar.number_input("張數／股數", 0.0, step=1.0)
my_stop = st.sidebar.slider("認賠出場 %", 5, 30, 15)

run = st.sidebar.button("開始分析", use_container_width=True, type="primary")

st.title("📊 股票分析系統")

tab1, tab_oil, tab_hot, tab2, tab3 = st.tabs(
    ["📍 個股分析", "🛢️ 原油ETF", "🔥 熱門類股", "🔬 驗證回測", "📖 說明"])


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

        # 一句話總結（不看任何圖也能懂）
        parts = []
        parts.append(f"這檔的大方向是**{trend}**")
        parts.append(f"趨勢結構是**{st_state}**")
        parts.append(f"你的三條規則過了 **{hit}/3** 條")
        parts.append(f"量價是**{pv[0]}**")
        if inst is not None and len(inst) >= 3:
            _fs = streak_of(inst["外資"].tolist())
            parts.append(f"外資**{'連買' if _fs>0 else '連賣'}{abs(_fs)}天**")
        one_line = "，".join(parts) + "。"

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

        st.write(f"**一句話：** {one_line}")
        st.write(f"**綜合分數 {score} 分**（{tf_name}）")

        with st.expander("📖 這個分數怎麼算出來的？"):
            st.write("""
            把下面幾項加減分，加總起來就是綜合分數：

            | 項目 | 加分 | 扣分 |
            |---|---|---|
            | 大方向（長期均線斜率） | 往上 +3 | 往下 −3 |
            | 均線排列 | 完美多頭 +2／站上長均 +1 | 跌破長均 −2 |
            | KD | K>50且上升 +2 | K<50且下降 −2／K>80 −1 |
            | MACD | 剛翻多 +2／維持多方 +1 | 在空方 −1 |
            | RSI | 50~75 +1 | 80以上 −1 |
            | 量價 | 價漲量增 +2 | 價跌量增 −3／價漲量縮 −1 |
            | 趨勢結構 | 頭頭高底底高 +2 | 頭頭低底底低 −2 |
            | 法人 | 外資投信同步連買 +2 | 外資連賣3天 −1 |
            | 大盤 | 在均價上 +1 | 跌破均價 −2 |

            **分數對照：**
            9分以上=條件很好　5~8分=條件不錯　2~4分=再等等　1分以下=別進

            **但如果大方向是「往下」，不管幾分都建議避開** —— 
            你的回測已經證明，股票本身在跌的時候照這套規則做會賠更慘。
            """)

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
                st.success(f"目前賺 {pnl:.1f}%。"
                           f"**你的規則沒有獲利了結的條件** —— "
                           f"出場只看「跌破均線」或「到停損」。"
                           f"只要這兩個都沒發生，照規則就是續抱。")

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

        st.write("**這張圖在說什麼（文字版）**")
        st.write(describe_line(show["Close"], "股價", digits=2))
        st.write(describe_line(show["MA60"], f"{ma_long}期均線（最粗那條）", digits=2))
        above = int((show["Close"] > show["MA60"]).sum())
        st.write(f"　這 {len(show)} 期裡，股價有 **{above} 期站在"
                 f"{ma_long}期均線之上**、{len(show)-above} 期在下面。")
        if above > len(show) * 0.7:
            st.write("　→ 大部分時間都在均線上，這是**多頭格局**。")
        elif above < len(show) * 0.3:
            st.write("　→ 大部分時間都在均線下，這是**空頭格局**。")
        else:
            st.write("　→ 上上下下穿來穿去，是**盤整格局**。這種時候訊號最容易失準。")

        st.write("")
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
        st.write("## 📊 KD 指標（黃金交叉 / 死亡交叉）")

        # ===== 兩組參數同時判斷 =====
        st.write("### 兩組參數的交叉狀況")

        rows_cross = []
        for label, kc, dc in [("標準 (9,3,3)", "K9", "D9"),
                              ("你的設定 (60,3,3)", "K", "D")]:
            kk = float(d[kc].iloc[-1]); kkp = float(d[kc].iloc[-2])
            dd_ = float(d[dc].iloc[-1]); ddp = float(d[dc].iloc[-2])
            gold = (kkp <= ddp) and (kk > dd_)
            dead = (kkp >= ddp) and (kk < dd_)

            # 找最近一次交叉是幾期前
            cr = find_crosses(d[kc], d[dc])
            if cr:
                li, lk = cr[-1]
                ago = len(d) - 1 - li
                last_txt = f"{lk}（{ago} 期前）"
            else:
                last_txt = "沒有紀錄"

            if gold:
                now_txt = "🟡 今天黃金交叉"
            elif dead:
                now_txt = "⚫ 今天死亡交叉"
            else:
                now_txt = ("K在D上（偏強）" if kk > dd_ else "K在D下（偏弱）")

            zone = ("超買" if kk > 80 else ("超賣" if kk < 20 else "中間"))

            rows_cross.append({
                "參數": label, "K值": round(kk, 1), "D值": round(dd_, 1),
                "K−D": round(kk - dd_, 1), "位置": zone,
                "目前": now_txt, "最近一次交叉": last_txt,
            })

        st.dataframe(pd.DataFrame(rows_cross), use_container_width=True,
                     hide_index=True)

        # ===== 兩組是否一致 =====
        k9 = float(d["K9"].iloc[-1]); d9 = float(d["D9"].iloc[-1])
        k60 = float(d["K"].iloc[-1]); d60 = float(d["D"].iloc[-1])
        s9 = k9 > d9
        s60 = k60 > d60
        st.write("")
        if s9 and s60:
            st.success("### ✅ 兩組參數都是「K在D上面」— 訊號一致偏強")
            st.write("短期（9）和中期（60）看法相同，可信度比較高。")
        elif (not s9) and (not s60):
            st.error("### ❌ 兩組參數都是「K在D下面」— 訊號一致偏弱")
            st.write("短期和中期看法相同，都不好。")
        elif s9 and not s60:
            st.warning("### ⚠️ 短期轉強，但中期還沒跟上")
            st.write("標準KD（9）已經翻多，但你的KD（60）還在弱勢。"
                     "這通常是**反彈**，不是趨勢反轉。要等中期也翻多才比較可靠。")
        else:
            st.warning("### ⚠️ 中期還強，但短期轉弱了")
            st.write("你的KD（60）還在多方，但標準KD（9）已經轉弱。"
                     "這通常是**中場休息或見頂前兆**，要盯緊。")

        # ===== 這檔的交叉準不準（實際統計）=====
        st.markdown("---")
        st.write("### 🔬 這檔的交叉訊號準不準？（歷史統計）")
        st.caption("拿這檔過去所有的交叉，統計交叉之後 5／10／20 期股價的實際表現。"
                   "這是回答「準不準」最直接的證據。")

        acc_pick = st.radio("要看哪組參數的統計",
                            ["標準 (9,3,3)", "你的設定 (60,3,3)"],
                            horizontal=True, key="accp")
        kc2, dc2 = ("K9", "D9") if "9,3,3" in acc_pick else ("K", "D")

        acc, cnt, crosses = cross_accuracy(d, kc2, dc2)

        st.write(f"這 {len(d)} 期資料裡，共出現 "
                 f"**{cnt['黃金交叉']} 次黃金交叉**、"
                 f"**{cnt['死亡交叉']} 次死亡交叉**。")

        for kind, icon in [("黃金交叉", "🟡"), ("死亡交叉", "⚫")]:
            if not acc[kind]:
                st.write(f"　{icon} **{kind}**：次數太少（不到3次），無法統計。")
                continue
            st.write("")
            st.write(f"**{icon} {kind}之後的表現**")
            trows = []
            for h, v in acc[kind].items():
                trows.append({
                    "過幾期": f"{h} 期後",
                    "統計次數": v["次數"],
                    "平均漲跌%": v["平均%"],
                    "上漲比例%": v["上漲比例%"],
                    "最好%": v["最好%"],
                    "最差%": v["最差%"],
                })
            st.dataframe(pd.DataFrame(trows), use_container_width=True,
                         hide_index=True)

            # 自動判讀
            v10 = acc[kind].get(10)
            if v10:
                rate = v10["上漲比例%"]
                avg = v10["平均%"]
                if kind == "黃金交叉":
                    if rate >= 60 and avg > 0:
                        st.success(f"　→ **這檔的黃金交叉還算可靠。**"
                                   f"10期後有 {rate:.0f}% 的機率是漲的，"
                                   f"平均漲 {avg:.2f}%。")
                    elif rate >= 50:
                        st.info(f"　→ **勉強及格。**10期後 {rate:.0f}% 機率上漲，"
                                f"平均 {avg:+.2f}%。跟丟銅板差不多。")
                    else:
                        st.error(f"　→ **這檔的黃金交叉不準。**"
                                 f"10期後只有 {rate:.0f}% 機率上漲，"
                                 f"平均 {avg:+.2f}%。照著買反而會賠。")
                else:
                    if rate <= 40 and avg < 0:
                        st.success(f"　→ **這檔的死亡交叉還算可靠。**"
                                   f"10期後只有 {rate:.0f}% 機率上漲，"
                                   f"平均 {avg:.2f}%。出場訊號有效。")
                    elif rate <= 50:
                        st.info(f"　→ **勉強及格。**10期後 {rate:.0f}% 機率上漲。")
                    else:
                        st.error(f"　→ **這檔的死亡交叉不準。**"
                                 f"10期後反而有 {rate:.0f}% 機率上漲，"
                                 f"平均 {avg:+.2f}%。看到死叉就跑會賣在低點。")

        st.warning("**重要：** 上面是「這一檔」的歷史統計，不是通則。"
                   "同一個訊號在不同股票上準確率差很多。"
                   "而且這只統計了交叉後的漲跌，沒有考慮手續費、"
                   "也沒有停損機制，不等於實際交易能賺這麼多。")

        # ===== 超買超賣 =====
        st.markdown("---")
        st.write("### 超買超賣位置")
        kk = float(d[kc2].iloc[-1])
        if kk > 80:
            st.warning(f"K值 {kk:.0f} 在**超買區**（80以上）。已經漲一段了，"
                       f"這時候買是追高。")
        elif kk < 20:
            st.warning(f"K值 {kk:.0f} 在**超賣區**（20以下）。跌深了，"
                       f"但不代表馬上會漲。")
        else:
            st.info(f"K值 {kk:.0f} 在中間區，不算超買也不算超賣。")

        st.write(f"**你的規則要求：K值(60,3,3)>50 且上升** → "
                 f"{'✅ 符合' if r2 else '❌ 不符合'}")
        st.caption("注意：你的666戰法用的是「K值站上50且上升」，"
                   "**不是黃金交叉**。上面的交叉資訊是給你對照參考用的。")

        # ===== 圖 =====
        st.markdown("---")
        show = d.tail({"日線": 180, "週線": 150, "月線": 120}[tf_name])
        f = go.Figure()
        f.add_trace(go.Scatter(x=show.index, y=show[kc2], name="K",
                               line=dict(color="blue", width=2)))
        f.add_trace(go.Scatter(x=show.index, y=show[dc2], name="D",
                               line=dict(color="orange", width=2)))

        # 標出交叉點
        off_ = len(d) - len(show)
        gx = [d.index[i] for i, k in crosses if k == "黃金交叉" and i >= off_]
        gy = [float(d[kc2].iloc[i]) for i, k in crosses
              if k == "黃金交叉" and i >= off_]
        dx = [d.index[i] for i, k in crosses if k == "死亡交叉" and i >= off_]
        dy = [float(d[kc2].iloc[i]) for i, k in crosses
              if k == "死亡交叉" and i >= off_]
        if gx:
            f.add_trace(go.Scatter(x=gx, y=gy, mode="markers", name="黃金交叉",
                marker=dict(symbol="triangle-up", size=12, color="gold",
                            line=dict(width=1, color="black"))))
        if dx:
            f.add_trace(go.Scatter(x=dx, y=dy, mode="markers", name="死亡交叉",
                marker=dict(symbol="triangle-down", size=12, color="black")))

        f.add_hline(y=80, line_dash="dash", line_color="red",
                    annotation_text="超買 80")
        f.add_hline(y=50, line_dash="dot", line_color="gray")
        f.add_hline(y=20, line_dash="dash", line_color="green",
                    annotation_text="超賣 20")
        f.update_layout(height=320, margin=dict(t=20, b=20),
                        hovermode="x unified", yaxis_range=[0, 100],
                        legend=dict(orientation="h", y=1.12))
        st.plotly_chart(f, use_container_width=True)

        st.write("**這張圖在說什麼（文字版）**")
        st.write(describe_line(show[kc2], "K線（藍）", digits=1))
        st.write(describe_line(show[dc2], "D線（橘）", digits=1))
        st.write(f"　圖上標了 **{len(gx)} 個黃金交叉**（金色朝上三角）、"
                 f"**{len(dx)} 個死亡交叉**（黑色朝下三角）。")
        over_b = int((show[kc2] > 80).sum())
        over_s = int((show[kc2] < 20).sum())
        st.write(f"　這 {len(show)} 期裡，K值有 **{over_b} 期在80以上（超買）**、"
                 f"**{over_s} 期在20以下（超賣）**。")
        if over_b > len(show) * 0.3:
            st.write("　→ 長時間停在高檔，代表這是**強勢股**。"
                     "強勢股可以在超買區待很久，看到80就賣會太早下車。")
        elif over_s > len(show) * 0.3:
            st.write("　→ 長時間停在低檔，代表這是**弱勢股**。"
                     "弱勢股可以一路超賣一直跌，看到20就買會接刀子。")

        if crosses:
            st.write("　**最近5次交叉：**")
            for i, k in crosses[-5:][::-1]:
                ago = len(d) - 1 - i
                px = float(d["Close"].iloc[i])
                nowp = float(d["Close"].iloc[-1])
                since = (nowp - px) / px * 100
                icon = "🟡" if k == "黃金交叉" else "⚫"
                st.write(f"　　• {d.index[i].strftime('%Y/%m/%d')}　{icon} {k}"
                         f"（{ago} 期前，當時股價 {px:,.2f}，"
                         f"到現在 {since:+.1f}%）")

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

        st.write("**這張圖在說什麼（文字版）**")
        st.write(describe_line(show["RSI"], "RSI（紫線）", digits=1))
        ob = int((show["RSI"] >= 70).sum())
        os_ = int((show["RSI"] <= 30).sum())
        mid = len(show) - ob - os_
        st.write(f"　這 {len(show)} 期裡，**{ob} 期在70以上（超買）**、"
                 f"**{os_} 期在30以下（超賣）**、{mid} 期在中間。")
        r_recent = float(show["RSI"].tail(10).mean())
        r_before = float(show["RSI"].iloc[-30:-10].mean()) if len(show) > 30 else r_recent
        if r_recent > r_before + 5:
            st.write(f"　→ 最近10期平均 {r_recent:.0f}，比之前的 {r_before:.0f} "
                     f"高，**動能在增強**。")
        elif r_recent < r_before - 5:
            st.write(f"　→ 最近10期平均 {r_recent:.0f}，比之前的 {r_before:.0f} "
                     f"低，**動能在減弱**。")
        else:
            st.write(f"　→ 最近10期平均 {r_recent:.0f}，跟之前差不多，動能持平。")

        st.caption("RSI 是「最近漲的力道 vs 跌的力道」的比例。"
                   "70以上代表漲得又快又多，30以下代表跌得又快又多。"
                   "超買不代表會跌、超賣不代表會漲。")
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

        st.write("**這張圖在說什麼（文字版）**")
        st.write(describe_line(show["MACD"], "MACD（藍線）", digits=3))
        st.write(describe_cross(show["MACD"], show["SIG"], "MACD", "訊號線"))
        pos_h = int((show["HIST"] > 0).sum())
        st.write(f"　柱狀體：這 {len(show)} 期裡有 **{pos_h} 期是紅的（多方）**、"
                 f"{len(show)-pos_h} 期是綠的（空方）。")
        h5 = float(show["HIST"].tail(5).mean())
        h10 = float(show["HIST"].iloc[-15:-5].mean()) if len(show) > 15 else h5
        if abs(h5) < abs(h10) * 0.7:
            st.write("　→ 柱狀體最近明顯縮短，**趨勢力道在減弱**。"
                     "就算價格還在動，通常快要停了。")
        elif abs(h5) > abs(h10) * 1.3:
            st.write("　→ 柱狀體最近明顯變長，**趨勢力道在增強**。")
        else:
            st.write("　→ 柱狀體長度沒什麼變化，力道持平。")

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

            st.write("**這張圖在說什麼（文字版）**")
            f_all = inst["外資"].dropna()
            i_all = inst["投信"].dropna()
            if len(f_all):
                fb = int((f_all > 0).sum()); fsz = len(f_all)
                st.write(f"　**外資：** 這 {fsz} 天裡買超 {fb} 天、賣超 {fsz-fb} 天，"
                         f"合計 {f_all.sum():+,.0f} 張。"
                         f"最大單日買超 {f_all.max():+,.0f} 張、"
                         f"最大單日賣超 {f_all.min():+,.0f} 張。")
            if len(i_all):
                ib = int((i_all > 0).sum()); isz = len(i_all)
                st.write(f"　**投信：** 這 {isz} 天裡買超 {ib} 天、賣超 {isz-ib} 天，"
                         f"合計 {i_all.sum():+,.0f} 張。")
            st.write(f"　**目前狀態：** 外資{'連買' if fs>0 else '連賣'} {abs(fs)} 天，"
                     f"投信{'連買' if iv>0 else '連賣'} {abs(iv)} 天。")
            st.write("　**逐日明細：**")
            for _, rw in inst.tail(5).iloc[::-1].iterrows():
                fo_ = rw["外資"]; in_ = rw["投信"]
                st.write(f"　　• {rw['日期']}　外資 {fo_:+,.0f} 張　"
                         f"投信 {in_:+,.0f} 張")

            st.caption("柱子在0以上=買超、以下=賣超。資料來自證交所，落後一個交易日。")
        st.markdown("---")

    # ---------- 大戶 ----------
    if show_big:
        st.write("## 🐋 大戶持股與籌碼換手")
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
                    now_ = bd.iloc[-1]
                    first = bd.iloc[0]
                    wk = len(bd)

                    d_pct = now_["大戶比例"] - first["大戶比例"]
                    d_ppl = ((now_["大戶人數"] - first["大戶人數"])
                             if (now_.get("大戶人數") and first.get("大戶人數"))
                             else None)
                    d_all = ((now_["總股東人數"] - first["總股東人數"])
                             if (now_.get("總股東人數") and first.get("總股東人數"))
                             else None)
                    d_sml = (now_["散戶比例"] - first["散戶比例"]
                             if not np.isnan(now_["散戶比例"]) else np.nan)

                    a, b, c, e = st.columns(4)
                    a.metric("大戶持股比例", f"{now_['大戶比例']:.2f}%",
                             f"{d_pct:+.2f}% vs {wk}週前")
                    if now_.get("大戶人數"):
                        b.metric("大戶人數", f"{now_['大戶人數']:,} 人",
                                 f"{d_ppl:+,} 人" if d_ppl is not None else None)
                    if not np.isnan(now_["散戶比例"]):
                        c.metric("散戶持股比例", f"{now_['散戶比例']:.2f}%",
                                 f"{d_sml:+.2f}%" if not np.isnan(d_sml) else None)
                    if now_.get("總股東人數"):
                        e.metric("總股東人數", f"{now_['總股東人數']:,} 人",
                                 f"{d_all:+,} 人" if d_all is not None else None)

                    # ===== 四種組合自動判斷 =====
                    st.write("")
                    st.write("### 籌碼在往哪裡流動")

                    if d_ppl is None:
                        if d_pct > 1:
                            st.success("### 🐋 大戶在買（比例增加）")
                        elif d_pct < -1:
                            st.error("### 📉 大戶在賣（比例減少）")
                        else:
                            st.info("### ▫️ 大戶持股穩定")
                    else:
                        up_pct = d_pct > 0.5
                        dn_pct = d_pct < -0.5
                        up_ppl = d_ppl > 0
                        dn_ppl = d_ppl < 0

                        if up_pct and dn_ppl:
                            st.success("### 🐋🐋 少數人在偷偷吃貨（最強的籌碼訊號）")
                            st.write(f"**大戶人數少了 {abs(d_ppl):,} 人，"
                                     f"但持股比例反而多了 {d_pct:.2f}%。**")
                            st.write("意思是：有人把其他大戶手上的股票也吃下來了，"
                                     "籌碼越來越集中在少數幾個帳戶。")
                            st.write("這是市場上講的「主力吸籌」最接近的客觀證據。")
                            chip_score = 3
                        elif up_pct and up_ppl:
                            st.success("### 🐋 有新的大戶進場")
                            st.write(f"**大戶人數多了 {d_ppl:,} 人，"
                                     f"持股比例也增加 {d_pct:.2f}%。**")
                            st.write("買盤來自多個帳戶，比較分散但也代表看好的人變多。")
                            chip_score = 2
                        elif dn_pct and up_ppl:
                            st.error("### 📉 大戶在分散出貨")
                            st.write(f"**大戶人數多了 {d_ppl:,} 人，"
                                     f"但持股比例卻少了 {abs(d_pct):.2f}%。**")
                            st.write("意思是：原本的大戶把股票拆散賣掉，"
                                     "變成很多小大戶。這通常是出貨的樣子。")
                            chip_score = -3
                        elif dn_pct and dn_ppl:
                            st.error("### 📉 大戶整批離場")
                            st.write(f"**大戶人數少了 {abs(d_ppl):,} 人，"
                                     f"持股比例也少了 {abs(d_pct):.2f}%。**")
                            st.write("有大戶整批賣掉走人了。")
                            chip_score = -2
                        else:
                            st.info("### ▫️ 沒有明顯變化")
                            st.write(f"大戶比例 {d_pct:+.2f}%，"
                                     f"人數 {d_ppl:+,} 人，變動都不大。")
                            chip_score = 0

                    # ===== 籌碼換手：大戶 vs 散戶 =====
                    if not np.isnan(d_sml) and d_ppl is not None:
                        st.write("")
                        st.write("### 籌碼換手方向")
                        if d_pct > 0.5 and d_sml < -0.5:
                            st.success(f"✅ **籌碼從散戶流向大戶**"
                                       f"（大戶 {d_pct:+.2f}%，散戶 {d_sml:+.2f}%）")
                            st.write("散戶把股票交出來，大戶接走。一般視為好事。")
                        elif d_pct < -0.5 and d_sml > 0.5:
                            st.error(f"❌ **籌碼從大戶流向散戶**"
                                     f"（大戶 {d_pct:+.2f}%，散戶 {d_sml:+.2f}%）")
                            st.write("大戶把股票倒給散戶。這通常不是好事。")
                        else:
                            st.info(f"▫️ 沒有明顯換手（大戶 {d_pct:+.2f}%，"
                                    f"散戶 {d_sml:+.2f}%）")

                    # ===== 每戶平均持股（集中度）=====
                    if "每戶平均" in bd.columns and bd["每戶平均"].notna().any():
                        e0 = float(first["每戶平均"])
                        e1 = float(now_["每戶平均"])
                        if e0 > 0:
                            chg_e = (e1 / e0 - 1) * 100
                            st.write("")
                            st.write("### 集中度")
                            st.write(f"平均每個大戶握有的比重："
                                     f"**{chg_e:+.1f}%**（{wk}週來）")
                            if chg_e > 5:
                                st.success("集中度上升 — 平均每個大戶握的股票變多了。")
                            elif chg_e < -5:
                                st.warning("集中度下降 — 大戶手上的股票被分散掉了。")
                            else:
                                st.info("集中度沒什麼變化。")

                    # ===== 成交換手率 =====
                    shares = (info.get("sharesOutstanding")
                              or info.get("floatShares")) if info else None
                    if shares:
                        tv5 = float(d["Volume"].tail(5).mean())
                        tv20 = float(d["Volume"].tail(20).mean())
                        to5 = tv5 / shares * 100
                        to20 = tv20 / shares * 100
                        st.write("")
                        st.write("### 成交換手率")
                        st.write(f"近5期 **{to5:.2f}%**　近20期 {to20:.2f}%")
                        st.caption("換手率 = 每天成交的股數 ÷ 在外流通股數。"
                                   "代表這檔股票每天有多少比例的股份易主。")
                        if to5 > 5:
                            st.warning("換手非常熱絡（>5%）— 短線資金進出激烈，"
                                       "波動會大。")
                        elif to5 > 2:
                            st.info("換手活躍（2-5%）— 有一定人氣。")
                        elif to5 < 0.5:
                            st.warning("換手冷清（<0.5%）— 沒什麼人交易，"
                                       "買賣可能不順。")
                        else:
                            st.info("換手正常。")
                        if to5 > to20 * 1.5:
                            st.write("**近期換手明顯放大** — 有事情在發生，"
                                     "配合股價方向一起看。")

                    # ===== 圖 =====
                    st.write("")
                    fg = make_subplots(rows=2, cols=1, shared_xaxes=True,
                                       row_heights=[0.55, 0.45],
                                       vertical_spacing=0.08,
                                       subplot_titles=("持股比例 %", "大戶人數"))
                    fg.add_trace(go.Scatter(x=bd["日期"], y=bd["大戶比例"],
                                            name="大戶（400張以上）",
                                            line=dict(color="crimson", width=3)),
                                 row=1, col=1)
                    if bd["散戶比例"].notna().any():
                        fg.add_trace(go.Scatter(x=bd["日期"], y=bd["散戶比例"],
                                                name="散戶（10張以下）",
                                                line=dict(color="steelblue", width=2)),
                                     row=1, col=1)
                    if bd["大戶人數"].notna().any():
                        fg.add_trace(go.Bar(x=bd["日期"], y=bd["大戶人數"],
                                            name="大戶人數",
                                            marker_color="darkorange", opacity=0.7),
                                     row=2, col=1)
                    fg.update_layout(height=480, margin=dict(t=50, b=40),
                                     hovermode="x unified",
                                     legend=dict(orientation="h", y=1.1))
                    st.plotly_chart(fg, use_container_width=True)

                    # ===== 圖的文字版（不看圖也能懂）=====
                    st.write("**這張圖在說什麼（文字版）**")

                    def move_txt(series, unit="%", digits=2):
                        v0 = float(series.iloc[0]); v1 = float(series.iloc[-1])
                        hi = float(series.max()); lo = float(series.min())
                        d_ = v1 - v0
                        if abs(d_) < (0.3 if unit == "%" else max(hi * 0.01, 1)):
                            shape = "幾乎沒動"
                        elif d_ > 0:
                            shape = "整體往上"
                        else:
                            shape = "整體往下"
                        return v0, v1, hi, lo, d_, shape

                    v0, v1, hi_, lo_, dd_, shape = move_txt(bd["大戶比例"])
                    st.write(f"　**上排紅線（大戶持股比例）：** {shape}。"
                             f"{wk}週前是 {v0:.2f}%，現在 {v1:.2f}%"
                             f"（{dd_:+.2f}%）。"
                             f"期間最高 {hi_:.2f}%、最低 {lo_:.2f}%。")

                    if bd["散戶比例"].notna().any():
                        s0, s1, sh, sl_, sd, sshape = move_txt(bd["散戶比例"])
                        st.write(f"　**上排藍線（散戶持股比例）：** {sshape}。"
                                 f"{wk}週前 {s0:.2f}%，現在 {s1:.2f}%"
                                 f"（{sd:+.2f}%）。")

                    if bd["大戶人數"].notna().any():
                        p0 = int(bd["大戶人數"].iloc[0])
                        p1 = int(bd["大戶人數"].iloc[-1])
                        ph = int(bd["大戶人數"].max())
                        pl = int(bd["大戶人數"].min())
                        pd_ = p1 - p0
                        pshape = ("整體變多" if pd_ > 0 else
                                  ("整體變少" if pd_ < 0 else "沒變"))
                        st.write(f"　**下排柱子（大戶人數）：** {pshape}。"
                                 f"{wk}週前 {p0:,} 人，現在 {p1:,} 人"
                                 f"（{pd_:+,} 人）。"
                                 f"期間最多 {ph:,} 人、最少 {pl:,} 人。")

                    # 最近三週逐週變化，不看圖也能追
                    if len(bd) >= 3:
                        st.write("　**最近三週的變化：**")
                        tail = bd.tail(4).reset_index(drop=True)
                        for i in range(1, len(tail)):
                            cur_ = tail.iloc[i]; pre_ = tail.iloc[i - 1]
                            dp = cur_["大戶比例"] - pre_["大戶比例"]
                            line = (f"　　• {cur_['日期']}　"
                                    f"大戶比例 {cur_['大戶比例']:.2f}%"
                                    f"（{dp:+.2f}）")
                            if (cur_.get("大戶人數") is not None
                                    and pre_.get("大戶人數") is not None):
                                dq = int(cur_["大戶人數"]) - int(pre_["大戶人數"])
                                line += f"　人數 {int(cur_['大戶人數']):,} 人（{dq:+,}）"
                                if dp > 0 and dq < 0:
                                    line += "　← 吃貨"
                                elif dp < 0 and dq > 0:
                                    line += "　← 出貨"
                            st.write(line)

                    st.caption("看圖的話：紅線往上=大戶持股比例增加。"
                               "**紅線往上、柱子往下 = 少數人在集中吃貨**，"
                               "這是最值得注意的組合。")

                    with st.expander("看每週明細"):
                        cols = [c for c in ["日期", "大戶比例", "散戶比例",
                                            "千張比例", "大戶人數", "總股東人數"]
                                if c in bd.columns]
                        st.dataframe(bd[cols].iloc[::-1].reset_index(drop=True),
                                     use_container_width=True, hide_index=True)

                    st.warning("**提醒：** 集保每週五結算、至少落後一週。"
                               "「大戶」是持股400張以上的帳戶，裡面包含公司派、外資、"
                               "券商自營，**不一定是在做多**。"
                               "籌碼集中不保證會漲，這只是多一個參考角度，沒有回測驗證。")
            except Exception as ex:
                st.error(f"集保連線失敗：{type(ex).__name__}")
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

            st.write("**這張圖在說什麼（文字版）**")
            st.write(describe_line(show["Close"], "大盤指數（黑線）", digits=0))
            st.write(describe_line(show["MA60"], "60日均價（紅線）", digits=0))
            ab = int((show["Close"] > show["MA60"]).sum())
            st.write(f"　這 {len(show)} 天裡，大盤有 **{ab} 天在均價之上**、"
                     f"{len(show)-ab} 天在下面。")
            if ab > len(show) * 0.7:
                st.write("　→ 大盤長期站在均價上，**多頭環境**。個股比較好做。")
            elif ab < len(show) * 0.3:
                st.write("　→ 大盤長期在均價下，**空頭環境**。這時候買個股是逆勢。")
            else:
                st.write("　→ 大盤來回穿越均價，**震盪環境**。方向不明。")

    # ---------- 夜盤 / 隔日開盤預期 ----------
    if show_night:
        st.markdown("---")
        st.write("## 🌙 夜盤與隔日開盤預期")
        st.caption("台股白天收盤後，國際市場還在跑。這些會反映在隔天開盤。")

        NIGHT = {
            "標普500期貨": "ES=F",
            "那斯達克期貨": "NQ=F",
            "道瓊期貨": "YM=F",
            "費城半導體": "^SOX",
            "美元指數": "DX-Y.NYB",
        }
        if mkt == "台股":
            NIGHT["布蘭特原油"] = "BZ=F"

        nrows = []
        with st.spinner("查國際市場..."):
            for nm, sy in NIGHT.items():
                nd = get_data(sy, "1d", "3mo")
                if nd is None or len(nd) < 3:
                    nrows.append({"項目": nm, "狀態": "查不到"})
                    continue
                c_ = nd["Close"]
                p1 = float(c_.iloc[-1]); p0 = float(c_.iloc[-2])
                chg = (p1 / p0 - 1) * 100
                w = (p1 / float(c_.iloc[-6]) - 1) * 100 if len(c_) > 6 else 0
                nrows.append({
                    "項目": nm, "最新": round(p1, 2),
                    "上一交易日比%": round(chg, 2),
                    "近5日%": round(w, 2),
                    "方向": "🔴 漲" if chg > 0.3 else ("🟢 跌" if chg < -0.3 else "▫️ 平"),
                })

        ndf = pd.DataFrame(nrows)
        st.dataframe(ndf, use_container_width=True, hide_index=True)

        # 自動判斷隔日開盤傾向
        valid = ndf.dropna(subset=["上一交易日比%"]) if "上一交易日比%" in ndf.columns \
            else pd.DataFrame()
        if not valid.empty:
            key = valid[valid["項目"].isin(["標普500期貨", "那斯達克期貨",
                                            "費城半導體"])]
            if not key.empty:
                avg = float(key["上一交易日比%"].mean())
                up = int((key["上一交易日比%"] > 0.3).sum())
                dn = int((key["上一交易日比%"] < -0.3).sum())

                st.write("")
                st.write("### 隔日開盤傾向（自動判斷）")

                if avg > 0.8:
                    st.success("### 🔴 偏向開高")
                    st.write(f"美股期貨與費半平均 **{avg:+.2f}%**，"
                             f"{up} 項明顯上漲。台股隔天通常會跟著開高。")
                elif avg < -0.8:
                    st.error("### 🟢 偏向開低")
                    st.write(f"美股期貨與費半平均 **{avg:+.2f}%**，"
                             f"{dn} 項明顯下跌。台股隔天通常會跟著開低。")
                else:
                    st.info("### ▫️ 平盤附近")
                    st.write(f"美股期貨與費半平均 **{avg:+.2f}%**，"
                             f"沒有明顯方向。")

                # 費半對台股特別重要
                sox = valid[valid["項目"] == "費城半導體"]
                if not sox.empty and mkt == "台股":
                    sv = float(sox.iloc[0]["上一交易日比%"])
                    st.write("")
                    st.write(f"**費城半導體 {sv:+.2f}%** — "
                             f"這個對台股最重要，因為台積電、聯發科這些權值股"
                             f"跟它連動最深。")
                    if abs(sv) > 2:
                        st.warning(f"費半波動超過 2%，台股半導體類股"
                                   f"隔天可能會有較大反應。")

        st.write("")
        st.write("**逐項說明**")
        for _, rw in ndf.iterrows():
            if "上一交易日比%" not in rw or pd.isna(rw.get("上一交易日比%")):
                st.write(f"　• **{rw['項目']}**：查不到資料")
                continue
            st.write(f"　• **{rw['項目']}**：最新 {rw['最新']:,.2f}，"
                     f"比上一交易日 {rw['上一交易日比%']:+.2f}%，"
                     f"近5日 {rw['近5日%']:+.2f}%")

        st.warning("**這一頁的用途有限。** 夜盤反映的是隔天、隔幾小時的情緒。"
                   "你的666規則是等「跌破均線」才出場，一筆常常會抱好幾週 —— "
                   "用看幾小時的資訊去決定這種部位，時間尺度不對。"
                   "**這裡只適合用來決定「今天下單還是明天下單」，"
                   "不適合用來決定「要不要買這檔」。**")

        st.caption("註：台指期夜盤資料期交所有提供，但需要另外處理且格式常變動。"
                   "這裡改用美股期貨與費半 —— 它們是台股夜間最主要的影響來源，"
                   "資料也比較穩定。")

    # ---------- 總體環境 ----------
    if show_macro:
        st.markdown("---")
        st.write("## 🌐 總體環境")
        st.caption("油價、美債殖利率、匯率、恐慌指數 —— "
                   "這些不看線圖，但會決定整個市場的水位。")

        MACRO = {
            "10年期美債殖利率": ("^TNX", "%", 0),
            "布蘭特原油": ("BZ=F", "美元", 0),
            "西德州原油": ("CL=F", "美元", 0),
            "美元指數": ("DX-Y.NYB", "", 0),
            "恐慌指數VIX": ("^VIX", "", 0),
            "黃金": ("GC=F", "美元", 0),
        }

        mrows = {}
        with st.spinner("查總體資料..."):
            for nm, (sy, unit, _) in MACRO.items():
                md = get_data(sy, "1d", "1y")
                if md is None or len(md) < 30:
                    mrows[nm] = None
                    continue
                c_ = md["Close"]
                cur_v = float(c_.iloc[-1])
                d1 = (cur_v / float(c_.iloc[-2]) - 1) * 100
                d20 = (cur_v / float(c_.iloc[-21]) - 1) * 100 if len(c_) > 21 else 0
                hi52 = float(c_.max()); lo52 = float(c_.min())
                pos = ((cur_v - lo52) / (hi52 - lo52) * 100) if hi52 > lo52 else 50
                mrows[nm] = {"值": cur_v, "日變%": d1, "月變%": d20,
                             "高": hi52, "低": lo52, "位置%": pos, "單位": unit}

        tbl = []
        for nm, v in mrows.items():
            if v is None:
                tbl.append({"項目": nm, "狀態": "查不到"})
                continue
            tbl.append({
                "項目": nm,
                "現值": round(v["值"], 2),
                "日變%": round(v["日變%"], 2),
                "近月變%": round(v["月變%"], 2),
                "一年區間位置%": round(v["位置%"], 0),
            })
        st.dataframe(pd.DataFrame(tbl), use_container_width=True, hide_index=True)
        st.caption("「一年區間位置」= 現在的值在過去一年最高最低之間的位置。"
                   "100%代表在一年高點，0%代表在一年低點。")

        # ===== 自動判讀傳導鏈 =====
        st.write("")
        st.write("### 傳導鏈自動判斷")

        y = mrows.get("10年期美債殖利率")
        oil = mrows.get("布蘭特原油") or mrows.get("西德州原油")
        vix = mrows.get("恐慌指數VIX")
        usd = mrows.get("美元指數")

        risk = 0
        notes = []

        if oil:
            if oil["位置%"] > 75:
                risk += 2
                notes.append(f"🔴 **油價在一年區間的高檔**（{oil['位置%']:.0f}%位置，"
                             f"現在 {oil['值']:.1f} 美元）。"
                             f"能源成本推高企業營運成本，也推高通膨。")
            elif oil["位置%"] < 30:
                risk -= 1
                notes.append(f"🟢 油價在低檔（{oil['位置%']:.0f}%位置）。"
                             f"通膨壓力小，對股市友善。")
            else:
                notes.append(f"▫️ 油價在中間位置（{oil['位置%']:.0f}%）。")

        if y:
            if y["值"] > 4.5:
                risk += 2
                notes.append(f"🔴 **美債殖利率偏高**（{y['值']:.2f}%）。"
                             f"殖利率越高，股票相對越沒吸引力 —— "
                             f"錢會從股市流向債市。成長股受衝擊最大。")
            elif y["值"] < 3.5:
                risk -= 1
                notes.append(f"🟢 殖利率偏低（{y['值']:.2f}%），資金環境寬鬆。")
            else:
                notes.append(f"▫️ 殖利率中性（{y['值']:.2f}%）。")

            if y["月變%"] > 3:
                risk += 1
                notes.append(f"⚠️ 殖利率近一個月上升 {y['月變%']:.1f}%，"
                             f"上升速度快的時候股市通常最難受。")

        if vix:
            if vix["值"] > 25:
                risk += 2
                notes.append(f"🔴 **VIX {vix['值']:.1f}，市場恐慌**。"
                             f"20以上就算緊張，25以上是明顯避險情緒。")
            elif vix["值"] < 15:
                risk -= 1
                notes.append(f"🟢 VIX {vix['值']:.1f}，市場平靜。"
                             f"但太低也代表大家都很樂觀，容易措手不及。")
            else:
                notes.append(f"▫️ VIX {vix['值']:.1f}，正常範圍。")

        if usd and mkt == "台股":
            if usd["月變%"] > 2:
                risk += 1
                notes.append(f"⚠️ 美元近月走強 {usd['月變%']:.1f}%。"
                             f"美元強通常代表資金回流美國，"
                             f"新興市場（含台股）容易被外資賣。")
            elif usd["月變%"] < -2:
                risk -= 1
                notes.append(f"🟢 美元近月走弱 {usd['月變%']:.1f}%，"
                             f"資金比較願意流向新興市場。")

        if risk >= 4:
            st.error("### 🔴 總體環境不利")
            st.write("多項總體指標同時偏壓，這種時候個股再好也容易被拖累。"
                     "**部位要放小，停損要嚴格。**")
        elif risk >= 2:
            st.warning("### 🟡 總體環境偏緊")
            st.write("有壓力但還不到危險。要留意。")
        elif risk <= -2:
            st.success("### 🟢 總體環境寬鬆")
            st.write("資金環境友善，個股比較好做。")
        else:
            st.info("### ▫️ 總體環境中性")

        st.write("")
        for n in notes:
            st.write(f"　• {n}")

        # ===== 傳導鏈說明 =====
        with st.expander("📖 這些東西怎麼影響台股？"):
            st.write("""
            **鏈條是這樣走的：**

            ```
            油價上漲
              ↓
            通膨壓力上升（能源是所有東西的成本）
              ↓
            市場預期央行要升息（或不能降息）
              ↓
            美債殖利率上升
              ↓
            股票相對沒吸引力（錢放債券就有4~5%，何必冒險）
              ↓
            股市估值下修，成長股跌最兇
              ↓
            台股跟著跌（尤其電子權值股）
            ```

            **為什麼台股特別敏感？**

            1. **台股是外資主導的市場。** 美元走強、美債殖利率高的時候，
               外資會把錢抽回美國，台股就會被賣。

            2. **台股權值股集中在半導體。** 半導體是典型的成長股，
               估值對利率最敏感。利率一升，本益比就要下修。

            3. **台灣是能源進口國。** 油價漲直接推高企業成本，
               尤其塑化、航運、水泥這些。

            **哪些台股會受惠？**

            油價漲的時候，能源相關（台塑集團在油價高時反而可能受惠庫存利益）、
            航運（運價可能跟漲）不一定跟著跌。但整體來說，油價高檔對台股是負面的。

            **這一頁的用途：** 不是拿來做進出場判斷，
            是拿來決定「**現在該不該重倉**」。
            總體環境緊的時候，同樣的訊號成功率會下降。
            """)

        st.warning("**這一頁沒有經過回測驗證。** 上面的傳導邏輯是經濟學上的常識，"
                   "不是從你的資料跑出來的結論。"
                   "而且總體環境變化很慢，不適合當短線進出的依據。")

    st.markdown("---")
    st.warning("以上都是照規則機械式計算的結果，不是買賣建議。"
               "要不要進出請自己判斷，並嚴守停損。")




# ============================================================
# 分頁：原油ETF（獨立，規則跟個股不同）
# ============================================================
with tab_oil:
    st.write("## 🛢️ 原油ETF 專用分析")
    st.info("**這一頁的規則跟個股分頁不一樣，而且是刻意的。**\n\n"
            "個股分頁用的666規則，出場條件是「跌破均線」，一筆常常會抱好幾週。"
            "但原油ETF因為波動耗損，**不能抱那麼久**。"
            "所以這一頁改用短線規則：方向對才進、2~4週內出場、停損拉緊。")

    if st.button("開始分析原油ETF", use_container_width=True,
                 type="primary", key="oilbtn"):
        st.markdown("---")
        st.write("## 🛢️ 原油ETF 買賣判斷")
        st.caption("00715L 布蘭特正2｜00642U S&P石油｜00673R 原油反1")

        OIL_ETF = {
            "00715L.TW": ("期街口布蘭特正2", "BZ=F", "布蘭特原油", 2.0),
            "00642U.TW": ("期元大S&P石油", "CL=F", "西德州原油", 1.0),
            "00673R.TW": ("期元大S&P原油反1", "CL=F", "西德州原油", -1.0),
        }

        with st.spinner("查油價與ETF..."):
            bz = get_data("BZ=F", "1d", "1y")
            wt = get_data("CL=F", "1d", "1y")

        if bz is None and wt is None:
            st.error("抓不到油價資料")
        else:
            # ===== 先判斷油價本身 =====
            st.write("### 第一步：油價本身在漲還是在跌")

            oil_view = {}
            for nm, dfo in [("布蘭特原油", bz), ("西德州原油", wt)]:
                if dfo is None or len(dfo) < 70:
                    continue
                do = add_ind(dfo)
                po = float(do["Close"].iloc[-1])
                mo = float(do["MA60"].iloc[-1])
                ko = float(do["K"].iloc[-1]); kop = float(do["K"].iloc[-2])
                mco = float(do["MACD"].iloc[-1]); sgo = float(do["SIG"].iloc[-1])
                to_, so_ = trend_of(do)
                hits = sum([po > mo, (ko > 50 and ko > kop), mco > sgo])

                # 波動度（判斷會不會被耗損吃掉）
                vol_ann = float(do["Close"].pct_change().tail(60).std()
                                * np.sqrt(252) * 100)

                oil_view[nm] = {"價": po, "均": mo, "趨勢": to_, "斜率": so_,
                                "規則": hits, "K": ko, "波動": vol_ann}

                icon = ("🔴" if to_ == "往上" else
                        ("🟢" if to_ == "往下" else "▫️"))
                st.write(f"**{icon} {nm}**　現價 {po:.2f} 美元　"
                         f"60日均 {mo:.2f}　趨勢**{to_}**（{so_:+.1f}%）　"
                         f"三條規則 {hits}/3　年化波動 {vol_ann:.0f}%")

            if not oil_view:
                st.error("油價資料不足")
            else:
                main = oil_view.get("布蘭特原油") or list(oil_view.values())[0]
                trend_oil = main["趨勢"]
                hits_oil = main["規則"]
                vol_oil = main["波動"]

                # ===== 第二步：三檔各自判斷 =====
                st.markdown("---")
                st.write("### 第二步：三檔各自該怎麼做")

                for etf, (nm, und, undnm, mult) in OIL_ETF.items():
                    de = get_data(etf, "1d", "1y")
                    base = oil_view.get(undnm)
                    if base is None:
                        continue

                    st.write("")
                    st.write(f"#### {etf}　{nm}　（{mult:+.0f}倍 {undnm}）")

                    cur_p = float(de["Close"].iloc[-1]) if de is not None else None

                    # 方向是否對得上
                    if mult > 0:
                        aligned = base["趨勢"] == "往上"
                        wrong = base["趨勢"] == "往下"
                    else:
                        aligned = base["趨勢"] == "往下"
                        wrong = base["趨勢"] == "往上"

                    chop = base["趨勢"] == "橫盤"

                    if chop:
                        st.error("### 🚫 不要碰")
                        st.write(f"**油價在盤整**（{base['斜率']:+.1f}%）。"
                                 f"這是槓桿／反向ETF最傷的環境 —— "
                                 f"上上下下的波動會不斷侵蝕淨值，"
                                 f"就算油價回到原點，你還是會虧。")
                    elif wrong:
                        st.error("### 🚫 方向相反，不要碰")
                        st.write(f"這檔是{'做多' if mult>0 else '做空'}油價的，"
                                 f"但油價現在**{base['趨勢']}**。方向不對。")
                    elif aligned and base["規則"] >= 2:
                        if abs(mult) > 1:
                            st.warning("### 🟡 方向對，但這是2倍槓桿")
                        else:
                            st.success("### 🟢 方向對，條件可以")
                        st.write(f"油價**{base['趨勢']}**（{base['斜率']:+.1f}%），"
                                 f"三條規則過 {base['規則']}/3。")
                        if cur_p:
                            sl = cur_p * (0.90 if abs(mult) > 1 else 0.92)
                            st.write(f"　**現價 {cur_p:.2f}**　"
                                     f"建議停損 **{sl:.2f}**"
                                     f"（{'-10%' if abs(mult)>1 else '-8%'}）")
                            st.write(f"　**建議持有：最多 2~4 週。**"
                                     f"這類產品不能長抱。")
                    elif aligned:
                        st.warning("### 🟡 方向對但訊號不夠強")
                        st.write(f"油價{base['趨勢']}，但三條規則只過 "
                                 f"{base['規則']}/3。再等等。")
                    else:
                        st.info("### ▫️ 觀望")

                    # 追蹤誤差
                    if de is not None and len(de) > 60:
                        und_df = bz if und == "BZ=F" else wt
                        if und_df is not None:
                            e = de["Close"].tail(120)
                            u = und_df["Close"].reindex(e.index).ffill().dropna()
                            e = e.reindex(u.index).dropna()
                            if len(e) > 20:
                                er = (float(e.iloc[-1])/float(e.iloc[0])-1)*100
                                ur = (float(u.iloc[-1])/float(u.iloc[0])-1)*100
                                th = ur * mult
                                gap = er - th
                                st.write(f"　**過去{len(e)}天實際表現：** "
                                         f"油價 {ur:+.1f}%，"
                                         f"照{mult:+.0f}倍算應該 {th:+.1f}%，"
                                         f"實際 {er:+.1f}%"
                                         f"（**落差 {gap:+.1f} 個百分點**）")
                                if gap < -5:
                                    st.write(f"　　→ 被吃掉 {abs(gap):.1f} 個百分點。"
                                             f"這就是波動耗損＋轉倉成本＋管理費。")

                # ===== 總結 =====
                st.markdown("---")
                st.write("### 📌 一句話總結")

                if trend_oil == "橫盤":
                    st.error(f"**油價在盤整（{main['斜率']:+.1f}%），"
                             f"三檔全部不要碰。**")
                    st.write("盤整是槓桿和反向ETF最傷的環境，"
                             "上下震盪會持續侵蝕淨值。")
                elif trend_oil == "往上":
                    st.write(f"**油價往上（{main['斜率']:+.1f}%）**　"
                             f"→ 方向上適合 00715L 或 00642U，"
                             f"00673R 反向不要碰。")
                    if hits_oil < 2:
                        st.warning("但三條規則只過 " f"{hits_oil}/3，訊號不夠強，建議再等。")
                else:
                    st.write(f"**油價往下（{main['斜率']:+.1f}%）**　"
                             f"→ 方向上適合 00673R，"
                             f"00715L 和 00642U 不要碰。")

                if vol_oil > 40:
                    st.error(f"⚠️ **油價年化波動 {vol_oil:.0f}%，非常高。**"
                             f"波動越大，槓桿ETF的耗損越嚴重。"
                             f"這種環境下 00715L 特別危險。")
                elif vol_oil > 30:
                    st.warning(f"⚠️ 油價年化波動 {vol_oil:.0f}%，偏高。"
                               f"槓桿產品耗損會比較明顯。")

        # ===== 固定警語 =====
        st.markdown("---")
        with st.expander("📖 為什麼這三檔不能像股票一樣抱？（必讀）"):
            st.write("""
            **這三檔都是「期貨型」ETF，不是持有真的石油。** 它們有三層成本：

            **1. 波動耗損（正2和反1才有）**
            槓桿ETF追蹤的是「每日」報酬的兩倍，不是長期報酬的兩倍。
            因為每天重設槓桿，漲跌的不對稱在複利下會放大損失。

            舉個實際的例子：

            | 油價走勢 | 油價 | 理論2倍 | 實際 | 落差 |
            |---|---|---|---|---|
            | 一路上漲 | +62.9% | +125.8% | +159.4% | **+33.6%** |
            | **來回震盪** | −4.9% | −9.8% | **−18.5%** | **−8.7%** |
            | 先漲後跌 | −3.2% | −6.3% | −12.2% | −5.8% |

            **只有一路單邊上漲時，槓桿才是你的朋友。**
            一旦震盪，耗損就開始吃你。而原油正是震盪最兇的商品之一。

            **2. 轉倉成本**
            期貨有到期日，每個月要換倉。當遠月比近月貴的時候（正價差），
            換倉等於「賣掉便宜的、買回貴的」，那個價差就是成本。

            **3. 管理費**
            經理費加保管費每年約 1.2%，不管賺賠都要收。

            ---

            **這裡有個你必須知道的矛盾：**

            你的666規則**沒有時間限制** —— 出場只看「跌破均線」或「到停損」。
            只要股價一直站在均線上，這筆就會一直抱著，可能好幾個月。

            但這三檔產品的設計，就是**不能抱那麼久**。

            **你的規則可能讓你抱很久，這些工具卻不允許。兩者會打架。**

            如果硬用666規則去做00715L，很可能發生：
            訊號叫你抱著，但淨值被耗損一天天吃掉，
            油價回到原點，你卻虧了15%。

            ---

            **比較合理的用法：**

            • **要用這三檔** → 幾天到兩三週內出場，當短線工具
            • **要用你的666規則** → 換成原型的、沒槓桿的標的

            兩者不要混用。
            """)

        st.warning("**再說一次：** 這三檔不適合長期持有。"
                   "台灣主管機關對槓桿反向ETF有特別警語，"
                   "元大S&P原油正2（00672L）在2020年就因為淨值過低而清算下市過。"
                   "**用小部位、設好停損、短期進出。**")
    else:
        st.write("按上面的按鈕開始分析。")
        st.write("")
        st.write("**這一頁會告訴你：**")
        st.write("　• 油價本身現在在漲、在跌、還是在盤整")
        st.write("　• 三檔各自該買、該賣、還是不要碰")
        st.write("　• 每一檔實際被耗損吃掉多少（用真實資料算）")
        st.write("　• 建議的停損價和持有時間")




# ============================================================
# 分頁：熱門類股（資金流向偵測）
# ============================================================
with tab_hot:
    st.write("## 🔥 現在資金往哪裡流")
    st.caption("用實際漲跌和成交量偵測，不是猜新聞。"
               "哪個類股最近突然變強，就是錢在往那邊跑。")

    SECTORS = {
        "半導體": ["2330.TW", "2454.TW", "3711.TW"],
        "被動元件": ["2327.TW", "2492.TW", "2456.TW"],
        "AI伺服器": ["2382.TW", "2376.TW", "6669.TW"],
        "金融": ["2881.TW", "2882.TW", "2891.TW"],
        "航運": ["2603.TW", "2609.TW", "2615.TW"],
        "塑化": ["1301.TW", "1303.TW", "1326.TW"],
        "鋼鐵": ["2002.TW", "2027.TW", "2023.TW"],
        "電信": ["2412.TW", "3045.TW", "4904.TW"],
        "重電綠能": ["1519.TW", "1503.TW", "1513.TW"],
        "生技": ["4142.TW", "6446.TW", "1795.TW"],
        "汽車零件": ["2227.TW", "1536.TW", "2231.TW"],
        "紡織": ["1476.TW", "1477.TW", "9910.TW"],
    }

    hot_days = st.radio("看多久的變化",
                        ["近5日（短線）", "近20日（一個月）", "近60日（三個月）"],
                        horizontal=True, index=0)
    win = {"近5日（短線）": 5, "近20日（一個月）": 20,
           "近60日（三個月）": 60}[hot_days]

    if st.button("開始偵測", use_container_width=True, type="primary", key="hotbtn"):
        rows = []
        bar = st.progress(0.0)
        total = sum(len(v) for v in SECTORS.values())
        done = 0

        for sec, codes in SECTORS.items():
            rets, vols, names, ok_n = [], [], [], 0
            for cd in codes:
                dfh = get_data(cd, "1d", "6mo")
                done += 1
                bar.progress(min(done / total, 1.0), text=f"{sec} {cd}")
                if dfh is None or len(dfh) < win + 25:
                    continue
                c_ = dfh["Close"]; v_ = dfh["Volume"]
                r = (float(c_.iloc[-1]) / float(c_.iloc[-1 - win]) - 1) * 100
                vr_ = (float(v_.tail(win).mean())
                       / float(v_.tail(60).mean())) if float(v_.tail(60).mean()) > 0 else 1
                rets.append(r); vols.append(vr_); names.append(cd); ok_n += 1
            if ok_n == 0:
                continue
            rows.append({
                "類股": sec,
                f"平均漲跌%": round(float(np.mean(rets)), 2),
                "量能倍數": round(float(np.mean(vols)), 2),
                "檔數": ok_n,
                "個股": "、".join(names),
            })
        bar.empty()

        if not rows:
            st.error("抓不到資料")
        else:
            res = pd.DataFrame(rows).sort_values(f"平均漲跌%", ascending=False)
            res = res.reset_index(drop=True)
            res.index = res.index + 1

            st.markdown("---")
            st.write(f"### 類股強弱排行（{hot_days}）")

            def colf(v):
                try:
                    return "color: green" if v > 0 else "color: red"
                except Exception:
                    return ""

            st.dataframe(
                res[["類股", f"平均漲跌%", "量能倍數", "檔數"]].style.map(
                    colf, subset=[f"平均漲跌%"]),
                use_container_width=True)

            # ===== 自動判讀 =====
            st.markdown("---")
            st.write("### 📌 自動判讀")

            top = res.iloc[0]
            bot = res.iloc[-1]
            avg_all = float(res[f"平均漲跌%"].mean())

            st.write(f"**最強：{top['類股']}**　{hot_days}平均 "
                     f"{top[f'平均漲跌%']:+.2f}%，量能是平常的 {top['量能倍數']:.2f} 倍")
            st.write(f"**最弱：{bot['類股']}**　{hot_days}平均 "
                     f"{bot[f'平均漲跌%']:+.2f}%")
            st.write(f"**全體平均：** {avg_all:+.2f}%")

            # 有量有價才算真的熱
            real_hot = res[(res[f"平均漲跌%"] > avg_all + 2)
                           & (res["量能倍數"] > 1.2)]
            fake_hot = res[(res[f"平均漲跌%"] > avg_all + 2)
                           & (res["量能倍數"] <= 1.0)]

            st.write("")
            if not real_hot.empty:
                st.success("### 🔥 有量有價的類股（真的有資金在進）")
                for _, r in real_hot.iterrows():
                    st.write(f"　• **{r['類股']}**　漲 {r[f'平均漲跌%']:+.2f}%，"
                             f"量放大到 {r['量能倍數']:.2f} 倍　"
                             f"（{r['個股']}）")
                st.write("")
                st.write("**價漲量增代表有人真的拿錢在買，不是虛漲。**"
                         "這是資金流入的證據。")
            else:
                st.info("目前沒有「明顯領漲又放量」的類股。")

            if not fake_hot.empty:
                st.warning("### ⚠️ 漲但沒量的類股（小心）")
                for _, r in fake_hot.iterrows():
                    st.write(f"　• **{r['類股']}**　漲 {r[f'平均漲跌%']:+.2f}%，"
                             f"但量只有 {r['量能倍數']:.2f} 倍")
                st.write("**價格在漲但沒人跟著買，這種漲勢通常撐不久。**")

            weak = res[res[f"平均漲跌%"] < avg_all - 2]
            if not weak.empty:
                st.write("")
                st.error("### 📉 明顯落後的類股")
                for _, r in weak.iterrows():
                    st.write(f"　• {r['類股']}　{r[f'平均漲跌%']:+.2f}%")

            # ===== 輪動偵測：短期 vs 中期排名變化 =====
            if win == 5:
                st.markdown("---")
                st.write("### 🔄 類股輪動偵測")
                st.caption("比較「近5日排名」和「近20日排名」。"
                           "名次往前跳的，就是最近才轉強的類股。")

                rows20 = []
                bar2 = st.progress(0.0)
                done2 = 0
                for sec, codes in SECTORS.items():
                    rr = []
                    for cd in codes:
                        dfh = get_data(cd, "1d", "6mo")
                        done2 += 1
                        bar2.progress(min(done2 / total, 1.0))
                        if dfh is None or len(dfh) < 45:
                            continue
                        c_ = dfh["Close"]
                        rr.append((float(c_.iloc[-1]) / float(c_.iloc[-21]) - 1) * 100)
                    if rr:
                        rows20.append({"類股": sec,
                                       "近20日%": round(float(np.mean(rr)), 2)})
                bar2.empty()

                if rows20:
                    r20 = pd.DataFrame(rows20).sort_values("近20日%",
                                                            ascending=False)
                    r20 = r20.reset_index(drop=True)
                    rank5 = {r["類股"]: i + 1 for i, r in res.iterrows()}
                    rank20 = {r["類股"]: i + 1 for i, r in r20.iterrows()}

                    moves = []
                    for sec in rank5:
                        if sec in rank20:
                            moves.append({
                                "類股": sec,
                                "近5日名次": rank5[sec],
                                "近20日名次": rank20[sec],
                                "名次變化": rank20[sec] - rank5[sec],
                            })
                    mv = pd.DataFrame(moves).sort_values("名次變化",
                                                          ascending=False)
                    st.dataframe(mv, use_container_width=True, hide_index=True)

                    rising = mv[mv["名次變化"] >= 3]
                    falling = mv[mv["名次變化"] <= -3]

                    if not rising.empty:
                        st.success("### ⬆️ 最近才轉強的（資金剛進來）")
                        for _, r in rising.iterrows():
                            st.write(f"　• **{r['類股']}**　"
                                     f"20日排第 {r['近20日名次']} 名 → "
                                     f"5日跳到第 {r['近5日名次']} 名"
                                     f"（前進 {r['名次變化']} 名）")
                        st.write("**名次快速往前 = 資金剛開始流入。**"
                                 "這通常是題材剛發酵的階段。")
                    if not falling.empty:
                        st.error("### ⬇️ 最近轉弱的（資金在撤）")
                        for _, r in falling.iterrows():
                            st.write(f"　• {r['類股']}　"
                                     f"20日第 {r['近20日名次']} 名 → "
                                     f"5日掉到第 {r['近5日名次']} 名")
                    if rising.empty and falling.empty:
                        st.info("沒有明顯的類股輪動，資金分布穩定。")

            # ===== 圖 =====
            st.markdown("---")
            fig = go.Figure()
            fig.add_trace(go.Bar(
                x=res["類股"], y=res[f"平均漲跌%"],
                marker_color=["#d62728" if v > 0 else "#2ca02c"
                              for v in res[f"平均漲跌%"]]))
            fig.add_hline(y=0, line_color="black")
            fig.add_hline(y=avg_all, line_dash="dash", line_color="gray",
                          annotation_text=f"全體平均 {avg_all:+.1f}%")
            fig.update_layout(height=380, margin=dict(t=30, b=80),
                              yaxis_title=f"{hot_days} 平均漲跌 %")
            st.plotly_chart(fig, use_container_width=True)

            st.write("**這張圖在說什麼（文字版）**")
            st.write(f"　共比較 {len(res)} 個類股。紅色是漲的、綠色是跌的，"
                     f"灰虛線是全體平均 {avg_all:+.1f}%。")
            up_n = int((res[f"平均漲跌%"] > 0).sum())
            st.write(f"　{up_n} 個類股是漲的、{len(res)-up_n} 個是跌的。")
            if up_n > len(res) * 0.7:
                st.write("　→ **普漲格局**，多數類股都在漲，市場氣氛好。")
            elif up_n < len(res) * 0.3:
                st.write("　→ **普跌格局**，多數類股都在跌。這種時候選股再好也難做。")
            else:
                st.write("　→ **輪動格局**，有漲有跌。錢在特定類股之間流動，"
                         "選對類股比選對個股重要。")

    else:
        st.write("按上面的按鈕開始偵測。")
        st.write("")
        st.write("**這一頁在做什麼：**")
        st.write("　用 12 個台股類股、每個類股 3 檔代表股的實際漲跌和成交量，"
                 "算出現在資金往哪裡流。")
        st.write("")
        st.write("**為什麼不用新聞判斷？**")
        st.write("　新聞是落後的 —— 等新聞寫出來，通常已經漲過一段了。"
                 "**價格和成交量會比新聞早反應。**"
                 "所以這裡直接看資料，不猜新聞。")
        st.write("")
        st.write("**偵測的類股：**")
        st.write("　" + "、".join(SECTORS.keys()))

    st.markdown("---")
    st.warning("**這一頁沒有回測驗證。** 「追強勢類股」在多頭市場通常有效，"
               "但在轉折點會買在最高點。而且**類股熱度變化很快**，"
               "今天最強的下週可能就換人。"
               "看到某個類股很熱，先問自己：我是第一批進的，還是最後一批？")


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

    st.write("## 大戶人數與比例的四種組合")
    st.write("""
    這是最值得看的籌碼訊號。把「持股比例」和「人數」一起看：

    | 比例 | 人數 | 意思 |
    |---|---|---|
    | ⬆️ 增 | ⬇️ 減 | 🐋🐋 **少數人在偷偷吃貨**（最強訊號）|
    | ⬆️ 增 | ⬆️ 增 | 🐋 有新的大戶進場（買盤分散）|
    | ⬇️ 減 | ⬆️ 增 | 📉 **大戶在分散出貨**（拆散賣給小大戶）|
    | ⬇️ 減 | ⬇️ 減 | 📉 大戶整批離場 |

    **為什麼「比例增、人數減」最強？**
    因為這代表有人連其他大戶的股票都吃下來了，籌碼越來越集中在少數帳戶手上。
    這是「主力吸籌」最接近的客觀證據 —— 但仍然只是推論，不是事實。
    """)

    st.write("## 換手率")
    st.write("""
    **籌碼換手** = 股票在大戶和散戶之間移轉。
    大戶比例上升、散戶比例下降 = 散戶把股票交出來給大戶，一般視為好事。

    **成交換手率** = 每天成交的股數 ÷ 在外流通股數。
    代表每天有多少比例的股份易主。

    - 超過 5% = 非常熱絡，短線資金進出激烈，波動大
    - 2~5% = 活躍，有人氣
    - 低於 0.5% = 冷清，買賣可能不順

    近期換手突然放大，代表有事情在發生，要配合股價方向一起看。
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

    **3. 短天期的交易在回測裡全是賠的。**
    42 筆交易裡，抱不到 10 天的沒有一筆賺錢。

    但要注意因果方向：**不是「抱久所以賺」，而是「賺了才抱得久」。**
    規則的出場條件是跌破均線 —— 股票漲上去才不會跌破，才抱得久；
    跌下來就馬上出場，所以短天期的都是賠的。
    **你沒辦法決定抱多久，那是市場決定的。**

    **4. 這套系統最多放你資金的 10-20%。**
    回測裡曾連續賠 14 次、資金縮水 48%。全押的話很難撐過去。
    """)

st.markdown("---")
st.caption("資料來源：Yahoo Finance、臺灣證券交易所、臺灣集中保管結算所。研究工具，不是投資建議。回測已扣手續費估算但未計滑價與流動性限制。過去表現不保證未來。")
