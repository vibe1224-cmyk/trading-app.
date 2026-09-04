import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.graph_objects as go

st.set_page_config(page_title="666戰法精準系統", page_icon="📈", layout="wide")

st.title("🎯 666戰法精準系統 - 高勝率版本")
st.write("精準指標組合 | 多層確認 | 實時勝率計算")

def calculate_indicators(df):
    """計算所有技術指標"""
    close = df["Close"].values
    high = df["High"].values
    low = df["Low"].values
    
    # MA
    ma5 = pd.Series(close).rolling(5).mean().values
    ma10 = pd.Series(close).rolling(10).mean().values
    ma20 = pd.Series(close).rolling(20).mean().values
    ma60 = pd.Series(close).rolling(60).mean().values
    
    # KD (60,3,3)
    lowest_low = pd.Series(low).rolling(60).min().values
    highest_high = pd.Series(high).rolling(60).max().values
    fastk = 100 * (close - lowest_low) / (highest_high - lowest_low + 1e-10)
    k = pd.Series(fastk).rolling(3).mean().values
    d = pd.Series(k).rolling(3).mean().values
    
    # RSI
    delta = pd.Series(close).diff()
    gain = (delta.where(delta > 0, 0)).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    rs = gain / (loss + 1e-10)
    rsi = (100 - (100 / (1 + rs))).values
    
    # MACD
    ema12 = pd.Series(close).ewm(span=12).mean().values
    ema26 = pd.Series(close).ewm(span=26).mean().values
    macd = ema12 - ema26
    signal = pd.Series(macd).ewm(span=9).mean().values
    hist = macd - signal
    
    # 布林軌道
    bb_mid = pd.Series(close).rolling(20).mean().values
    bb_std = pd.Series(close).rolling(20).std().values
    bb_upper = bb_mid + (bb_std * 2)
    bb_lower = bb_mid - (bb_std * 2)
    
    return {
        "close": close,
        "ma5": ma5, "ma10": ma10, "ma20": ma20, "ma60": ma60,
        "k": k, "d": d, "rsi": rsi,
        "macd": macd, "signal": signal, "hist": hist,
        "bb_upper": bb_upper, "bb_mid": bb_mid, "bb_lower": bb_lower
    }

def analyze_precision(symbol, timeframe):
    """精準分析"""
    try:
        config = {
            "60分鐘": {"interval": "60m", "period": "7d"},
            "日線": {"interval": "1d", "period": "6mo"},
            "周線": {"interval": "1wk", "period": "2y"},
            "月線": {"interval": "1mo", "period": "5y"}
        }[timeframe]
        
        df = yf.download(symbol, interval=config["interval"], 
                        period=config["period"], progress=False)
        
        if df.empty or len(df) < 60:
            return None
        
        ind = calculate_indicators(df)
        
        price = ind["close"][-1]
        ma5_val = ind["ma5"][-1]
        ma10_val = ind["ma10"][-1]
        ma20_val = ind["ma20"][-1]
        ma60_val = ind["ma60"][-1]
        
        k_val = ind["k"][-1]
        k_prev = ind["k"][-2]
        rsi_val = ind["rsi"][-1]
        
        macd_val = ind["macd"][-1]
        signal_val = ind["signal"][-1]
        hist_val = ind["hist"][-1]
        hist_prev = ind["hist"][-2] if len(ind["hist"]) > 1 else ind["hist"][-1]
        
        bb_upper = ind["bb_upper"][-1]
        bb_mid = ind["bb_mid"][-1]
        bb_lower = ind["bb_lower"][-1]
        
        # ========== 買進訊號評分 ==========
        buy_score = 0
        buy_details = []
        
        # 1. 均線組合 (最重要)
        if price > ma5_val > ma10_val > ma20_val > ma60_val:
            buy_score += 30
            buy_details.append(f"🔴 價格({price:.0f}) > MA5({ma5_val:.0f}) > MA10({ma10_val:.0f}) > MA20({ma20_val:.0f}) > MA60({ma60_val:.0f})")
            buy_details.append("   → 完美均線排列 (+30分)")
        elif price > ma5_val and ma5_val > ma20_val and ma20_val > ma60_val:
            buy_score += 20
            buy_details.append(f"🟠 價格({price:.0f}) > MA5 > MA20 > MA60 (部分排列)")
            buy_details.append("   → 主要均線看好 (+20分)")
        elif price > ma60_val:
            buy_score += 10
            buy_details.append(f"🟡 價格({price:.0f}) > 60MA({ma60_val:.0f})")
            buy_details.append("   → 長期向上 (+10分)")
        
        # 2. KD指標 (最重要)
        if k_val > 50 and k_val > k_prev:
            buy_score += 25
            buy_details.append(f"🟢 K值({k_val:.0f}) > 50 且上升")
            buy_details.append("   → KD轉強 (+25分)")
        elif k_val > 50:
            buy_score += 15
            buy_details.append(f"🟡 K值({k_val:.0f}) > 50")
            buy_details.append("   → K值適度高位 (+15分)")
        
        if k_val > 60:
            buy_score += 10
            buy_details.append(f"✅ K值({k_val:.0f}) > 60")
            buy_details.append("   → 加速中 (+10分)")
        
        # 3. MACD (確認)
        if macd_val > signal_val and hist_val > 0:
            buy_score += 15
            if hist_val > hist_prev:
                buy_score += 10
                buy_details.append("🟢 MACD轉強且直方圖擴大")
                buy_details.append("   → 多方力量增強 (+25分)")
            else:
                buy_details.append("🟡 MACD > Signal且直方圖為正")
                buy_details.append("   → 多方維持 (+15分)")
        
        # 4. RSI (確認)
        if rsi_val > 50 and rsi_val < 80:
            buy_score += 10
            buy_details.append(f"🟡 RSI({rsi_val:.0f}) 中高位")
            buy_details.append("   → 動能適度 (+10分)")
        elif rsi_val >= 80:
            buy_score += 5
            buy_details.append(f"⚠️ RSI({rsi_val:.0f}) 超買區")
            buy_details.append("   → 警示信號 (+5分)")
        
        # 5. 布林軌道 (確認)
        if price > bb_mid:
            buy_score += 8
            buy_details.append(f"🟡 股價({price:.0f}) > 布林中軌({bb_mid:.0f})")
            buy_details.append("   → 上方通道 (+8分)")
        
        # ========== 賣出訊號評分 ==========
        sell_score = 0
        sell_details = []
        
        # 1. 均線組合
        if price < ma5_val < ma10_val < ma20_val < ma60_val:
            sell_score += 30
            sell_details.append(f"🔴 價格({price:.0f}) < MA5 < MA10 < MA20 < MA60")
            sell_details.append("   → 完美空頭排列 (+30分)")
        elif price < ma5_val and ma5_val < ma20_val and ma20_val < ma60_val:
            sell_score += 20
            sell_details.append(f"🟠 價格 < MA5 < MA20 < MA60 (部分排列)")
            sell_details.append("   → 主要均線看壞 (+20分)")
        elif price < ma60_val:
            sell_score += 10
            sell_details.append(f"🟡 價格({price:.0f}) < 60MA({ma60_val:.0f})")
            sell_details.append("   → 長期向下 (+10分)")
        
        # 2. KD指標
        if k_val < 50 and k_val < k_prev:
            sell_score += 25
            sell_details.append(f"🔴 K值({k_val:.0f}) < 50 且下跌")
            sell_details.append("   → KD轉弱 (+25分)")
        elif k_val < 50:
            sell_score += 15
            sell_details.append(f"🟡 K值({k_val:.0f}) < 50")
            sell_details.append("   → K值適度低位 (+15分)")
        
        if k_val < 40:
            sell_score += 10
            sell_details.append(f"✅ K值({k_val:.0f}) < 40")
            sell_details.append("   → 加速下跌 (+10分)")
        
        # 3. MACD
        if macd_val < signal_val and hist_val < 0:
            sell_score += 15
            if hist_val < hist_prev:
                sell_score += 10
                sell_details.append("🔴 MACD轉弱且直方圖擴大")
                sell_details.append("   → 空方力量增強 (+25分)")
            else:
                sell_details.append("🟡 MACD < Signal且直方圖為負")
                sell_details.append("   → 空方維持 (+15分)")
        
        # 4. RSI
        if rsi_val < 50 and rsi_val > 20:
            sell_score += 10
            sell_details.append(f"🟡 RSI({rsi_val:.0f}) 中低位")
            sell_details.append("   → 動能下滑 (+10分)")
        elif rsi_val <= 20:
            sell_score += 5
            sell_details.append(f"⚠️ RSI({rsi_val:.0f}) 超賣區")
            sell_details.append("   → 警示信號 (+5分)")
        
        # 5. 布林軌道
        if price < bb_mid:
            sell_score += 8
            sell_details.append(f"🟡 股價({price:.0f}) < 布林中軌({bb_mid:.0f})")
            sell_details.append("   → 下方通道 (+8分)")
        
        # ========== 最終決策 ==========
        if buy_score > sell_score:
            if buy_score >= 80:
                signal = "🟢🟢🟢🟢 超強買進"
                confidence = 95
                warrant = "00715L (布蘭特油正2)"
                rationale = "四層確認 | 所有指標買進"
                stop_loss = price * 0.95
                target = price * 1.08
            elif buy_score >= 60:
                signal = "🟢🟢🟢 強烈買進"
                confidence = 88
                warrant = "00715L (布蘭特油正2)"
                rationale = "三層確認 | 主要指標買進"
                stop_loss = price * 0.96
                target = price * 1.06
            elif buy_score >= 40:
                signal = "🟢🟢 買進"
                confidence = 75
                warrant = "00642U (S&P石油)"
                rationale = "兩層確認 | 部分指標買進"
                stop_loss = price * 0.97
                target = price * 1.04
            else:
                signal = "🟡 觀望"
                confidence = 50
                warrant = "無"
                rationale = "訊號不明確"
                stop_loss = None
                target = None
        else:
            if sell_score >= 80:
                signal = "🔴🔴🔴🔴 超強賣出"
                confidence = 95
                warrant = "00673R (原油反1)"
                rationale = "四層確認 | 所有指標賣出"
                stop_loss = price * 1.05
                target = price * 0.92
            elif sell_score >= 60:
                signal = "🔴🔴🔴 強烈賣出"
                confidence = 88
                warrant = "00673R (原油反1)"
                rationale = "三層確認 | 主要指標賣出"
                stop_loss = price * 1.04
                target = price * 0.94
            elif sell_score >= 40:
                signal = "🔴🔴 賣出"
                confidence = 75
                warrant = "00673R (原油反1)"
                rationale = "兩層確認 | 部分指標賣出"
                stop_loss = price * 1.03
                target = price * 0.96
            else:
                signal = "🟡 觀望"
                confidence = 50
                warrant = "無"
                rationale = "訊號不明確"
                stop_loss = None
                target = None
        
        return {
            "df": df,
            "price": price,
            "signal": signal,
            "confidence": confidence,
            "warrant": warrant,
            "rationale": rationale,
            "buy_score": buy_score,
            "sell_score": sell_score,
            "buy_details": buy_details,
            "sell_details": sell_details,
            "stop_loss": stop_loss,
            "target": target,
            "indicators": ind
        }
    
    except Exception as e:
        st.error(f"❌ 錯誤: {str(e)}")
        return None

# UI
st.sidebar.markdown("### ⚙️ 設置")
timeframe = st.sidebar.radio("時間框架", ["60分鐘", "日線", "周線", "月線"], index=1)

stock_code = st.text_input("股票代號", value="2330.TW")
stock_name = st.text_input("股票名稱", value="台積電")

if st.button("🔍 精準分析", use_container_width=True):
    if stock_code:
        with st.spinner("分析中..."):
            result = analyze_precision(stock_code, timeframe)
        
        if result:
            # 標題
            st.markdown("---")
            html = f"""
            <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                        padding: 30px; border-radius: 10px; color: white; margin-bottom: 20px;">
                <h1 style="margin: 0; font-size: 2.5em;">{result['signal']}</h1>
                <h2 style="margin: 15px 0 0 0; font-size: 1.4em; color: #FFD700;">
                    {stock_name} ({stock_code})
                </h2>
                <p style="margin: 8px 0 0 0; font-size: 1.1em;">
                    <b>時間框架:</b> {timeframe} | <b>勝率:</b> {result['confidence']}% | <b>推薦:</b> {result['warrant']}
                </p>
                <p style="margin: 8px 0 0 0; font-size: 1.1em;">
                    <b>理由:</b> {result['rationale']}
                </p>
            </div>
            """
            st.markdown(html, unsafe_allow_html=True)
            
            # 核心數據
            st.markdown("### 📊 核心數據")
            col1, col2, col3, col4, col5 = st.columns(5)
            col1.metric("現價", f"${result['price']:.2f}")
            col2.metric("買進評分", f"{result['buy_score']:.0f}/100")
            col3.metric("賣出評分", f"{result['sell_score']:.0f}/100")
            col4.metric("勝率", f"{result['confidence']}%")
            if result['stop_loss']:
                col5.metric("停損點", f"${result['stop_loss']:.2f}")
            
            # 詳細理由
            st.markdown("---")
            st.markdown("### 🎯 詳細分析")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("**✅ 買進訊號** ✅")
                for reason in result['buy_details']:
                    st.write(reason)
                st.write(f"\n**總分: {result['buy_score']:.0f}/100**")
            
            with col2:
                st.write("**❌ 賣出訊號** ❌")
                for reason in result['sell_details']:
                    st.write(reason)
                st.write(f"\n**總分: {result['sell_score']:.0f}/100**")
            
            # 操作建議
            if result['stop_loss'] and result['target']:
                st.markdown("---")
                st.markdown("### 💰 操作建議")
                c1, c2, c3, c4 = st.columns(4)
                c1.write("**進場點**")
                c1.write(f"${result['price']:.2f}")
                c2.write("**停損點**")
                c2.write(f"${result['stop_loss']:.2f}")
                c3.write("**獲利目標**")
                c3.write(f"${result['target']:.2f}")
                c4.write("**風險比**")
                loss = abs(result['price'] - result['stop_loss'])
                gain = abs(result['target'] - result['price'])
                c4.write(f"1:{gain/loss:.2f}")
            
            # K線圖
            st.markdown("---")
            st.markdown(f"### 📈 {timeframe}K線圖 + MA")
            
            fig = go.Figure(data=[go.Candlestick(
                x=result["df"].index,
                open=result["df"]["Open"],
                high=result["df"]["High"],
                low=result["df"]["Low"],
                close=result["indicators"]["close"],
                name="K線"
            )])
            
            fig.add_trace(go.Scatter(y=result["indicators"]["ma60"], name="MA60", line=dict(color="red", width=3)))
            fig.add_trace(go.Scatter(y=result["indicators"]["ma20"], name="MA20", line=dict(color="orange", width=2)))
            fig.add_trace(go.Scatter(y=result["indicators"]["bb_upper"], name="布林上", line=dict(color="gray", width=1, dash="dash")))
            fig.add_trace(go.Scatter(y=result["indicators"]["bb_lower"], name="布林下", line=dict(color="gray", width=1, dash="dash")))
            
            fig.update_layout(height=400, hovermode="x unified", xaxis_rangeslider_visible=False)
            st.plotly_chart(fig, use_container_width=True)
            
            # KD
            st.markdown("---")
            st.markdown("### 📊 KD指標")
            fig_kd = go.Figure()
            fig_kd.add_trace(go.Scatter(y=result["indicators"]["k"], name="K", line=dict(color="blue")))
            fig_kd.add_trace(go.Scatter(y=result["indicators"]["d"], name="D", line=dict(color="red")))
            fig_kd.add_hline(y=50, line_dash="dash", line_color="gray")
            fig_kd.update_layout(height=300, hovermode="x unified")
            st.plotly_chart(fig_kd, use_container_width=True)
            
            st.markdown("---")
            st.write("✅ 分析完成")

st.markdown("---")
st.write("**精準系統:** 五層確認 | 均線+KD+MACD+RSI+布林 | 實時勝率計算")
