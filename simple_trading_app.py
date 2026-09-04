import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.graph_objects as go
from datetime import datetime

st.set_page_config(page_title="Stock Signal", page_icon="📈", layout="centered")

st.title("📈 股票買賣訊號助手")
st.write("輸入股票代號 → 立即獲得買入/賣出訊息")

def calculate_kd(close, high, low, period=9):
    lowest_low = pd.Series(low).rolling(window=period).min().values
    highest_high = pd.Series(high).rolling(window=period).max().values
    fastk = 100 * (close - lowest_low) / (highest_high - lowest_low + 1e-10)
    k = pd.Series(fastk).rolling(window=3).mean().values
    d = pd.Series(k).rolling(window=3).mean().values
    return k, d

def calculate_rsi(close, period=14):
    close_series = pd.Series(close)
    delta = close_series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi.values

def calculate_macd(close, fast=12, slow=26, signal=9):
    close_series = pd.Series(close)
    ema_fast = close_series.ewm(span=fast).mean()
    ema_slow = close_series.ewm(span=slow).mean()
    macd = ema_fast - ema_slow
    signal_line = macd.ewm(span=signal).mean()
    hist = macd - signal_line
    return macd.values, signal_line.values, hist.values

def generate_signal(symbol, name):
    try:
        st.info(f"正在分析 {name} ({symbol})...")
        df = yf.download(symbol, period='3mo', progress=False)
        
        if df.empty or len(df) < 30:
            st.error(f"無法獲取 {symbol} 的數據")
            return None
        
        close = df['Close'].values
        high = df['High'].values
        low = df['Low'].values
        
        k, d = calculate_kd(close, high, low)
        rsi = calculate_rsi(close)
        macd, signal_line, hist = calculate_macd(close)
        
        buy_votes = 0
        sell_votes = 0
        reasons = []
        
        if k[-1] < 20:
            buy_votes += 1
            reasons.append("KD < 20 (超賣)")
        elif k[-1] > 80:
            sell_votes += 1
            reasons.append("KD > 80 (超買)")
        
        if rsi[-1] < 30:
            buy_votes += 1
            reasons.append("RSI < 30 (超賣)")
        elif rsi[-1] > 70:
            sell_votes += 1
            reasons.append("RSI > 70 (超買)")
        
        if len(macd) > 1 and macd[-2] < signal_line[-2] and macd[-1] > signal_line[-1]:
            buy_votes += 1
            reasons.append("MACD 黃金交叉")
        elif len(macd) > 1 and macd[-2] > signal_line[-2] and macd[-1] < signal_line[-1]:
            sell_votes += 1
            reasons.append("MACD 死亡交叉")
        
        if buy_votes > sell_votes:
            final_signal = "BUY (買入)"
            signal_type = "buy"
        elif sell_votes > buy_votes:
            final_signal = "SELL (賣出)"
            signal_type = "sell"
        else:
            final_signal = "HOLD (觀望)"
            signal_type = "hold"
        
        current_price = close[-1]
        prev_price = close[-5] if len(close) > 5 else close[0]
        change = ((current_price - prev_price) / prev_price * 100)
        
        return {
            'signal': final_signal,
            'signal_type': signal_type,
            'price': current_price,
            'change': change,
            'kd': (k[-1], d[-1]),
            'rsi': rsi[-1],
            'macd': (macd[-1], signal_line[-1]),
            'reasons': reasons,
            'k': k,
            'd': d,
            'rsi_values': rsi,
            'macd_values': macd,
            'signal_line': signal_line
        }
    
    except Exception as e:
        st.error(f"出錯: {str(e)}")
        return None

col1, col2 = st.columns(2)

with col1:
    stock_code = st.text_input("股票代號", value="2330.TW")

with col2:
    stock_name = st.text_input("股票名稱", value="台積電")

if st.button("分析", use_container_width=True):
    if stock_code and stock_name:
        result = generate_signal(stock_code, stock_name)
        
        if result:
            st.markdown("---")
            
            if result['signal_type'] == 'buy':
                st.success(f"🟢 {result['signal']} - 建議買進")
            elif result['signal_type'] == 'sell':
                st.error(f"🔴 {result['signal']} - 建議賣出")
            else:
                st.warning(f"⚪ {result['signal']} - 繼續觀望")
            
            st.markdown("---")
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("現價", f"${result['price']:.2f}")
            
            with col2:
                st.metric("漲跌", f"{result['change']:+.2f}%")
            
            with col3:
                st.metric("時間", datetime.now().strftime("%H:%M:%S"))
            
            st.markdown("---")
            st.subheader("技術指標")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.write("KD 隨機指標")
                st.write(f"K值: {result['kd'][0]:.2f}")
                st.write(f"D值: {result['kd'][1]:.2f}")
            
            with col2:
                st.write("RSI")
                st.write(f"RSI(14): {result['rsi']:.2f}")
            
            with col3:
                st.write("MACD")
                st.write(f"MACD: {result['macd'][0]:.4f}")
                st.write(f"信號: {result['macd'][1]:.4f}")
            
            st.markdown("---")
            st.subheader("訊號原因")
            for reason in result['reasons']:
                st.write(f"• {reason}")
            
            st.markdown("---")
            fig_kd = go.Figure()
            fig_kd.add_trace(go.Scatter(y=result['k'][-60:], name='K值'))
            fig_kd.add_trace(go.Scatter(y=result['d'][-60:], name='D值'))
            fig_kd.update_layout(title="KD 指標", height=300)
            st.plotly_chart(fig_kd, use_container_width=True)
            
            fig_rsi = go.Figure()
            fig_rsi.add_trace(go.Scatter(y=result['rsi_values'][-60:], name='RSI'))
            fig_rsi.update_layout(title="RSI 指標", height=300)
            st.plotly_chart(fig_rsi, use_container_width=True)
            
            st.warning("重要提示: 本系統僅供參考，不構成投資建議。投資有風險，請謹慎決策。")
    else:
        st.warning("請輸入股票代號和名稱")

st.markdown("---")
st.write("股票代號格式: 台股 2330.TW | 美股 AAPL | 港股 0700.HK")
