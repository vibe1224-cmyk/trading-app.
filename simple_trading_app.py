"""
🤖 股票買賣訊號助手 - 超簡單版本
只需輸入股票代碼，立即獲得買入/賣出訊息
"""

import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import talib
import plotly.graph_objects as go
from datetime import datetime

# ============================================================================
# 頁面設定
# ============================================================================
st.set_page_config(
    page_title="📈 股票買賣訊號",
    page_icon="📈",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# 自訂樣式
st.markdown("""
<style>
    body { font-family: 'Arial', sans-serif; }
    .big-title { font-size: 2.5rem; font-weight: bold; text-align: center; }
    .signal-buy { 
        background-color: #d4edda; 
        padding: 20px; 
        border-radius: 10px; 
        border-left: 5px solid #28a745;
        font-size: 1.3rem;
    }
    .signal-sell { 
        background-color: #f8d7da; 
        padding: 20px; 
        border-radius: 10px; 
        border-left: 5px solid #dc3545;
        font-size: 1.3rem;
    }
    .signal-hold { 
        background-color: #fff3cd; 
        padding: 20px; 
        border-radius: 10px; 
        border-left: 5px solid #ffc107;
        font-size: 1.3rem;
    }
    .metric-box {
        background: #f8f9fa;
        padding: 15px;
        border-radius: 8px;
        margin: 10px 0;
    }
</style>
""", unsafe_allow_html=True)

# ============================================================================
# 標題
# ============================================================================
st.markdown('<div class="big-title">📈 股票買賣訊號助手</div>', unsafe_allow_html=True)
st.markdown('<div style="text-align: center; color: #666; margin-bottom: 30px;">輸入股票代號 → 立即獲得買入/賣出訊息</div>', unsafe_allow_html=True)

# ============================================================================
# 技術指標計算
# ============================================================================
def calculate_kd(close, high, low, period=9):
    """計算 KD 指標"""
    lowest_low = pd.Series(low).rolling(window=period).min().values
    highest_high = pd.Series(high).rolling(window=period).max().values
    fastk = 100 * (close - lowest_low) / (highest_high - lowest_low + 1e-10)
    k = pd.Series(fastk).rolling(window=3).mean().values
    d = pd.Series(k).rolling(window=3).mean().values
    return k, d

def calculate_rsi(close, period=14):
    """計算 RSI"""
    return talib.RSI(close, timeperiod=period)

def calculate_macd(close):
    """計算 MACD"""
    macd, signal, hist = talib.MACD(close, fastperiod=12, slowperiod=26, signalperiod=9)
    return macd, signal, hist

# ============================================================================
# 訊號生成
# ============================================================================
def generate_signal(symbol, name):
    """生成買賣訊號"""
    try:
        st.info(f"⏳ 正在分析 {name} ({symbol})...")
        
        # 獲取數據
        df = yf.download(symbol, period='3mo', progress=False)
        
        if df.empty or len(df) < 30:
            st.error(f"❌ 無法獲取 {symbol} 的數據，請檢查代號是否正確")
            return None
        
        close = df['Close'].values
        high = df['High'].values
        low = df['Low'].values
        
        # 計算指標
        k, d = calculate_kd(close, high, low)
        rsi = calculate_rsi(close)
        macd, signal_line, hist = calculate_macd(close)
        
        # 生成訊號
        buy_votes = 0
        sell_votes = 0
        reasons = []
        
        # KD 訊號
        if k[-1] < 20:
            buy_votes += 1
            reasons.append("✅ KD < 20 (超賣區域)")
        elif k[-1] > 80:
            sell_votes += 1
            reasons.append("⚠️ KD > 80 (超買區域)")
        
        # RSI 訊號
        if rsi[-1] < 30:
            buy_votes += 1
            reasons.append("✅ RSI < 30 (超賣)")
        elif rsi[-1] > 70:
            sell_votes += 1
            reasons.append("⚠️ RSI > 70 (超買)")
        
        # MACD 訊號
        if len(macd) > 1 and macd[-2] < signal_line[-2] and macd[-1] > signal_line[-1]:
            buy_votes += 1
            reasons.append("✅ MACD 黃金交叉")
        elif len(macd) > 1 and macd[-2] > signal_line[-2] and macd[-1] < signal_line[-1]:
            sell_votes += 1
            reasons.append("⚠️ MACD 死亡交叉")
        
        # 決定最終訊號
        if buy_votes > sell_votes:
            final_signal = "🟢 BUY (買入)"
            signal_type = "buy"
        elif sell_votes > buy_votes:
            final_signal = "🔴 SELL (賣出)"
            signal_type = "sell"
        else:
            final_signal = "⚪ HOLD (觀望)"
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
            'df': df,
            'k': k,
            'd': d,
            'rsi_values': rsi,
            'macd_values': macd,
            'signal_line': signal_line
        }
    
    except Exception as e:
        st.error(f"❌ 出錯: {str(e)}")
        return None

# ============================================================================
# 主程序
# ============================================================================

col1, col2 = st.columns(2)

with col1:
    stock_code = st.text_input(
        "📌 股票代號",
        value="2330.TW",
        placeholder="例: 2330.TW (台積電)",
        help="台股: XXXX.TW | 美股: XXXX | 港股: XXXX.HK"
    )

with col2:
    stock_name = st.text_input(
        "📝 股票名稱",
        value="台積電",
        placeholder="例: 台積電",
        help="自訂顯示的股票名稱"
    )

# 分析按鈕
if st.button("🔍 分析", use_container_width=True, type="primary"):
    if stock_code and stock_name:
        result = generate_signal(stock_code, stock_name)
        
        if result:
            # ============================================================================
            # 顯示結果
            # ============================================================================
            
            # 主要訊號
            st.markdown("---")
            
            if result['signal_type'] == 'buy':
                st.markdown(
                    f'<div class="signal-buy"><b>{result["signal"]}</b> - 建議買進</div>',
                    unsafe_allow_html=True
                )
            elif result['signal_type'] == 'sell':
                st.markdown(
                    f'<div class="signal-sell"><b>{result["signal"]}</b> - 建議賣出</div>',
                    unsafe_allow_html=True
                )
            else:
                st.markdown(
                    f'<div class="signal-hold"><b>{result["signal"]}</b> - 繼續觀望</div>',
                    unsafe_allow_html=True
                )
            
            # 價格信息
            st.markdown("---")
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.markdown(
                    f'<div class="metric-box"><b>📊 現價</b><br>${result["price"]:.2f}</div>',
                    unsafe_allow_html=True
                )
            
            with col2:
                change_color = "🟢" if result['change'] > 0 else "🔴" if result['change'] < 0 else "⚪"
                st.markdown(
                    f'<div class="metric-box"><b>{change_color} 漲跌</b><br>{result["change"]:+.2f}%</div>',
                    unsafe_allow_html=True
                )
            
            with col3:
                st.markdown(
                    f'<div class="metric-box"><b>⏰ 更新時間</b><br>{datetime.now().strftime("%H:%M:%S")}</div>',
                    unsafe_allow_html=True
                )
            
            # 技術指標詳情
            st.markdown("---")
            st.subheader("📊 技術指標詳情")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.markdown(f"**KD 隨機指標**")
                st.markdown(f"K值: {result['kd'][0]:.2f}")
                st.markdown(f"D值: {result['kd'][1]:.2f}")
                if result['kd'][0] < 20:
                    st.markdown("🟢 **超賣** (買進信號)")
                elif result['kd'][0] > 80:
                    st.markdown("🔴 **超買** (賣出信號)")
                else:
                    st.markdown("⚪ **中性**")
            
            with col2:
                st.markdown(f"**RSI 相對強弱**")
                st.markdown(f"RSI(14): {result['rsi']:.2f}")
                if result['rsi'] < 30:
                    st.markdown("🟢 **超賣** (買進信號)")
                elif result['rsi'] > 70:
                    st.markdown("🔴 **超買** (賣出信號)")
                else:
                    st.markdown("⚪ **中性**")
            
            with col3:
                st.markdown(f"**MACD 指標**")
                st.markdown(f"MACD: {result['macd'][0]:.4f}")
                st.markdown(f"訊號線: {result['macd'][1]:.4f}")
                if result['macd'][0] > result['macd'][1]:
                    st.markdown("🟢 **上升** (買進訊號)")
                else:
                    st.markdown("🔴 **下降** (賣出訊號)")
            
            # 訊號原因
            st.markdown("---")
            st.subheader("🎯 訊號原因")
            for reason in result['reasons']:
                st.markdown(f"• {reason}")
            
            # 圖表
            st.markdown("---")
            st.subheader("📈 技術指標圖表")
            
            # KD 圖表
            fig_kd = go.Figure()
            fig_kd.add_trace(go.Scatter(y=result['k'][-60:], name='K值', line=dict(color='blue')))
            fig_kd.add_trace(go.Scatter(y=result['d'][-60:], name='D值', line=dict(color='orange')))
            fig_kd.add_hline(y=80, line_dash="dash", line_color="red", annotation_text="超買(80)")
            fig_kd.add_hline(y=20, line_dash="dash", line_color="green", annotation_text="超賣(20)")
            fig_kd.update_layout(
                title="KD 隨機指標 (最近 60 日)",
                xaxis_title="日期",
                yaxis_title="KD 值",
                height=300,
                hovermode='x unified'
            )
            st.plotly_chart(fig_kd, use_container_width=True)
            
            # RSI 圖表
            fig_rsi = go.Figure()
            fig_rsi.add_trace(go.Scatter(y=result['rsi_values'][-60:], name='RSI', line=dict(color='green')))
            fig_rsi.add_hline(y=70, line_dash="dash", line_color="red", annotation_text="超買(70)")
            fig_rsi.add_hline(y=30, line_dash="dash", line_color="green", annotation_text="超賣(30)")
            fig_rsi.update_layout(
                title="RSI 相對強弱指標 (最近 60 日)",
                xaxis_title="日期",
                yaxis_title="RSI 值",
                height=300,
                hovermode='x unified'
            )
            st.plotly_chart(fig_rsi, use_container_width=True)
            
            # 警告
            st.markdown("---")
            st.warning("""
            ⚠️ **重要提示**
            - 本系統僅供參考，不構成投資建議
            - 技術指標可能失效，請結合基本面分析
            - 過去表現不代表未來結果
            - 投資有風險，請謹慎決策
            """)
    else:
        st.warning("⚠️ 請輸入股票代號和名稱")

# ============================================================================
# 底部說明
# ============================================================================
st.markdown("---")
st.markdown("""
### 📖 使用說明

**股票代號格式:**
- 🇹🇼 台股: `2330.TW` (台積電)、`2454.TW` (聯發科)、`0050.TW` (0050 ETF)
- 🇺🇸 美股: `AAPL` (Apple)、`MSFT` (Microsoft)
- 🇭🇰 港股: `0700.HK` (騰訊)、`0388.HK` (港交所)

**訊號說明:**
- 🟢 **BUY** = 買進訊號 (多個技術指標顯示超賣或上升趨勢)
- 🔴 **SELL** = 賣出訊號 (多個技術指標顯示超買或下降趨勢)
- ⚪ **HOLD** = 觀望訊號 (指標未有明確方向)
""")
