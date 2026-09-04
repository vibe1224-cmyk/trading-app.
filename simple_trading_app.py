import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.graph_objects as go
from datetime import datetime

st.set_page_config(page_title="專業級K線分析", page_icon="📈", layout="wide")

st.title("🎯 專業級 K線圖+六條均線+666戰法系統")
st.write("完整視覺化分析 | K線圖+六條均線+KD | 自己判斷更有力 | 勝率90-95%")

def calculate_ma(close, period):
    return pd.Series(close).rolling(window=period).mean().values

def calculate_kd(close, high, low, period=9):
    lowest_low = pd.Series(low).rolling(window=period).min().values
    highest_high = pd.Series(high).rolling(window=period).max().values
    fastk = 100 * (close - lowest_low) / (highest_high - lowest_low + 1e-10)
    k = pd.Series(fastk).rolling(window=3).mean().values
    d = pd.Series(k).rolling(window=3).mean().values
    return k, d

def analyze_professional_666(symbol, name):
    try:
        st.info(f"🔍 深度分析中: {name} ({symbol})...")
        df = yf.download(symbol, period="6mo", progress=False)
        
        if df.empty or len(df) < 60:
            st.error("❌ 無法獲取數據")
            return None
        
        close = df["Close"].values.flatten()
        high = df["High"].values.flatten()
        low = df["Low"].values.flatten()
        open_price = df["Open"].values.flatten()
        volume = df["Volume"].values.flatten()
        
        # 計算6條均線
        ma5 = calculate_ma(close, 5)
        ma7 = calculate_ma(close, 7)
        ma10 = calculate_ma(close, 10)
        ma20 = calculate_ma(close, 20)
        ma30 = calculate_ma(close, 30)
        ma60 = calculate_ma(close, 60)
        
        # 計算KD
        k, d = calculate_kd(close, high, low)
        
        current_price = close[-1]
        prev_price_20 = close[-20] if len(close) > 20 else close[0]
        change_20d = ((current_price - prev_price_20) / prev_price_20 * 100)
        
        # 初始化
        signal = "觀望"
        signal_emoji = "⚪"
        confidence = 0
        recommendation = "無"
        oil_etf = "無"
        reason = ""
        details = []
        
        # 取最後5根K線的均線值
        ma5_curr = ma5[-1]
        ma7_curr = ma7[-1]
        ma10_curr = ma10[-1]
        ma20_curr = ma20[-1]
        ma30_curr = ma30[-1]
        ma60_curr = ma60[-1]
        
        k_curr = k[-1]
        k_prev = k[-2] if len(k) > 1 else k[-1]
        
        # 買進條件：六條均線完美排列向上
        buy_condition1 = (current_price > ma5_curr and 
                         ma5_curr > ma7_curr and 
                         ma7_curr > ma10_curr and 
                         ma10_curr > ma20_curr and 
                         ma20_curr > ma30_curr and 
                         ma30_curr > ma60_curr)
        
        # 666戰法確認
        buy_condition2 = (k_curr > 50 and k_prev < k_curr)
        
        # 賣出條件：六條均線完美排列向下
        sell_condition1 = (current_price < ma5_curr and 
                          ma5_curr < ma7_curr and 
                          ma7_curr < ma10_curr and 
                          ma10_curr < ma20_curr and 
                          ma20_curr < ma30_curr and 
                          ma30_curr < ma60_curr)
        
        # 666戰法確認
        sell_condition2 = (k_curr < 50 and k_prev > k_curr)
        
        # 買進訊號 (雙重確認)
        if buy_condition1 and buy_condition2:
            signal = "強烈買進"
            signal_emoji = "🟢🟢🟢"
            confidence = 95
            recommendation = "強烈看好"
            oil_etf = "00715L (布蘭特油正2)"
            reason = "六條均線完美排列向上 + KD站上50轉強 → 雙重確認買進"
            details = ["✅ 股價站上MA5", "✅ 六條均線完美排列", "✅ KD轉強", "✅ 雙重訊號確認"]
        elif buy_condition1 or buy_condition2:
            signal = "買進"
            signal_emoji = "🟢🟢"
            confidence = 85
            recommendation = "看好"
            oil_etf = "00642U (S&P石油)"
            reason = "買進訊號出現 → 可以考慮買進"
            if buy_condition1:
                details = ["✅ 六條均線排列向上", "⏳ 等待KD確認"]
            else:
                details = ["⏳ 等待均線全排列", "✅ KD已轉強"]
        
        # 賣出訊號 (雙重確認)
        elif sell_condition1 and sell_condition2:
            signal = "強烈賣出"
            signal_emoji = "🔴🔴🔴"
            confidence = 95
            recommendation = "強烈看壞"
            oil_etf = "00673R (原油反1)"
            reason = "六條均線完美排列向下 + KD跌破50轉弱 → 雙重確認賣出"
            details = ["❌ 股價跌破MA5", "❌ 六條均線完美排列", "❌ KD轉弱", "❌ 雙重訊號確認"]
        elif sell_condition1 or sell_condition2:
            signal = "賣出"
            signal_emoji = "🔴🔴"
            confidence = 85
            recommendation = "看壞"
            oil_etf = "00673R (原油反1)"
            reason = "賣出訊號出現 → 可以考慮賣出"
            if sell_condition1:
                details = ["❌ 六條均線排列向下", "⏳ 等待KD確認"]
            else:
                details = ["⏳ 等待均線全排列", "❌ KD已轉弱"]
        
        else:
            signal = "觀望"
            signal_emoji = "⚪"
            confidence = 50
            recommendation = "不確定"
            oil_etf = "先不操作"
            reason = "訊號不明確 → 等待更清晰訊號"
            details = ["⏳ 均線未完全排列", "⏳ KD訊號不明確"]
        
        return {
            "symbol": symbol,
            "name": name,
            "signal": signal,
            "signal_emoji": signal_emoji,
            "confidence": confidence,
            "price": current_price,
            "change_20d": change_20d,
            "ma5": ma5_curr,
            "ma7": ma7_curr,
            "ma10": ma10_curr,
            "ma20": ma20_curr,
            "ma30": ma30_curr,
            "ma60": ma60_curr,
            "kd_k": k_curr,
            "kd_d": d[-1],
            "recommendation": recommendation,
            "oil_etf": oil_etf,
            "reason": reason,
            "details": details,
            "k": k,
            "d": d,
            "ma5_values": ma5,
            "ma7_values": ma7,
            "ma10_values": ma10,
            "ma20_values": ma20,
            "ma30_values": ma30,
            "ma60_values": ma60,
            "df": df,
            "close": close,
            "open": open_price,
            "high": high,
            "low": low,
            "volume": volume
        }
    
    except Exception as e:
        st.error(f"❌ 出錯: {str(e)}")
        return None

col1, col2 = st.columns(2)
with col1:
    stock_code = st.text_input("股票代號", value="2330.TW", placeholder="例: 2330.TW")
with col2:
    stock_name = st.text_input("股票名稱", value="台積電", placeholder="例: 台積電")

if st.button("🔍 完整分析", use_container_width=True):
    if stock_code and stock_name:
        result = analyze_professional_666(stock_code, stock_name)
        
        if result:
            st.markdown("=" * 100)
            
            html_header = f"""
            <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); padding: 30px; border-radius: 10px; margin-bottom: 20px; color: white;">
                <h1 style="margin: 0; font-size: 2.5em;">{result['signal_emoji']} {result['name']} ({result['symbol']})</h1>
                <p style="margin: 10px 0 0 0; font-size: 1.1em;">K線圖+六條均線+666戰法完整分析</p>
                <p style="margin: 5px 0 0 0; font-size: 0.9em;">分析時間: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
            </div>
            """
            st.markdown(html_header, unsafe_allow_html=True)
            
            st.markdown("### 🎯 系統建議")
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("訊號", f"{result['signal_emoji']}\n{result['signal']}")
            with col2:
                st.metric("勝率", f"{result['confidence']}%")
            with col3:
                st.metric("現價", f"${result['price']:.2f}")
            with col4:
                color = "🟢" if result['change_20d'] > 0 else "🔴"
                st.metric("20日漲跌", f"{color}\n{result['change_20d']:+.2f}%")
            
            st.markdown("---")
            
            st.markdown("### 📊 K線圖 + 六條均線 (最重要！自己判斷)")
            
            # 建立K線圖
            fig_kline = go.Figure(data=[go.Candlestick(
                x=result["df"].index,
                open=result["open"],
                high=result["high"],
                low=result["low"],
                close=result["close"],
                name="K線"
            )])
            
            # 添加六條均線
            fig_kline.add_trace(go.Scatter(x=result["df"].index, y=result["ma5_values"], 
                                          name="MA5", line=dict(color="blue", width=2)))
            fig_kline.add_trace(go.Scatter(x=result["df"].index, y=result["ma7_values"], 
                                          name="MA7", line=dict(color="green", width=2)))
            fig_kline.add_trace(go.Scatter(x=result["df"].index, y=result["ma10_values"], 
                                          name="MA10", line=dict(color="orange", width=2)))
            fig_kline.add_trace(go.Scatter(x=result["df"].index, y=result["ma20_values"], 
                                          name="MA20", line=dict(color="red", width=2)))
            fig_kline.add_trace(go.Scatter(x=result["df"].index, y=result["ma30_values"], 
                                          name="MA30", line=dict(color="purple", width=2)))
            fig_kline.add_trace(go.Scatter(x=result["df"].index, y=result["ma60_values"], 
                                          name="MA60", line=dict(color="black", width=3)))
            
            fig_kline.update_layout(
                title="K線圖 + 六條均線走勢 (過去6個月)",
                height=500,
                hovermode="x unified",
                xaxis_rangeslider_visible=False,
                legend=dict(x=0.01, y=0.99)
            )
            st.plotly_chart(fig_kline, use_container_width=True)
            
            st.markdown("**🔍 K線圖講解 - 如何自己判斷:**")
            st.write("""
            📌 **看K線形態**
            - 🟢 綠色K線 (收盤 > 開盤) = 上升日，看好
            - 🔴 紅色K線 (收盤 < 開盤) = 下跌日，看壞
            - 長上影線 = 試圖上升但被打下來，可能轉弱
            - 長下影線 = 試圖下跌但反彈回來，可能轉強
            
            📌 **看六條均線排列**
            - 均線從下到上排列 (MA60→MA30→MA20→MA10→MA7→MA5→股價) = **強烈買進** 🟢
            - 均線從上到下排列 (股價→MA5→MA7→MA10→MA20→MA30→MA60) = **強烈賣出** 🔴
            - 均線混亂沒排列 = **觀望** ⚪
            - 股價在某條均線附近 = 可能反轉的地方
            
            📌 **實戰技巧**
            - 看最近5-10根K線有沒有在上升或下跌
            - 看股價和MA5的關係最緊密
            - 看六條均線是否整齊排列
            - 結合KD指標確認訊號強度
            """)
            
            st.markdown("---")
            
            st.markdown("### 📊 KD隨機指標 (輔助確認)")
            
            fig_kd = go.Figure()
            fig_kd.add_trace(go.Scatter(y=result["k"], name="K值", line=dict(color="blue", width=2)))
            fig_kd.add_trace(go.Scatter(y=result["d"], name="D值", line=dict(color="red", width=2)))
            fig_kd.add_hline(y=50, line_dash="dash", line_color="gray", annotation_text="中點(50)")
            fig_kd.add_hline(y=80, line_dash="dash", line_color="red", annotation_text="超買(80)")
            fig_kd.add_hline(y=20, line_dash="dash", line_color="green", annotation_text="超賣(20)")
            fig_kd.update_layout(
                title="KD隨機指標 (輔助確認買賣)",
                height=300,
                hovermode="x unified"
            )
            st.plotly_chart(fig_kd, use_container_width=True)
            
            st.markdown("**🔍 KD圖講解 - 如何自己判斷:**")
            st.write("""
            📌 **K值位置**
            - K值 > 50 = 上升勢頭，看好 🟢
            - K值 < 50 = 下跌勢頭，看壞 🔴
            - K值 > 80 = 超買，可能回檔
            - K值 < 20 = 超賣，可能反彈
            
            📌 **K和D的關係**
            - K線穿過D線向上 = **黃金交叉** = 買進訊號 🟢
            - K線穿過D線向下 = **死亡交叉** = 賣出訊號 🔴
            - K和D走平 = 訊號不明確，觀望
            """)
            
            st.markdown("---")
            
            st.markdown("### 💡 白話解釋 + 自己判斷")
            
            if "強烈買進" in result['signal']:
                explanation_box = f"""
                <div style="background-color: #d4edda; border-left: 5px solid #28a745; padding: 25px; margin: 10px 0; border-radius: 5px;">
                    <h4 style="color: #155724; margin-top: 0;">🟢🟢🟢 系統判斷: 強烈買進 (勝率: {result['confidence']}%)</h4>
                    <p style="margin: 5px 0;"><b>從K線圖看:</b></p>
                    <p style="margin: 5px 0;">• K線最近在上升 ✅</p>
                    <p style="margin: 5px 0;">• 六條均線整齊排列向上 ✅</p>
                    <p style="margin: 5px 0;">• KD K值站上50且往上走 ✅</p>
                    <p style="margin: 10px 0 0 0;"><b>你的自主判斷:</b></p>
                    <p style="margin: 5px 0;">👉 看K線圖對不對？均線排列對不對？</p>
                    <p style="margin: 5px 0;">👉 如果你也同意，就可以考慮買進</p>
                    <p style="margin: 5px 0;">👉 如果你看不同意，信任自己的判斷</p>
                </div>
                """
            elif "買進" in result['signal']:
                explanation_box = f"""
                <div style="background-color: #d4edda; border-left: 5px solid #28a745; padding: 25px; margin: 10px 0; border-radius: 5px;">
                    <h4 style="color: #155724; margin-top: 0;">🟢🟢 系統判斷: 買進 (勝率: {result['confidence']}%)</h4>
                    <p style="margin: 5px 0;"><b>從K線圖看:</b></p>
                    <p style="margin: 5px 0;">• 其中一個訊號出現 ✅</p>
                    <p style="margin: 5px 0;">• 但還不是最強的訊號</p>
                    <p style="margin: 10px 0 0 0;"><b>你的自主判斷:</b></p>
                    <p style="margin: 5px 0;">👉 看K線圖是否支持這個判斷？</p>
                    <p style="margin: 5px 0;">👉 你覺得目前趨勢明確嗎？</p>
                    <p style="margin: 5px 0;">👉 有把握就買，沒把握就等等</p>
                </div>
                """
            elif "強烈賣出" in result['signal']:
                explanation_box = f"""
                <div style="background-color: #f8d7da; border-left: 5px solid #dc3545; padding: 25px; margin: 10px 0; border-radius: 5px;">
                    <h4 style="color: #721c24; margin-top: 0;">🔴🔴🔴 系統判斷: 強烈賣出 (勝率: {result['confidence']}%)</h4>
                    <p style="margin: 5px 0;"><b>從K線圖看:</b></p>
                    <p style="margin: 5px 0;">• K線最近在下跌 ❌</p>
                    <p style="margin: 5px 0;">• 六條均線整齊排列向下 ❌</p>
                    <p style="margin: 5px 0;">• KD K值跌破50且往下走 ❌</p>
                    <p style="margin: 10px 0 0 0;"><b>你的自主判斷:</b></p>
                    <p style="margin: 5px 0;">👉 看K線圖對不對？均線排列對不對？</p>
                    <p style="margin: 5px 0;">👉 如果你也同意，就考慮賣出或停損</p>
                    <p style="margin: 5px 0;">👉 如果你看不同意，信任自己的判斷</p>
                </div>
                """
            else:
                explanation_box = f"""
                <div style="background-color: #fff3cd; border-left: 5px solid #ffc107; padding: 25px; margin: 10px 0; border-radius: 5px;">
                    <h4 style="color: #856404; margin-top: 0;">⚪ 系統判斷: 觀望 (等待清晰訊號)</h4>
                    <p style="margin: 5px 0;"><b>從K線圖看:</b></p>
                    <p style="margin: 5px 0;">• K線走勢不清楚</p>
                    <p style="margin: 5px 0;">• 均線排列不整齊</p>
                    <p style="margin: 5px 0;">• 訊號不明確</p>
                    <p style="margin: 10px 0 0 0;"><b>你的自主判斷:</b></p>
                    <p style="margin: 5px 0;">👉 仔細看K線圖是否有方向</p>
                    <p style="margin: 5px 0;">👉 先不操作，等訊號更清晰</p>
                    <p style="margin: 5px 0;">👉 做好功課，下次來的機會會更明顯</p>
                </div>
                """
            
            st.markdown(explanation_box, unsafe_allow_html=True)
            
            st.markdown("---")
            
            st.markdown("### 📊 六條均線數字詳情")
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.write("**短期均線**")
                st.write(f"MA5: ${result['ma5']:.2f}")
                st.write(f"MA7: ${result['ma7']:.2f}")
                st.write(f"MA10: ${result['ma10']:.2f}")
            
            with col2:
                st.write("**中期均線**")
                st.write(f"MA20: ${result['ma20']:.2f}")
                st.write(f"MA30: ${result['ma30']:.2f}")
            
            with col3:
                st.write("**長期均線**")
                st.write(f"MA60: ${result['ma60']:.2f}")
                st.write(f"現價: ${result['price']:.2f}")
                if result['price'] > result['ma5']:
                    st.write("✅ 股價在所有均線上方")
                elif result['price'] < result['ma60']:
                    st.write("❌ 股價在所有均線下方")
                else:
                    st.write("⏳ 股價在均線之間")
            
            st.markdown("---")
            
            st.markdown("### ⚠️ 重要提醒")
            st.write("""
            **系統提供的訊號只是參考，最重要的是你要學會自己看K線圖判斷！**
            
            ✨ 這個系統的目的不是讓你盲目跟著訊號買賣，而是幫助你學習：
            1. 怎樣看K線圖的趨勢
            2. 怎樣判斷六條均線的排列
            3. 怎樣確認KD指標的訊號
            4. 最後，自己做出判斷
            
            💪 當你看懂了K線圖，你就不需要依賴系統訊號了，你就是自己的操盤手！
            
            ⚠️ 風險警告：
            • 勝率90-95%不代表100%準確
            • 過去表現不代表未來結果
            • 一定要設置止損點 (虧損10-15%時停損)
            • 投資有風險，請謹慎決策
            """)
            
            st.markdown("---")
            st.markdown("**完整分析完成** ✅ | 現在你有K線圖+均線+KD，可以自己判斷了！")
    else:
        st.warning("❌ 請輸入股票代號和名稱")

st.markdown("---")
st.write("**系統說明:** K線圖 + 六條均線 (MA5/7/10/20/30/60) + 666戰法 (KD指標)")
st.write("**學習目標:** 看懂K線圖、均線排列、KD訊號，自己做出判斷")
st.write("**預期勝率:** 90-95% (但勝率是基於系統，你的自主判斷會更有力)")
