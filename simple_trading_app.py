import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.graph_objects as go
from datetime import datetime, timedelta
import feedparser
import requests

st.set_page_config(page_title="666戰法+新聞系統", page_icon="📈", layout="wide")

st.title("🎯 666戰法 + 新聞事件驅動系統")
st.write("60分鐘|日線|周線|月線 + 美國新聞監控 + 台股/原油自動建議 + 詳細理由")

def get_latest_news():
    """獲取最近的新聞"""
    try:
        # 使用Google News RSS feed
        url = "https://news.google.com/rss/search?q=美國 台灣 貿易 戰爭 關稅 股市&hl=zh-TW&gl=TW&ceid=TW:zh-Hant"
        
        feed = feedparser.parse(url)
        news_list = []
        
        for entry in feed.entries[:10]:  # 取最近10條新聞
            news_list.append({
                "title": entry.title,
                "published": entry.published,
                "link": entry.link
            })
        
        return news_list
    except:
        return []

def analyze_news_sentiment(news_list):
    """分析新聞情緒對台股和原油的影響"""
    
    # 定義關鍵詞
    negative_keywords = ["戰爭", "打擊", "制裁", "下跌", "崩盤", "危機", "風險", "衝擊", "暴跌"]
    positive_keywords = ["上升", "強勁", "利好", "反彈", "復甦", "增長", "機會"]
    oil_related = ["原油", "能源", "石油", "俄烏", "中東", "伊朗", "沙特"]
    tech_related = ["科技", "晶片", "半導體", "台積電", "電子", "AI", "算力"]
    trade_related = ["貿易", "關稅", "進口", "出口", "美中", "美國", "關係"]
    
    analysis = {
        "tw_stock_sentiment": "中性",
        "oil_sentiment": "中性",
        "major_events": [],
        "recommended_action": "觀望",
        "reasons": [],
        "news_summary": []
    }
    
    tw_positive = 0
    tw_negative = 0
    oil_positive = 0
    oil_negative = 0
    
    for news in news_list:
        title = news["title"]
        analysis["news_summary"].append(title)
        
        # 檢查負面詞
        for word in negative_keywords:
            if word in title:
                tw_negative += 2
                oil_positive += 1  # 原油在危機時上升
        
        # 檢查正面詞
        for word in positive_keywords:
            if word in title:
                tw_positive += 2
                oil_negative += 1
        
        # 檢查原油相關
        if any(word in title for word in oil_related):
            if any(word in title for word in negative_keywords):
                oil_positive += 2
            if any(word in title for word in positive_keywords):
                oil_negative += 2
        
        # 檢查科技相關
        if any(word in title for word in tech_related):
            if any(word in title for word in negative_keywords):
                tw_negative += 1
            if any(word in title for word in positive_keywords):
                tw_positive += 1
        
        # 檢查貿易相關
        if any(word in title for word in trade_related):
            if "打擊" in title or "制裁" in title or "關稅" in title:
                tw_negative += 2
                analysis["major_events"].append(f"⚠️ 貿易風險: {title[:50]}")
            if "協議" in title or "好轉" in title:
                tw_positive += 2
                analysis["major_events"].append(f"✅ 貿易利好: {title[:50]}")
    
    # 判斷台股情緒
    if tw_negative > tw_positive + 2:
        analysis["tw_stock_sentiment"] = "看壞"
        analysis["reasons"].append(f"🔴 新聞偏負面 (負面:{tw_negative} vs 正面:{tw_positive})")
    elif tw_positive > tw_negative + 2:
        analysis["tw_stock_sentiment"] = "看好"
        analysis["reasons"].append(f"🟢 新聞偏正面 (正面:{tw_positive} vs 負面:{tw_negative})")
    else:
        analysis["tw_stock_sentiment"] = "中性"
        analysis["reasons"].append(f"⚪ 新聞平衡 (正面:{tw_positive} vs 負面:{tw_negative})")
    
    # 判斷原油情緒
    if oil_positive > oil_negative + 2:
        analysis["oil_sentiment"] = "看漲"
        analysis["reasons"].append(f"📈 原油偏漲 (利好:{oil_positive})")
    elif oil_negative > oil_positive + 2:
        analysis["oil_sentiment"] = "看跌"
        analysis["reasons"].append(f"📉 原油偏跌 (利空:{oil_negative})")
    else:
        analysis["oil_sentiment"] = "中性"
        analysis["reasons"].append(f"⚪ 原油平衡")
    
    # 推薦操作
    if analysis["tw_stock_sentiment"] == "看好" and analysis["oil_sentiment"] == "看漲":
        analysis["recommended_action"] = "🟢 買台股 + 原油"
        analysis["suggested_warrant"] = "00715L (布蘭特油正2)"
    elif analysis["tw_stock_sentiment"] == "看壞" and analysis["oil_sentiment"] == "看跌":
        analysis["recommended_action"] = "🔴 看空台股 + 原油"
        analysis["suggested_warrant"] = "00673R (原油反1)"
    elif analysis["tw_stock_sentiment"] == "看壞" and analysis["oil_sentiment"] == "看漲":
        analysis["recommended_action"] = "⚖️ 台股看壞 + 原油看漲"
        analysis["suggested_warrant"] = "00715L (布蘭特油正2)"
        analysis["reasons"].append("💡 建議買原油對沖台股風險")
    elif analysis["tw_stock_sentiment"] == "看好" and analysis["oil_sentiment"] == "看跌":
        analysis["recommended_action"] = "⚖️ 台股看好 + 原油看跌"
        analysis["suggested_warrant"] = "0050 (台灣50)"
        analysis["reasons"].append("💡 建議買台股，原油風險降低")
    else:
        analysis["recommended_action"] = "⏳ 觀望"
        analysis["suggested_warrant"] = "無"
    
    return analysis

def calculate_ma(close, period):
    return pd.Series(close).rolling(window=period).mean().values

def calculate_kd_custom(close, high, low, k_period=60, d_period=3):
    """KD(60,3,3)參數"""
    lowest_low = pd.Series(low).rolling(window=k_period).min().values
    highest_high = pd.Series(high).rolling(window=k_period).max().values
    fastk = 100 * (close - lowest_low) / (highest_high - lowest_low + 1e-10)
    k = pd.Series(fastk).rolling(window=d_period).mean().values
    d = pd.Series(k).rolling(window=d_period).mean().values
    return k, d

def calculate_macd(close, fast=12, slow=26, signal=9):
    """計算MACD"""
    ema_fast = pd.Series(close).ewm(span=fast).mean().values
    ema_slow = pd.Series(close).ewm(span=slow).mean().values
    macd_line = ema_fast - ema_slow
    signal_line = pd.Series(macd_line).ewm(span=signal).mean().values
    histogram = macd_line - signal_line
    return macd_line, signal_line, histogram

def analyze_666_strategy(symbol, name, selected_mas, timeframe):
    try:
        timeframe_config = {
            "60分鐘": {"interval": "60m", "period": "7d", "ma_period": 60},
            "日線": {"interval": "1d", "period": "6mo", "ma_period": 60},
            "周線": {"interval": "1wk", "period": "2y", "ma_period": 50},
            "月線": {"interval": "1mo", "period": "5y", "ma_period": 40}
        }
        
        config = timeframe_config[timeframe]
        st.info(f"🔍 分析中: {name} ({symbol}) - {timeframe}...")
        
        df = yf.download(symbol, interval=config["interval"], period=config["period"], progress=False)
        
        if df.empty or len(df) < config["ma_period"]:
            st.error(f"❌ 無法獲取{timeframe}數據")
            return None
        
        close = df["Close"].values.flatten()
        high = df["High"].values.flatten()
        low = df["Low"].values.flatten()
        open_price = df["Open"].values.flatten()
        
        ma_period = config["ma_period"]
        ma60 = calculate_ma(close, ma_period)
        k, d = calculate_kd_custom(close, high, low, k_period=60, d_period=3)
        macd_line, signal_line, histogram = calculate_macd(close)
        
        ma5 = calculate_ma(close, 5) if "MA5" in selected_mas else None
        ma7 = calculate_ma(close, 7) if "MA7" in selected_mas else None
        ma10 = calculate_ma(close, 10) if "MA10" in selected_mas else None
        
        current_price = close[-1]
        current_k = k[-1]
        current_d = d[-1]
        current_macd = macd_line[-1]
        current_signal = signal_line[-1]
        current_histogram = histogram[-1]
        current_ma60 = ma60[-1]
        
        k_prev = k[-2] if len(k) > 1 else k[-1]
        macd_prev = macd_line[-2] if len(macd_line) > 1 else macd_line[-1]
        signal_prev = signal_line[-2] if len(signal_line) > 1 else signal_line[-1]
        
        price_above_ma60 = current_price > current_ma60
        k_above_50 = current_k > 50
        k_turning_up = current_k > k_prev
        k_accelerating = current_k > 60
        macd_bullish = (current_macd > current_signal) and (current_histogram > 0)
        macd_turning_up = (macd_prev <= signal_prev) and (current_macd > current_signal)
        
        buy_signal_count = 0
        buy_details = []
        
        if price_above_ma60:
            buy_signal_count += 2
            buy_details.append(f"✅ 股價 > {ma_period}MA")
        
        if k_above_50 and k_turning_up:
            buy_signal_count += 2
            buy_details.append("✅ K值 > 50 且上升")
        
        if k_accelerating:
            buy_signal_count += 1
            buy_details.append("✅ K值 > 60（漲勢加速）")
        
        if macd_turning_up:
            buy_signal_count += 1
            buy_details.append("✅ MACD轉強")
        elif macd_bullish:
            buy_signal_count += 0.5
            buy_details.append("✅ MACD保持強勢")
        
        price_below_ma60 = current_price < current_ma60
        k_below_50 = current_k < 50
        k_turning_down = current_k < k_prev
        k_dropping = current_k < 40
        macd_bearish = (current_macd < current_signal) and (current_histogram < 0)
        macd_turning_down = (macd_prev >= signal_prev) and (current_macd < current_signal)
        
        sell_signal_count = 0
        sell_details = []
        
        if price_below_ma60:
            sell_signal_count += 2
            sell_details.append(f"❌ 股價 < {ma_period}MA")
        
        if k_below_50 and k_turning_down:
            sell_signal_count += 2
            sell_details.append("❌ K值 < 50 且下跌")
        
        if k_dropping:
            sell_signal_count += 1
            sell_details.append("❌ K值 < 40（跌勢加速）")
        
        if macd_turning_down:
            sell_signal_count += 1
            sell_details.append("❌ MACD轉弱")
        elif macd_bearish:
            sell_signal_count += 0.5
            sell_details.append("❌ MACD保持弱勢")
        
        if buy_signal_count >= 5:
            signal = "🟢 強烈買進"
            confidence = 90
            signal_type = "buy_strong"
        elif buy_signal_count >= 3:
            signal = "🟢 買進"
            confidence = 80
            signal_type = "buy"
        elif sell_signal_count >= 5:
            signal = "🔴 強烈賣出"
            confidence = 90
            signal_type = "sell_strong"
        elif sell_signal_count >= 3:
            signal = "🔴 賣出"
            confidence = 80
            signal_type = "sell"
        else:
            signal = "⚪ 觀望"
            confidence = 50
            signal_type = "wait"
        
        return {
            "symbol": symbol,
            "name": name,
            "signal": signal,
            "signal_type": signal_type,
            "confidence": confidence,
            "price": current_price,
            "ma60": current_ma60,
            "k": current_k,
            "macd": current_macd,
            "buy_details": buy_details,
            "sell_details": sell_details,
            "buy_count": buy_signal_count,
            "sell_count": sell_signal_count,
            "df": df,
            "close": close,
            "open": open_price,
            "high": high,
            "low": low,
            "ma60_values": ma60,
            "ma5_values": ma5,
            "ma7_values": ma7,
            "ma10_values": ma10,
            "k_values": k,
            "d_values": d,
            "macd_values": macd_line,
            "signal_values": signal_line,
            "timeframe": timeframe
        }
    
    except Exception as e:
        st.error(f"❌ 出錯: {str(e)}")
        return None

# 側邊欄設置
st.sidebar.markdown("### ⚙️ 系統設置")

show_news = st.sidebar.checkbox("✅ 顯示新聞監控", value=True)

timeframe = st.sidebar.radio(
    "選擇時間框架",
    ["60分鐘", "日線", "周線", "月線"],
    index=1
)

selected_mas = st.sidebar.multiselect(
    "選擇要顯示的均線",
    ["MA5", "MA7", "MA10"],
    default=["MA5"]
)

col1, col2 = st.columns(2)
with col1:
    stock_code = st.text_input("股票代號", value="^TWII", placeholder="例: ^TWII (台股指數)")
with col2:
    stock_name = st.text_input("股票名稱", value="台股指數", placeholder="例: 台股指數")

# ========== 新聞監控部分 ==========
if show_news:
    st.markdown("---")
    st.markdown("### 📰 美國新聞監控 + 事件驅動分析")
    
    with st.spinner("📡 正在獲取最新新聞..."):
        news_list = get_latest_news()
        news_analysis = analyze_news_sentiment(news_list)
    
    # 新聞情緒總結
    col1, col2 = st.columns(2)
    
    with col1:
        if news_analysis["tw_stock_sentiment"] == "看好":
            html_tw = '<div style="background-color: #d4edda; padding: 15px; border-radius: 5px;"><h4 style="color: #155724; margin: 0;">🟢 台股: 看好</h4></div>'
        elif news_analysis["tw_stock_sentiment"] == "看壞":
            html_tw = '<div style="background-color: #f8d7da; padding: 15px; border-radius: 5px;"><h4 style="color: #721c24; margin: 0;">🔴 台股: 看壞</h4></div>'
        else:
            html_tw = '<div style="background-color: #fff3cd; padding: 15px; border-radius: 5px;"><h4 style="color: #856404; margin: 0;">⚪ 台股: 中性</h4></div>'
        st.markdown(html_tw, unsafe_allow_html=True)
    
    with col2:
        if news_analysis["oil_sentiment"] == "看漲":
            html_oil = '<div style="background-color: #d4edda; padding: 15px; border-radius: 5px;"><h4 style="color: #155724; margin: 0;">📈 原油: 看漲</h4></div>'
        elif news_analysis["oil_sentiment"] == "看跌":
            html_oil = '<div style="background-color: #f8d7da; padding: 15px; border-radius: 5px;"><h4 style="color: #721c24; margin: 0;">📉 原油: 看跌</h4></div>'
        else:
            html_oil = '<div style="background-color: #fff3cd; padding: 15px; border-radius: 5px;"><h4 style="color: #856404; margin: 0;">⚪ 原油: 中性</h4></div>'
        st.markdown(html_oil, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # 新聞推薦
    st.markdown("### 💡 新聞驅動建議")
    
    html_recommendation = f"""
    <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); padding: 20px; border-radius: 10px; color: white;">
        <h2 style="margin: 0; font-size: 1.5em;">{news_analysis['recommended_action']}</h2>
        <p style="margin: 10px 0 0 0;"><b>推薦權證:</b> {news_analysis['suggested_warrant']}</p>
    </div>
    """
    st.markdown(html_recommendation, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # 新聞理由
    st.markdown("### 🔍 分析理由")
    for reason in news_analysis["reasons"]:
        st.write(reason)
    
    st.markdown("---")
    
    # 主要事件
    if news_analysis["major_events"]:
        st.markdown("### ⚠️ 主要事件")
        for event in news_analysis["major_events"][:5]:
            st.write(event)
    
    st.markdown("---")
    
    # 最近新聞標題
    st.markdown("### 📡 最近新聞")
    for i, news in enumerate(news_analysis["news_summary"][:8]):
        st.write(f"{i+1}. {news}")
    
    st.markdown("---")

# ========== 666戰法分析部分 ==========
if st.button("🔍 分析", use_container_width=True):
    if stock_code and stock_name:
        result = analyze_666_strategy(stock_code, stock_name, selected_mas, timeframe)
        
        if result:
            st.markdown("---")
            
            # 標題
            html_signal = f"""
            <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); padding: 25px; border-radius: 10px; margin-bottom: 20px; color: white;">
                <h1 style="margin: 0; font-size: 2em;">{result['signal']}</h1>
                <p style="margin: 10px 0 0 0;">時間框架: {result['timeframe']} | 勝率: {result['confidence']}%</p>
            </div>
            """
            st.markdown(html_signal, unsafe_allow_html=True)
            
            st.markdown("---")
            
            # 核心數據
            st.markdown(f"### 📊 核心數據（{result['timeframe']}）")
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("現價", f"${result['price']:.0f}")
            with col2:
                st.metric("60MA", f"${result['ma60']:.0f}")
            with col3:
                st.metric("K值", f"{result['k']:.0f}")
            with col4:
                st.metric("勝率", f"{result['confidence']}%")
            
            st.markdown("---")
            
            # 買賣訊號
            st.markdown("### 🎯 666戰法訊號")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("#### ✅ 買進訊號")
                if result['buy_details']:
                    for detail in result['buy_details']:
                        st.write(detail)
                    st.write(f"**點數: {result['buy_count']:.1f}/5**")
                else:
                    st.write("無買進訊號")
            
            with col2:
                st.markdown("#### ❌ 賣出訊號")
                if result['sell_details']:
                    for detail in result['sell_details']:
                        st.write(detail)
                    st.write(f"**點數: {result['sell_count']:.1f}/5**")
                else:
                    st.write("無賣出訊號")
            
            st.markdown("---")
            
            # K線圖
            st.markdown(f"### 📈 {result['timeframe']}K線圖")
            
            fig = go.Figure(data=[go.Candlestick(
                x=result["df"].index,
                open=result["open"],
                high=result["high"],
                low=result["low"],
                close=result["close"],
                name="K線"
            )])
            
            fig.add_trace(go.Scatter(
                x=result["df"].index, y=result["ma60_values"], 
                name="60MA", line=dict(color="red", width=3)
            ))
            
            if result['ma5_values'] is not None:
                fig.add_trace(go.Scatter(
                    x=result["df"].index, y=result["ma5_values"], 
                    name="MA5", line=dict(color="blue", width=1)
                ))
            
            fig.update_layout(
                title=f"{result['timeframe']}K線 + 60MA",
                height=400,
                hovermode="x unified",
                xaxis_rangeslider_visible=False
            )
            st.plotly_chart(fig, use_container_width=True)
            
            st.markdown("---")
            st.markdown("**分析完成** ✅")
    else:
        st.warning("❌ 請輸入股票代號和名稱")

st.markdown("---")
st.write("**系統說明:** 666戰法 + 新聞監控 + 事件驅動")
st.write("**功能:** 自動判斷美國新聞對台股和原油的影響，給出買賣建議")
