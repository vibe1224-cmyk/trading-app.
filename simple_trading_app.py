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
        
        close = df['Close'].values.flatten()
        high = df['High'].values.flatten()
        low = df['Low'].values.flatten()
        
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
            reasons.append("MACD
