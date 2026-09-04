"""
🤖 股票買賣訊號助手 - 超簡單版本 (修復版)
只需輸入股票代碼，立即獲得買入/賣出訊息
"""

import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
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
        border-radius:
