import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import datetime
import matplotlib.pyplot as plt
import seaborn as sns

# ---------- SETTINGS ----------
INDEX = "^GSPC"  # S&P 500
SECTORS = {
    "Technology": "XLK",
    "Healthcare": "XLV",
    "Energy": "XLE",
    "Financials": "XLF",
    "Consumer Discretionary": "XLY"
}
CRASH_YEARS = [2008, 2020, 2022]

st.set_page_config(page_title="Stock Crash Analysis", layout="wide")

# ---------- DATA DOWNLOAD ----------
@st.cache_data
def download_data():
    tickers = [INDEX] + list(SECTORS.values())
    raw = yf.download(
        tickers,
        start="2007-01-01",
        end="2024-01-01",
        group_by="ticker",
        auto_adjust=True,
        threads=True,
    )
    return raw

df_all = download_data()

# Helpers to extract Adj Close
def get_adj_close(raw_df, ticker):
    if isinstance(raw_df.columns, pd.MultiIndex):
        return raw_df[ticker]["Close"]
    else:
        return raw_df["Close"]

# Drawdown & Recovery
def max_drawdown(series):
    roll_max = series.cummax()
    return ((series - roll_max) / roll_max).min()

def recovery_time(series):
    roll_max = series.cummax()
    drawdown = (series - roll_max) / roll_max
    trough = drawdown.idxmin()
    peak_before = series.loc[:trough].max()
    post = series.loc[trough:]
    recover_idx = post[post >= peak_before].first_valid_index()
    if recover_idx:
        return (recover_idx - trough).days
    return None

# ---------- UI ----------
st.title("Stock Market Crash Analysis Dashboard")
year = st.selectbox("Select Crash Year", CRASH_YEARS)
start = f"{year-1}-01-01"
end   = f"{year+1}-01-01"

# Slice index series
idx_ser = get_adj_close(df_all, INDEX).loc[start:end]

# --- 1. Index Plot ---
st.subheader(f"S&P 500 Performance during {year} Crash")
st.line_chart(idx_ser)

# --- 2. Summary Stats ---
st.markdown("### Summary Statistics")
dd = max_drawdown(idx_ser)
rt = recovery_time(idx_ser)
st.markdown(f"- **Max Drawdown:** {dd:.2%}")
st.markdown(f"- **Recovery Time:** {rt if rt else 'Not recovered'} days")

# --- 3. Sector Performance ---
st.markdown("### Sector Performance Comparison")
df_perf = pd.DataFrame()
for name, ticker in SECTORS.items():
    ser = get_adj_close(df_all, ticker).loc[start:end]
    df_perf[name] = ser / ser.iloc[0]

fig, ax = plt.subplots(figsize=(12, 6))
for col in df_perf:
    ax.plot(df_perf[col], label=col)
ax.set_title("Normalized Sector Performance")
ax.legend(loc="upper left")
st.pyplot(fig)

# --- 4. Volatility Analysis ---
st.markdown("### Volatility Analysis (30-day Rolling Std Dev)")
ret = idx_ser.pct_change().dropna()
vol30 = ret.rolling(window=30).std()
st.line_chart(vol30.rename("Volatility"))

# --- 5. Correlation Analysis ---
st.markdown("### Correlation with S&P 500")
corr_df = pd.DataFrame({"S&P 500": ret})
for name, ticker in SECTORS.items():
    sec_ret = get_adj_close(df_all, ticker).loc[start:end].pct_change()
    corr_df[name] = sec_ret

corr = corr_df.corr()
st.dataframe(corr.round(2), use_container_width=True)

# Optional heatmap
st.markdown("#### Heatmap of Correlations")
fig2, ax2 = plt.subplots(figsize=(6, 4))
sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", ax=ax2)
st.pyplot(fig2)
