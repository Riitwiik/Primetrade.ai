"""
Bitcoin Market Sentiment vs Hyperliquid Trader Performance Analysis
===================================================================
A single-file Streamlit application that performs end-to-end analysis
of the relationship between Bitcoin Fear & Greed Index sentiment
and Hyperliquid trader performance.

Run with:  streamlit run app.py

Dependencies:
    pip install pandas numpy matplotlib seaborn plotly scipy scikit-learn streamlit
"""

# ============================================================================
# IMPORTS
# ============================================================================

import io
import os
import warnings
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import streamlit as st

# Matplotlib setup
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

# Register fonts for potential CJK / special characters
try:
    fm.fontManager.addfont("/usr/share/fonts/truetype/chinese/NotoSansSC[wght].ttf")
except Exception:
    pass
try:
    fm.fontManager.addfont("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf")
except Exception:
    pass
plt.rcParams["font.sans-serif"] = ["DejaVu Sans", "Noto Sans SC"]
plt.rcParams["axes.unicode_minus"] = False

import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from scipy import stats
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    roc_auc_score,
    classification_report,
)

warnings.filterwarnings("ignore")

# ============================================================================
# CONSTANTS & CONFIGURATION
# ============================================================================

# Column name mappings: standardised names -> actual CSV column names
COL_ACCOUNT = "Account"
COL_SYMBOL = "Coin"
COL_EXEC_PRICE = "Execution Price"
COL_SIZE_TOKENS = "Size Tokens"
COL_SIZE_USD = "Size USD"
COL_SIDE = "Side"
COL_TIMESTAMP_IST = "Timestamp IST"
COL_START_POS = "Start Position"
COL_DIRECTION = "Direction"
COL_CLOSED_PNL = "Closed PnL"
COL_TX_HASH = "Transaction Hash"
COL_ORDER_ID = "Order ID"
COL_CROSSED = "Crossed"
COL_FEE = "Fee"
COL_TRADE_ID = "Trade ID"
COL_TIMESTAMP = "Timestamp"

# Fear & Greed columns
FG_TIMESTAMP = "timestamp"
FG_VALUE = "value"
FG_CLASSIFICATION = "classification"
FG_DATE = "date"

# Derived column names
COL_TRADE_DATE = "trade_date"
COL_PNL = "pnl"
COL_WIN_LOSS = "win_loss"
COL_LEVERAGE_BUCKET = "leverage_bucket"
COL_TRADE_SIZE = "trade_size"
COL_AVG_EXEC_PRICE = "avg_execution_price"
COL_SENTIMENT_CATEGORY = "sentiment_category"
COL_ENCODED_SENTIMENT = "encoded_sentiment"
COL_CUMULATIVE_PNL = "cumulative_pnl"
COL_ROLLING_PNL = "rolling_pnl"
COL_DAILY_PNL = "daily_pnl"
COL_DAILY_TRADE_COUNT = "daily_trade_count"
COL_AVG_LEVERAGE = "avg_leverage"
COL_AVG_PNL = "avg_pnl"
COL_ESTIMATED_LEVERAGE = "estimated_leverage"

# Sentiment encoding
SENTIMENT_ENCODING = {
    "Extreme Fear": 0,
    "Fear": 1,
    "Neutral": 2,
    "Greed": 3,
    "Extreme Greed": 4,
}

# Default file paths — searches multiple locations in priority order:
#   1. data/ folder next to app.py  (best for Streamlit Cloud — commit CSVs here)
#   2. /home/z/my-project/upload/   (current dev environment)
#   3. Same directory as app.py      (fallback)
_APP_DIR = os.path.dirname(os.path.abspath(__file__))

_DATA_SEARCH_PATHS = {
    "fg": [
        os.path.join(_APP_DIR, "data", "fear_greed_index.csv"),
        "/home/z/my-project/upload/fear_greed_index.csv",
        os.path.join(_APP_DIR, "fear_greed_index.csv"),
    ],
    "hl": [
        os.path.join(_APP_DIR, "data", "historical_data.csv"),
        "/home/z/my-project/upload/historical_data.csv",
        os.path.join(_APP_DIR, "historical_data.csv"),
    ],
}


def _find_data_file(key: str) -> Optional[str]:
    """Return the first existing file path for the given data key, or None."""
    for path in _DATA_SEARCH_PATHS.get(key, []):
        if os.path.isfile(path):
            return path
    return None


DEFAULT_FG_PATH = _find_data_file("fg")
DEFAULT_HL_PATH = _find_data_file("hl")

# Color palettes
SENTIMENT_COLORS = {
    "Extreme Fear": "#FF4444",
    "Fear": "#FF8C00",
    "Neutral": "#FFD700",
    "Greed": "#32CD32",
    "Extreme Greed": "#006400",
}

PLOTLY_TEMPLATE = "plotly_white"

# ============================================================================
# DATA LOADING FUNCTIONS
# ============================================================================


def detect_encoding(file_path: str) -> str:
    """Attempt to detect file encoding. Falls back to utf-8."""
    try:
        import chardet
        with open(file_path, "rb") as f:
            raw = f.read(10000)
            result = chardet.detect(raw)
            return result.get("encoding", "utf-8") or "utf-8"
    except ImportError:
        return "utf-8"
    except Exception:
        return "utf-8"


def load_fear_greed_data(file_path: str) -> pd.DataFrame:
    """Load and perform initial cleaning of the Fear & Greed Index CSV.

    Args:
        file_path: Path to fear_greed_index.csv

    Returns:
        Cleaned DataFrame with standardised column names.
    """
    encoding = detect_encoding(file_path)
    try:
        df = pd.read_csv(file_path, encoding=encoding)
    except UnicodeDecodeError:
        df = pd.read_csv(file_path, encoding="latin-1")

    # Standardise column names (strip whitespace, lowercase for matching)
    df.columns = [c.strip() for c in df.columns]

    # Convert date column to datetime
    if FG_DATE in df.columns:
        df[FG_DATE] = pd.to_datetime(df[FG_DATE], errors="coerce")
    elif "Date" in df.columns:
        df[FG_DATE] = pd.to_datetime(df["Date"], errors="coerce")

    # Ensure value is numeric
    if FG_VALUE in df.columns:
        df[FG_VALUE] = pd.to_numeric(df[FG_VALUE], errors="coerce")

    # Strip classification strings
    if FG_CLASSIFICATION in df.columns:
        df[FG_CLASSIFICATION] = df[FG_CLASSIFICATION].astype(str).str.strip()
    elif "Classification" in df.columns:
        df[FG_CLASSIFICATION] = df["Classification"].astype(str).str.strip()

    # Drop rows with missing critical fields
    df.dropna(subset=[FG_DATE, FG_CLASSIFICATION], inplace=True)

    # Remove duplicates
    df.drop_duplicates(subset=[FG_DATE], keep="first", inplace=True)
    df.reset_index(drop=True, inplace=True)

    return df


def load_historical_data(file_path: str) -> pd.DataFrame:
    """Load and perform initial cleaning of the Hyperliquid historical trade data.

    Args:
        file_path: Path to historical_data.csv

    Returns:
        Cleaned DataFrame.
    """
    encoding = detect_encoding(file_path)
    try:
        df = pd.read_csv(file_path, encoding=encoding, low_memory=False)
    except UnicodeDecodeError:
        df = pd.read_csv(file_path, encoding="latin-1", low_memory=False)

    # Standardise column names
    df.columns = [c.strip() for c in df.columns]

    # Convert numeric columns
    numeric_cols = [COL_EXEC_PRICE, COL_SIZE_TOKENS, COL_SIZE_USD,
                    COL_START_POS, COL_CLOSED_PNL, COL_FEE]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Drop rows with missing critical fields (account, closedPnL)
    critical_cols = [col for col in [COL_ACCOUNT, COL_CLOSED_PNL] if col in df.columns]
    if critical_cols:
        df.dropna(subset=critical_cols, inplace=True)

    # Remove exact duplicate rows
    df.drop_duplicates(inplace=True)
    df.reset_index(drop=True, inplace=True)

    return df


def parse_timestamp_ist(ts_str: str) -> Optional[pd.Timestamp]:
    """Parse Timestamp IST string in DD-MM-YYYY HH:MM format."""
    if pd.isna(ts_str):
        return pd.NaT
    try:
        return pd.to_datetime(ts_str, format="%d-%m-%Y %H:%M", errors="coerce")
    except Exception:
        return pd.NaT


def parse_timestamp_ms(ts_val) -> Optional[pd.Timestamp]:
    """Parse Unix-millisecond timestamp."""
    try:
        return pd.to_datetime(float(ts_val), unit="ms", errors="coerce")
    except Exception:
        return pd.NaT


# ============================================================================
# FEATURE ENGINEERING
# ============================================================================


def engineer_features(df: pd.DataFrame, fg_df: pd.DataFrame) -> pd.DataFrame:
    """Create all derived features and merge with sentiment data.

    Args:
        df: Cleaned Hyperliquid trade DataFrame.
        fg_df: Cleaned Fear & Greed Index DataFrame.

    Returns:
        Enriched DataFrame with all engineered features.
    """
    # --- Parse timestamps and extract trade date ---
    if COL_TIMESTAMP_IST in df.columns:
        df[COL_TIMESTAMP_IST] = df[COL_TIMESTAMP_IST].apply(parse_timestamp_ist)
        df[COL_TRADE_DATE] = df[COL_TIMESTAMP_IST].dt.date
        df[COL_TRADE_DATE] = pd.to_datetime(df[COL_TRADE_DATE])
    elif COL_TIMESTAMP in df.columns:
        df[COL_TIMESTAMP] = df[COL_TIMESTAMP].apply(parse_timestamp_ms)
        df[COL_TRADE_DATE] = df[COL_TIMESTAMP].dt.date
        df[COL_TRADE_DATE] = pd.to_datetime(df[COL_TRADE_DATE])

    # --- PnL alias ---
    if COL_CLOSED_PNL in df.columns:
        df[COL_PNL] = df[COL_CLOSED_PNL].copy()

    # --- Win / Loss flag ---
    if COL_PNL in df.columns:
        df[COL_WIN_LOSS] = np.where(df[COL_PNL] > 0, "Win",
                                     np.where(df[COL_PNL] < 0, "Loss", "Break-even"))

    # --- Estimated leverage (Size USD based percentile buckets) ---
    if COL_SIZE_USD in df.columns:
        df[COL_ESTIMATED_LEVERAGE] = df[COL_SIZE_USD] / df[COL_SIZE_USD].replace(0, np.nan).median()
        df[COL_ESTIMATED_LEVERAGE] = df[COL_ESTIMATED_LEVERAGE].fillna(1.0)
        # Create leverage buckets
        quantile_labels = ["Low", "Medium", "High", "Very High"]
        df[COL_LEVERAGE_BUCKET] = pd.qcut(
            df[COL_ESTIMATED_LEVERAGE].clip(lower=0),
            q=4,
            labels=quantile_labels,
            duplicates="drop",
        )

    # --- Trade size (alias) ---
    if COL_SIZE_USD in df.columns:
        df[COL_TRADE_SIZE] = df[COL_SIZE_USD].copy()

    # --- Average execution price (already per-row, just alias) ---
    if COL_EXEC_PRICE in df.columns:
        df[COL_AVG_EXEC_PRICE] = df[COL_EXEC_PRICE].copy()

    # --- Merge with Fear & Greed data ---
    # Ensure FG date is date-only for merging
    fg_merge = fg_df[[FG_DATE, FG_VALUE, FG_CLASSIFICATION]].copy()
    fg_merge[FG_DATE] = pd.to_datetime(fg_merge[FG_DATE]).dt.normalize()

    df[COL_TRADE_DATE] = pd.to_datetime(df[COL_TRADE_DATE]).dt.normalize()

    df = df.merge(
        fg_merge,
        left_on=COL_TRADE_DATE,
        right_on=FG_DATE,
        how="left",
    )
    # Drop duplicate date column from right side
    if FG_DATE in df.columns and COL_TRADE_DATE in df.columns:
        df.drop(columns=[FG_DATE], inplace=True, errors="ignore")

    # --- Sentiment category ---
    if FG_CLASSIFICATION in df.columns:
        df[COL_SENTIMENT_CATEGORY] = df[FG_CLASSIFICATION].fillna("Unknown")

    # --- Encoded sentiment ---
    if FG_CLASSIFICATION in df.columns:
        df[COL_ENCODED_SENTIMENT] = (
            df[FG_CLASSIFICATION]
            .map(SENTIMENT_ENCODING)
            .fillna(-1)
            .astype(int)
        )

    # --- Cumulative PnL ---
    if COL_PNL in df.columns:
        df = df.sort_values(by=[COL_TRADE_DATE, COL_TIMESTAMP_IST]
                            if COL_TIMESTAMP_IST in df.columns
                            else [COL_TRADE_DATE])
        df[COL_CUMULATIVE_PNL] = df[COL_PNL].cumsum()

    # --- Rolling PnL (7-trade window) ---
    if COL_PNL in df.columns:
        df[COL_ROLLING_PNL] = df[COL_PNL].rolling(window=7, min_periods=1).mean()

    # --- Daily PnL ---
    if COL_PNL in df.columns and COL_TRADE_DATE in df.columns:
        daily_pnl = df.groupby(COL_TRADE_DATE)[COL_PNL].sum().reset_index()
        daily_pnl.columns = [COL_TRADE_DATE, COL_DAILY_PNL]
        df = df.merge(daily_pnl, on=COL_TRADE_DATE, how="left")

    # --- Daily trade count ---
    if COL_TRADE_DATE in df.columns:
        daily_count = df.groupby(COL_TRADE_DATE).size().reset_index(name=COL_DAILY_TRADE_COUNT)
        df = df.merge(daily_count, on=COL_TRADE_DATE, how="left")

    # --- Average leverage per trader ---
    if COL_ESTIMATED_LEVERAGE in df.columns and COL_ACCOUNT in df.columns:
        avg_lev = df.groupby(COL_ACCOUNT)[COL_ESTIMATED_LEVERAGE].mean().reset_index()
        avg_lev.columns = [COL_ACCOUNT, COL_AVG_LEVERAGE]
        df = df.merge(avg_lev, on=COL_ACCOUNT, how="left")

    # --- Average PnL per trader ---
    if COL_PNL in df.columns and COL_ACCOUNT in df.columns:
        avg_pnl = df.groupby(COL_ACCOUNT)[COL_PNL].mean().reset_index()
        avg_pnl.columns = [COL_ACCOUNT, COL_AVG_PNL]
        df = df.merge(avg_pnl, on=COL_ACCOUNT, how="left")

    # Final sort
    df.reset_index(drop=True, inplace=True)
    return df


# ============================================================================
# EXPLORATORY DATA ANALYSIS — PLOTTING FUNCTIONS
# ============================================================================


def plot_sentiment_distribution(fg_df: pd.DataFrame) -> go.Figure:
    """1. Distribution of sentiment classes."""
    counts = fg_df[FG_CLASSIFICATION].value_counts().reset_index()
    counts.columns = ["Sentiment", "Count"]
    fig = px.bar(
        counts, x="Sentiment", y="Count", color="Sentiment",
        color_discrete_map=SENTIMENT_COLORS,
        title="Distribution of Fear & Greed Sentiment Classes",
        template=PLOTLY_TEMPLATE,
    )
    fig.update_layout(xaxis_title="Sentiment Class", yaxis_title="Number of Days")
    return fig


def plot_pnl_distribution(df: pd.DataFrame) -> go.Figure:
    """2. Distribution of closedPnL."""
    pnl_data = df[COL_PNL].dropna()
    # Clip extreme outliers for visualisation
    lower, upper = pnl_data.quantile(0.01), pnl_data.quantile(0.99)
    pnl_clipped = pnl_data.clip(lower, upper)
    fig = px.histogram(
        pnl_clipped, nbins=80,
        title="Distribution of Closed PnL (1st-99th percentile)",
        template=PLOTLY_TEMPLATE,
        labels={"value": "Closed PnL", "count": "Frequency"},
    )
    fig.update_layout(showlegend=False)
    return fig


def plot_leverage_histogram(df: pd.DataFrame) -> go.Figure:
    """3. Histogram of estimated leverage."""
    lev = df[COL_ESTIMATED_LEVERAGE].dropna()
    lower, upper = lev.quantile(0.01), lev.quantile(0.99)
    lev_clipped = lev.clip(lower, upper)
    fig = px.histogram(
        lev_clipped, nbins=60,
        title="Histogram of Estimated Leverage (1st-99th percentile)",
        template=PLOTLY_TEMPLATE,
        labels={"value": "Estimated Leverage", "count": "Frequency"},
    )
    fig.update_layout(showlegend=False)
    return fig


def plot_pnl_by_sentiment_box(df: pd.DataFrame) -> go.Figure:
    """4. Boxplot of PnL grouped by sentiment."""
    # Aggregate by sentiment to avoid millions of rows
    sentiment_order = ["Extreme Fear", "Fear", "Neutral", "Greed", "Extreme Greed"]
    present = [s for s in sentiment_order if s in df[COL_SENTIMENT_CATEGORY].unique()]
    fig = px.box(
        df[df[COL_SENTIMENT_CATEGORY].isin(present)],
        x=COL_SENTIMENT_CATEGORY, y=COL_PNL,
        category_orders={COL_SENTIMENT_CATEGORY: present},
        color=COL_SENTIMENT_CATEGORY,
        color_discrete_map=SENTIMENT_COLORS,
        title="Closed PnL by Sentiment Category (Boxplot)",
        template=PLOTLY_TEMPLATE,
    )
    fig.update_layout(yaxis_title="Closed PnL", xaxis_title="Sentiment")
    return fig


def plot_leverage_by_sentiment_violin(df: pd.DataFrame) -> go.Figure:
    """5. Violin plot of leverage grouped by sentiment."""
    sentiment_order = ["Extreme Fear", "Fear", "Neutral", "Greed", "Extreme Greed"]
    present = [s for s in sentiment_order if s in df[COL_SENTIMENT_CATEGORY].unique()]
    fig = px.violin(
        df[df[COL_SENTIMENT_CATEGORY].isin(present)],
        x=COL_SENTIMENT_CATEGORY, y=COL_ESTIMATED_LEVERAGE,
        category_orders={COL_SENTIMENT_CATEGORY: present},
        color=COL_SENTIMENT_CATEGORY,
        color_discrete_map=SENTIMENT_COLORS,
        box=True,
        title="Estimated Leverage by Sentiment Category (Violin Plot)",
        template=PLOTLY_TEMPLATE,
    )
    fig.update_layout(yaxis_title="Estimated Leverage", xaxis_title="Sentiment")
    return fig


def plot_correlation_heatmap(df: pd.DataFrame) -> go.Figure:
    """6. Correlation heatmap of numeric features."""
    numeric_cols = [COL_PNL, COL_ESTIMATED_LEVERAGE, COL_TRADE_SIZE,
                    COL_AVG_EXEC_PRICE, COL_ENCODED_SENTIMENT, COL_FEE]
    existing = [c for c in numeric_cols if c in df.columns]
    corr = df[existing].corr()
    fig = px.imshow(
        corr, text_auto=".2f", aspect="auto",
        title="Correlation Heatmap of Key Features",
        color_continuous_scale="RdBu_r",
        template=PLOTLY_TEMPLATE,
    )
    fig.update_layout(height=550)
    return fig


def plot_leverage_vs_pnl_scatter(df: pd.DataFrame) -> go.Figure:
    """7. Scatter plot: leverage vs PnL."""
    sample = df[[COL_ESTIMATED_LEVERAGE, COL_PNL, COL_SENTIMENT_CATEGORY]].dropna()
    # Sample for performance
    if len(sample) > 20000:
        sample = sample.sample(20000, random_state=42)
    fig = px.scatter(
        sample, x=COL_ESTIMATED_LEVERAGE, y=COL_PNL,
        color=COL_SENTIMENT_CATEGORY,
        color_discrete_map=SENTIMENT_COLORS,
        title="Estimated Leverage vs Closed PnL",
        template=PLOTLY_TEMPLATE,
        opacity=0.5,
    )
    fig.update_layout(xaxis_title="Estimated Leverage", yaxis_title="Closed PnL")
    return fig


def plot_size_vs_pnl_scatter(df: pd.DataFrame) -> go.Figure:
    """8. Scatter plot: trade size vs PnL."""
    sample = df[[COL_TRADE_SIZE, COL_PNL, COL_SENTIMENT_CATEGORY]].dropna()
    if len(sample) > 20000:
        sample = sample.sample(20000, random_state=42)
    fig = px.scatter(
        sample, x=COL_TRADE_SIZE, y=COL_PNL,
        color=COL_SENTIMENT_CATEGORY,
        color_discrete_map=SENTIMENT_COLORS,
        title="Trade Size (USD) vs Closed PnL",
        template=PLOTLY_TEMPLATE,
        opacity=0.5,
    )
    fig.update_layout(xaxis_title="Trade Size (USD)", yaxis_title="Closed PnL")
    return fig


def plot_daily_trade_counts(df: pd.DataFrame) -> go.Figure:
    """9. Daily trade counts over time."""
    daily = df.groupby(COL_TRADE_DATE).size().reset_index(name="Trade Count")
    fig = px.bar(
        daily, x=COL_TRADE_DATE, y="Trade Count",
        title="Daily Trade Counts",
        template=PLOTLY_TEMPLATE,
    )
    fig.update_layout(xaxis_title="Date", yaxis_title="Number of Trades")
    return fig


def plot_daily_pnl_trend(df: pd.DataFrame) -> go.Figure:
    """10. Daily PnL trend."""
    if COL_DAILY_PNL not in df.columns:
        daily = df.groupby(COL_TRADE_DATE)[COL_PNL].sum().reset_index()
        daily.columns = [COL_TRADE_DATE, "Daily PnL"]
    else:
        daily = df.groupby(COL_TRADE_DATE)[COL_DAILY_PNL].first().reset_index()
        daily.columns = [COL_TRADE_DATE, "Daily PnL"]
    fig = px.bar(
        daily, x=COL_TRADE_DATE, y="Daily PnL",
        title="Daily PnL Trend",
        template=PLOTLY_TEMPLATE,
        color="Daily PnL",
        color_continuous_scale="RdYlGn",
    )
    fig.update_layout(xaxis_title="Date", yaxis_title="Daily PnL (USD)")
    return fig


def plot_cumulative_pnl(df: pd.DataFrame) -> go.Figure:
    """11. Cumulative PnL over time."""
    daily = df.groupby(COL_TRADE_DATE)[COL_PNL].sum().reset_index()
    daily.columns = [COL_TRADE_DATE, "Daily PnL"]
    daily = daily.sort_values(COL_TRADE_DATE)
    daily["Cumulative PnL"] = daily["Daily PnL"].cumsum()
    fig = px.line(
        daily, x=COL_TRADE_DATE, y="Cumulative PnL",
        title="Cumulative PnL Over Time",
        template=PLOTLY_TEMPLATE,
    )
    fig.update_layout(xaxis_title="Date", yaxis_title="Cumulative PnL (USD)")
    return fig


def plot_top_profitable_traders(df: pd.DataFrame, n: int = 20) -> go.Figure:
    """12. Top N profitable traders."""
    trader_pnl = df.groupby(COL_ACCOUNT)[COL_PNL].sum().nlargest(n).reset_index()
    trader_pnl["Account Short"] = trader_pnl[COL_ACCOUNT].str[:10] + "..."
    fig = px.bar(
        trader_pnl, x="Account Short", y=COL_PNL,
        title=f"Top {n} Most Profitable Traders",
        template=PLOTLY_TEMPLATE,
        color=COL_PNL, color_continuous_scale="Greens",
    )
    fig.update_layout(xaxis_title="Account", yaxis_title="Total PnL (USD)")
    return fig


def plot_top_losing_traders(df: pd.DataFrame, n: int = 20) -> go.Figure:
    """13. Top N losing traders."""
    trader_pnl = df.groupby(COL_ACCOUNT)[COL_PNL].sum().nsmallest(n).reset_index()
    trader_pnl["Account Short"] = trader_pnl[COL_ACCOUNT].str[:10] + "..."
    fig = px.bar(
        trader_pnl, x="Account Short", y=COL_PNL,
        title=f"Top {n} Biggest Losing Traders",
        template=PLOTLY_TEMPLATE,
        color=COL_PNL, color_continuous_scale="Reds",
    )
    fig.update_layout(xaxis_title="Account", yaxis_title="Total PnL (USD)")
    return fig


def plot_avg_pnl_by_sentiment(df: pd.DataFrame) -> go.Figure:
    """14. Average PnL by sentiment."""
    avg_pnl = df.groupby(COL_SENTIMENT_CATEGORY)[COL_PNL].mean().reset_index()
    sentiment_order = ["Extreme Fear", "Fear", "Neutral", "Greed", "Extreme Greed"]
    present = [s for s in sentiment_order if s in avg_pnl[COL_SENTIMENT_CATEGORY].values]
    avg_pnl = avg_pnl.set_index(COL_SENTIMENT_CATEGORY).loc[present].reset_index()
    fig = px.bar(
        avg_pnl, x=COL_SENTIMENT_CATEGORY, y=COL_PNL,
        color=COL_SENTIMENT_CATEGORY,
        color_discrete_map=SENTIMENT_COLORS,
        title="Average PnL by Sentiment Category",
        template=PLOTLY_TEMPLATE,
    )
    fig.update_layout(xaxis_title="Sentiment", yaxis_title="Average PnL (USD)")
    return fig


def plot_avg_leverage_by_sentiment(df: pd.DataFrame) -> go.Figure:
    """15. Average leverage by sentiment."""
    avg_lev = df.groupby(COL_SENTIMENT_CATEGORY)[COL_ESTIMATED_LEVERAGE].mean().reset_index()
    sentiment_order = ["Extreme Fear", "Fear", "Neutral", "Greed", "Extreme Greed"]
    present = [s for s in sentiment_order if s in avg_lev[COL_SENTIMENT_CATEGORY].values]
    avg_lev = avg_lev.set_index(COL_SENTIMENT_CATEGORY).loc[present].reset_index()
    fig = px.bar(
        avg_lev, x=COL_SENTIMENT_CATEGORY, y=COL_ESTIMATED_LEVERAGE,
        color=COL_SENTIMENT_CATEGORY,
        color_discrete_map=SENTIMENT_COLORS,
        title="Average Estimated Leverage by Sentiment Category",
        template=PLOTLY_TEMPLATE,
    )
    fig.update_layout(xaxis_title="Sentiment", yaxis_title="Average Estimated Leverage")
    return fig


def plot_symbol_pnl(df: pd.DataFrame, top_n: int = 20) -> go.Figure:
    """16. Symbol-wise PnL."""
    sym_pnl = df.groupby(COL_SYMBOL)[COL_PNL].sum().reset_index()
    sym_pnl = sym_pnl.reindex(
        sym_pnl[COL_PNL].abs().sort_values(ascending=False).index
    ).head(top_n)
    fig = px.bar(
        sym_pnl, x=COL_SYMBOL, y=COL_PNL,
        title=f"Top {top_n} Symbols by Total PnL",
        template=PLOTLY_TEMPLATE,
        color=COL_PNL, color_continuous_scale="RdYlGn",
    )
    fig.update_layout(xaxis_title="Symbol", yaxis_title="Total PnL (USD)")
    return fig


def plot_side_analysis(df: pd.DataFrame) -> go.Figure:
    """17. Side (Buy/Sell) analysis."""
    side_data = df.groupby(COL_SIDE).agg(
        Total_PnL=(COL_PNL, "sum"),
        Avg_PnL=(COL_PNL, "mean"),
        Trade_Count=(COL_PNL, "count"),
        Win_Rate=(COL_WIN_LOSS, lambda x: (x == "Win").mean()),
    ).reset_index()
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=("Total PnL", "Average PnL", "Trade Count", "Win Rate"),
    )
    colors = {"BUY": "#2196F3", "SELL": "#FF5722"}
    for side in side_data[COL_SIDE].unique():
        row_data = side_data[side_data[COL_SIDE] == side]
        c = colors.get(side, "#888888")
        fig.add_trace(go.Bar(x=[side], y=row_data["Total_PnL"], name=f"{side} Total PnL",
                             marker_color=c), row=1, col=1)
        fig.add_trace(go.Bar(x=[side], y=row_data["Avg_PnL"], name=f"{side} Avg PnL",
                             marker_color=c), row=1, col=2)
        fig.add_trace(go.Bar(x=[side], y=row_data["Trade_Count"], name=f"{side} Count",
                             marker_color=c), row=2, col=1)
        fig.add_trace(go.Bar(x=[side], y=row_data["Win_Rate"], name=f"{side} Win Rate",
                             marker_color=c), row=2, col=2)

    fig.update_layout(title="Side (Buy/Sell) Analysis", template=PLOTLY_TEMPLATE,
                      showlegend=False, height=600)
    return fig


def plot_event_distribution(df: pd.DataFrame) -> go.Figure:
    """18. Event (Direction) distribution."""
    if COL_DIRECTION not in df.columns:
        fig = go.Figure()
        fig.update_layout(title="Direction data not available", template=PLOTLY_TEMPLATE)
        return fig
    direction_counts = df[COL_DIRECTION].value_counts().reset_index()
    direction_counts.columns = ["Direction", "Count"]
    fig = px.bar(
        direction_counts, x="Direction", y="Count",
        title="Trade Direction (Event) Distribution",
        template=PLOTLY_TEMPLATE,
        color="Direction",
    )
    fig.update_layout(xaxis_title="Direction", yaxis_title="Number of Trades")
    return fig


def plot_rolling_7day_pnl(df: pd.DataFrame) -> go.Figure:
    """19. Rolling 7-day PnL."""
    daily = df.groupby(COL_TRADE_DATE)[COL_PNL].sum().reset_index()
    daily.columns = [COL_TRADE_DATE, "Daily PnL"]
    daily = daily.sort_values(COL_TRADE_DATE)
    daily["Rolling 7-Day PnL"] = daily["Daily PnL"].rolling(window=7, min_periods=1).mean()
    fig = px.line(
        daily, x=COL_TRADE_DATE, y="Rolling 7-Day PnL",
        title="Rolling 7-Day Average PnL",
        template=PLOTLY_TEMPLATE,
    )
    fig.update_layout(xaxis_title="Date", yaxis_title="7-Day Rolling Avg PnL (USD)")
    return fig


def plot_rolling_trade_volume(df: pd.DataFrame) -> go.Figure:
    """20. Rolling trade volume."""
    daily = df.groupby(COL_TRADE_DATE)[COL_TRADE_SIZE].sum().reset_index()
    daily.columns = [COL_TRADE_DATE, "Daily Volume"]
    daily = daily.sort_values(COL_TRADE_DATE)
    daily["Rolling 7-Day Volume"] = daily["Daily Volume"].rolling(window=7, min_periods=1).mean()
    fig = px.line(
        daily, x=COL_TRADE_DATE, y="Rolling 7-Day Volume",
        title="Rolling 7-Day Trade Volume (USD)",
        template=PLOTLY_TEMPLATE,
    )
    fig.update_layout(xaxis_title="Date", yaxis_title="7-Day Rolling Volume (USD)")
    return fig


# ============================================================================
# STATISTICAL ANALYSIS
# ============================================================================


def compute_statistical_analysis(df: pd.DataFrame) -> Dict:
    """Perform comprehensive statistical analysis on PnL by sentiment.

    Returns:
        Dictionary containing all statistical results.
    """
    results = {}

    # --- Descriptive statistics by sentiment ---
    sentiment_groups = df.groupby(COL_SENTIMENT_CATEGORY)[COL_PNL]
    desc = sentiment_groups.describe()
    results["descriptive"] = desc

    # --- Mean, median, std, variance by sentiment ---
    mean_pnl = sentiment_groups.mean()
    median_pnl = sentiment_groups.median()
    std_pnl = sentiment_groups.std()
    var_pnl = sentiment_groups.var()
    results["mean_pnl_by_sentiment"] = mean_pnl
    results["median_pnl_by_sentiment"] = median_pnl
    results["std_pnl_by_sentiment"] = std_pnl
    results["var_pnl_by_sentiment"] = var_pnl

    # --- Confidence intervals (95%) ---
    ci_results = {}
    for name, group in sentiment_groups:
        n = len(group)
        if n < 2:
            ci_results[name] = (np.nan, np.nan)
            continue
        mean_val = group.mean()
        se = group.std() / np.sqrt(n)
        t_crit = stats.t.ppf(0.975, df=n - 1)
        ci_results[name] = (mean_val - t_crit * se, mean_val + t_crit * se)
    results["confidence_intervals"] = ci_results

    # --- Pearson correlation ---
    valid = df[[COL_ENCODED_SENTIMENT, COL_PNL]].dropna()
    valid = valid[valid[COL_ENCODED_SENTIMENT] >= 0]  # exclude unknown
    if len(valid) > 2:
        pearson_r, pearson_p = stats.pearsonr(valid[COL_ENCODED_SENTIMENT], valid[COL_PNL])
        results["pearson"] = {"r": pearson_r, "p_value": pearson_p}
    else:
        results["pearson"] = {"r": np.nan, "p_value": np.nan}

    # --- Spearman correlation ---
    if len(valid) > 2:
        spearman_r, spearman_p = stats.spearmanr(valid[COL_ENCODED_SENTIMENT], valid[COL_PNL])
        results["spearman"] = {"rho": spearman_r, "p_value": spearman_p}
    else:
        results["spearman"] = {"rho": np.nan, "p_value": np.nan}

    # --- T-test: Fear vs Greed ---
    fear_pnl = df[df[COL_SENTIMENT_CATEGORY].isin(["Extreme Fear", "Fear"])][COL_PNL].dropna()
    greed_pnl = df[df[COL_SENTIMENT_CATEGORY].isin(["Greed", "Extreme Greed"])][COL_PNL].dropna()
    if len(fear_pnl) > 1 and len(greed_pnl) > 1:
        t_stat, t_p = stats.ttest_ind(fear_pnl, greed_pnl, equal_var=False)
        results["ttest_fear_greed"] = {
            "t_statistic": t_stat,
            "p_value": t_p,
            "fear_mean": fear_pnl.mean(),
            "greed_mean": greed_pnl.mean(),
            "fear_n": len(fear_pnl),
            "greed_n": len(greed_pnl),
            "significant": t_p < 0.05,
        }
    else:
        results["ttest_fear_greed"] = None

    # --- ANOVA across all sentiment classes ---
    groups_for_anova = []
    group_names = []
    for name, group in sentiment_groups:
        g = group.dropna()
        if len(g) > 1:
            groups_for_anova.append(g)
            group_names.append(name)
    if len(groups_for_anova) >= 2:
        f_stat, anova_p = stats.f_oneway(*groups_for_anova)
        results["anova"] = {
            "f_statistic": f_stat,
            "p_value": anova_p,
            "groups": group_names,
            "significant": anova_p < 0.05,
        }
    else:
        results["anova"] = None

    return results


def format_statistical_results(results: Dict) -> str:
    """Format statistical analysis results into a readable string."""
    lines = []
    lines.append("=" * 60)
    lines.append("STATISTICAL ANALYSIS RESULTS")
    lines.append("=" * 60)

    # Descriptive statistics
    lines.append("\n--- Descriptive Statistics by Sentiment ---")
    for sentiment in results["mean_pnl_by_sentiment"].index:
        lines.append(f"\n  {sentiment}:")
        lines.append(f"    Mean PnL:    {results['mean_pnl_by_sentiment'][sentiment]:,.2f}")
        lines.append(f"    Median PnL:  {results['median_pnl_by_sentiment'][sentiment]:,.2f}")
        lines.append(f"    Std Dev:     {results['std_pnl_by_sentiment'][sentiment]:,.2f}")
        lines.append(f"    Variance:    {results['var_pnl_by_sentiment'][sentiment]:,.2f}")
        ci = results["confidence_intervals"].get(sentiment, (np.nan, np.nan))
        lines.append(f"    95% CI:      ({ci[0]:,.2f}, {ci[1]:,.2f})")

    # Correlations
    lines.append("\n--- Correlation Analysis ---")
    p = results["pearson"]
    lines.append(f"  Pearson r:  {p['r']:.4f}  (p-value: {p['p_value']:.4e})")
    s = results["spearman"]
    lines.append(f"  Spearman rho: {s['rho']:.4f}  (p-value: {s['p_value']:.4e})")

    # T-test
    ttest = results.get("ttest_fear_greed")
    if ttest:
        lines.append("\n--- T-Test: Fear vs Greed ---")
        lines.append(f"  t-statistic: {ttest['t_statistic']:.4f}")
        lines.append(f"  p-value:     {ttest['p_value']:.4e}")
        lines.append(f"  Fear mean:   {ttest['fear_mean']:,.2f} (n={ttest['fear_n']})")
        lines.append(f"  Greed mean:  {ttest['greed_mean']:,.2f} (n={ttest['greed_n']})")
        sig = "YES" if ttest["significant"] else "NO"
        lines.append(f"  Significant (p<0.05): {sig}")

    # ANOVA
    anova = results.get("anova")
    if anova:
        lines.append("\n--- ANOVA Across All Sentiment Classes ---")
        lines.append(f"  F-statistic: {anova['f_statistic']:.4f}")
        lines.append(f"  p-value:     {anova['p_value']:.4e}")
        lines.append(f"  Groups:      {', '.join(anova['groups'])}")
        sig = "YES" if anova["significant"] else "NO"
        lines.append(f"  Significant (p<0.05): {sig}")

    lines.append("\n" + "=" * 60)
    return "\n".join(lines)


# ============================================================================
# TRADER ANALYSIS
# ============================================================================


def generate_trader_tables(df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    """Generate summary tables for various trader metrics.

    Returns:
        Dictionary of table name -> DataFrame.
    """
    tables = {}
    trader_agg = df.groupby(COL_ACCOUNT).agg(
        total_pnl=(COL_PNL, "sum"),
        avg_pnl=(COL_PNL, "mean"),
        trade_count=(COL_PNL, "count"),
        avg_leverage=(COL_ESTIMATED_LEVERAGE, "mean"),
        win_count=(COL_WIN_LOSS, lambda x: (x == "Win").sum()),
        loss_count=(COL_WIN_LOSS, lambda x: (x == "Loss").sum()),
    ).reset_index()
    trader_agg["win_rate"] = trader_agg["win_count"] / trader_agg["trade_count"]
    trader_agg["account_short"] = trader_agg[COL_ACCOUNT].str[:12] + "..."

    # Best traders (highest total PnL)
    tables["best_traders"] = (
        trader_agg.nlargest(20, "total_pnl")[
            ["account_short", "total_pnl", "avg_pnl", "trade_count", "win_rate"]
        ]
        .reset_index(drop=True)
    )
    tables["best_traders"].columns = ["Account", "Total PnL", "Avg PnL", "Trades", "Win Rate"]

    # Worst traders (lowest total PnL)
    tables["worst_traders"] = (
        trader_agg.nsmallest(20, "total_pnl")[
            ["account_short", "total_pnl", "avg_pnl", "trade_count", "win_rate"]
        ]
        .reset_index(drop=True)
    )
    tables["worst_traders"].columns = ["Account", "Total PnL", "Avg PnL", "Trades", "Win Rate"]

    # Highest leverage traders
    tables["highest_leverage"] = (
        trader_agg.nlargest(20, "avg_leverage")[
            ["account_short", "avg_leverage", "total_pnl", "trade_count", "win_rate"]
        ]
        .reset_index(drop=True)
    )
    tables["highest_leverage"].columns = ["Account", "Avg Leverage", "Total PnL", "Trades", "Win Rate"]

    # Lowest leverage traders
    tables["lowest_leverage"] = (
        trader_agg.nsmallest(20, "avg_leverage")[
            ["account_short", "avg_leverage", "total_pnl", "trade_count", "win_rate"]
        ]
        .reset_index(drop=True)
    )
    tables["lowest_leverage"].columns = ["Account", "Avg Leverage", "Total PnL", "Trades", "Win Rate"]

    # Highest average PnL
    tables["highest_avg_pnl"] = (
        trader_agg.nlargest(20, "avg_pnl")[
            ["account_short", "avg_pnl", "total_pnl", "trade_count", "win_rate"]
        ]
        .reset_index(drop=True)
    )
    tables["highest_avg_pnl"].columns = ["Account", "Avg PnL", "Total PnL", "Trades", "Win Rate"]

    # Most trades
    tables["most_trades"] = (
        trader_agg.nlargest(20, "trade_count")[
            ["account_short", "trade_count", "total_pnl", "avg_pnl", "win_rate"]
        ]
        .reset_index(drop=True)
    )
    tables["most_trades"].columns = ["Account", "Trades", "Total PnL", "Avg PnL", "Win Rate"]

    return tables


# ============================================================================
# MARKET INSIGHTS
# ============================================================================


def generate_market_insights(df: pd.DataFrame, stat_results: Dict) -> List[str]:
    """Generate human-readable market insights from the data.

    Args:
        df: Enriched DataFrame.
        stat_results: Output of compute_statistical_analysis().

    Returns:
        List of insight strings.
    """
    insights = []

    # Filter out "Unknown" sentiment for cleaner insights
    df_known = df[df[COL_SENTIMENT_CATEGORY] != "Unknown"]

    # Which sentiment is most profitable?
    avg_pnl = df_known.groupby(COL_SENTIMENT_CATEGORY)[COL_PNL].mean()
    if len(avg_pnl) > 0:
        best_sentiment = avg_pnl.idxmax()
        best_val = avg_pnl.max()
        insights.append(
            f"Most profitable sentiment: **{best_sentiment}** with an average PnL of ${best_val:,.2f} per trade."
        )

    # Which sentiment has highest losses?
    worst_sentiment = avg_pnl.idxmin()
    worst_val = avg_pnl.min()
    insights.append(
        f"Least profitable sentiment: **{worst_sentiment}** with an average PnL of ${worst_val:,.2f} per trade."
    )

    # Does leverage increase during Fear?
    avg_lev = df_known.groupby(COL_SENTIMENT_CATEGORY)[COL_ESTIMATED_LEVERAGE].mean()
    fear_lev = avg_lev.get("Fear", avg_lev.get("Extreme Fear", np.nan))
    greed_lev = avg_lev.get("Greed", avg_lev.get("Extreme Greed", np.nan))
    if not np.isnan(fear_lev) and not np.isnan(greed_lev):
        if fear_lev > greed_lev:
            insights.append(
                f"Leverage tends to **increase during Fear** (avg leverage {fear_lev:.2f}) "
                f"compared to Greed ({greed_lev:.2f}). Traders take larger positions when fearful."
            )
        else:
            insights.append(
                f"Leverage tends to **increase during Greed** (avg leverage {greed_lev:.2f}) "
                f"compared to Fear ({fear_lev:.2f}). Traders take larger positions when greedy."
            )

    # Does leverage increase during Greed?
    extreme_fear_lev = avg_lev.get("Extreme Fear", np.nan)
    extreme_greed_lev = avg_lev.get("Extreme Greed", np.nan)
    if not np.isnan(extreme_greed_lev) and not np.isnan(extreme_fear_lev):
        if extreme_greed_lev > extreme_fear_lev:
            insights.append(
                f"Extreme Greed shows higher leverage ({extreme_greed_lev:.2f}) than "
                f"Extreme Fear ({extreme_fear_lev:.2f}), suggesting euphoria drives risk-taking."
            )
        else:
            insights.append(
                f"Extreme Fear shows higher leverage ({extreme_fear_lev:.2f}) than "
                f"Extreme Greed ({extreme_greed_lev:.2f}), suggesting panic drives overtrading."
            )

    # Does higher leverage improve returns?
    lev_pnl = df.groupby(COL_LEVERAGE_BUCKET)[COL_PNL].mean()
    if len(lev_pnl) > 1:
        best_lev_bucket = lev_pnl.idxmax()
        worst_lev_bucket = lev_pnl.idxmin()
        insights.append(
            f"Traders with **{best_lev_bucket} leverage** have the highest average PnL "
            f"(${lev_pnl[best_lev_bucket]:,.2f}), while **{worst_lev_bucket} leverage** "
            f"traders average ${lev_pnl[worst_lev_bucket]:,.2f}."
        )

    # Which symbols perform best?
    sym_pnl = df.groupby(COL_SYMBOL)[COL_PNL].sum().nlargest(5)
    top_syms = ", ".join([f"{s} (${v:,.0f})" for s, v in sym_pnl.items()])
    insights.append(f"Top-performing symbols by total PnL: {top_syms}.")

    # Which side performs better?
    side_pnl = df.groupby(COL_SIDE)[COL_PNL].mean()
    if len(side_pnl) >= 2:
        better_side = side_pnl.idxmax()
        insights.append(
            f"**{better_side}** side performs better with average PnL of "
            f"${side_pnl[better_side]:,.2f} vs ${side_pnl.min():,.2f} for the other side."
        )

    # Are profitable traders using lower leverage?
    profitable = df[df[COL_PNL] > 0][COL_ESTIMATED_LEVERAGE].mean()
    unprofitable = df[df[COL_PNL] <= 0][COL_ESTIMATED_LEVERAGE].mean()
    if not np.isnan(profitable) and not np.isnan(unprofitable):
        if profitable < unprofitable:
            insights.append(
                f"Profitable traders use **lower average leverage** ({profitable:.2f}) "
                f"than unprofitable traders ({unprofitable:.2f}). Discipline pays off."
            )
        else:
            insights.append(
                f"Profitable traders use **higher average leverage** ({profitable:.2f}) "
                f"than unprofitable traders ({unprofitable:.2f}). Aggression rewards the skilled."
            )

    # Statistical significance
    ttest = stat_results.get("ttest_fear_greed")
    if ttest:
        if ttest["significant"]:
            insights.append(
                f"The difference in PnL between Fear and Greed periods is **statistically "
                f"significant** (p={ttest['p_value']:.4e})."
            )
        else:
            insights.append(
                f"The difference in PnL between Fear and Greed periods is **not statistically "
                f"significant** (p={ttest['p_value']:.4e})."
            )

    anova = stat_results.get("anova")
    if anova and anova["significant"]:
        insights.append(
            f"ANOVA confirms **significant differences** in PnL across sentiment "
            f"categories (F={anova['f_statistic']:.2f}, p={anova['p_value']:.4e})."
        )

    return insights


# ============================================================================
# MACHINE LEARNING
# ============================================================================


def build_ml_model(df: pd.DataFrame) -> Dict:
    """Build a RandomForestClassifier to predict profitable trades.

    Args:
        df: Enriched DataFrame with engineered features.

    Returns:
        Dictionary containing model, metrics, and feature importance.
    """
    result = {}

    # --- Prepare features ---
    feature_cols = []
    if COL_ESTIMATED_LEVERAGE in df.columns:
        feature_cols.append(COL_ESTIMATED_LEVERAGE)
    if COL_TRADE_SIZE in df.columns:
        feature_cols.append(COL_TRADE_SIZE)
    if COL_AVG_EXEC_PRICE in df.columns:
        feature_cols.append(COL_AVG_EXEC_PRICE)
    if COL_ENCODED_SENTIMENT in df.columns:
        feature_cols.append(COL_ENCODED_SENTIMENT)

    # Encode Side
    if COL_SIDE in df.columns:
        df["side_encoded"] = df[COL_SIDE].map({"BUY": 1, "SELL": 0}).fillna(-1).astype(int)
        feature_cols.append("side_encoded")

    # Encode Symbol (top N, rest as "Other")
    if COL_SYMBOL in df.columns:
        top_symbols = df[COL_SYMBOL].value_counts().nlargest(20).index.tolist()
        df["symbol_simplified"] = df[COL_SYMBOL].apply(lambda x: x if x in top_symbols else "Other")
        symbol_dummies = pd.get_dummies(df["symbol_simplified"], prefix="sym", dtype=int)
        feature_cols.extend(symbol_dummies.columns.tolist())
        df = pd.concat([df, symbol_dummies], axis=1)

    # Target variable
    df["target_profitable"] = (df[COL_PNL] > 0).astype(int)

    # Drop NaN in features or target
    model_df = df[feature_cols + ["target_profitable"]].dropna()
    model_df = model_df[model_df[COL_ENCODED_SENTIMENT] >= 0]  # exclude unknown sentiment

    if len(model_df) < 100:
        result["error"] = "Insufficient data for ML model (need at least 100 valid rows)."
        return result

    X = model_df[feature_cols]
    y = model_df["target_profitable"]

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Train model
    clf = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        random_state=42,
        n_jobs=-1,
        class_weight="balanced",
    )
    clf.fit(X_train, y_train)

    # Predictions
    y_pred = clf.predict(X_test)
    y_pred_proba = clf.predict_proba(X_test)[:, 1]

    # Metrics
    result["accuracy"] = accuracy_score(y_test, y_pred)
    result["precision"] = precision_score(y_test, y_pred, zero_division=0)
    result["recall"] = recall_score(y_test, y_pred, zero_division=0)
    result["f1"] = f1_score(y_test, y_pred, zero_division=0)
    result["confusion_matrix"] = confusion_matrix(y_test, y_pred)

    # ROC AUC
    try:
        result["roc_auc"] = roc_auc_score(y_test, y_pred_proba)
    except ValueError:
        result["roc_auc"] = np.nan

    # Classification report
    result["classification_report"] = classification_report(
        y_test, y_pred, zero_division=0
    )

    # Feature importance
    importance = pd.DataFrame({
        "feature": feature_cols,
        "importance": clf.feature_importances_,
    }).sort_values("importance", ascending=False)
    result["feature_importance"] = importance
    result["feature_cols"] = feature_cols
    result["model"] = clf

    # Confusion matrix figure
    cm = result["confusion_matrix"]
    fig_cm = px.imshow(
        cm, text_auto=True,
        x=["Not Profitable", "Profitable"],
        y=["Not Profitable", "Profitable"],
        title="Confusion Matrix",
        template=PLOTLY_TEMPLATE,
        color_continuous_scale="Blues",
    )
    fig_cm.update_layout(xaxis_title="Predicted", yaxis_title="Actual")
    result["confusion_matrix_fig"] = fig_cm

    # Feature importance figure
    top_n = min(15, len(importance))
    fig_imp = px.bar(
        importance.head(top_n), x="importance", y="feature",
        orientation="h",
        title=f"Top {top_n} Feature Importance (Random Forest)",
        template=PLOTLY_TEMPLATE,
    )
    fig_imp.update_layout(yaxis={"categoryorder": "total ascending"},
                          xaxis_title="Importance", yaxis_title="Feature")
    result["feature_importance_fig"] = fig_imp

    return result


# ============================================================================
# STREAMLIT DASHBOARD
# ============================================================================


def init_streamlit_config():
    """Configure Streamlit page settings."""
    st.set_page_config(
        page_title="Bitcoin Sentiment vs Hyperliquid Performance",
        page_icon="📊",
        layout="wide",
        initial_sidebar_state="expanded",
    )


def render_sidebar(fg_df: pd.DataFrame, df: pd.DataFrame) -> Dict:
    """Render the Streamlit sidebar with filters and return filter state.

    Args:
        fg_df: Fear & Greed DataFrame.
        df: Enriched trade DataFrame.

    Returns:
        Dictionary of filter values.
    """
    st.sidebar.title("⚙️ Controls & Filters")

    # --- File uploads ---
    st.sidebar.markdown("### 📁 Data Upload")
    fg_upload = st.sidebar.file_uploader(
        "Upload Fear & Greed CSV", type=["csv"], key="fg_upload"
    )
    hl_upload = st.sidebar.file_uploader(
        "Upload Hyperliquid CSV", type=["csv"], key="hl_upload"
    )

    # --- Sentiment filter ---
    st.sidebar.markdown("### 🔍 Filters")
    all_sentiments = sorted(df[COL_SENTIMENT_CATEGORY].unique().tolist())
    selected_sentiments = st.sidebar.multiselect(
        "Sentiment Filter", all_sentiments, default=all_sentiments
    )

    # --- Symbol filter ---
    all_symbols = sorted(df[COL_SYMBOL].unique().tolist())
    top_symbols_default = all_symbols[:10] if len(all_symbols) > 10 else all_symbols
    selected_symbols = st.sidebar.multiselect(
        "Symbol Filter", all_symbols, default=top_symbols_default
    )

    # --- Trader filter ---
    all_accounts = df[COL_ACCOUNT].unique().tolist()
    selected_trader = st.sidebar.text_input(
        "Trader Filter (account prefix)", value=""
    )

    # --- Date range filter ---
    if COL_TRADE_DATE in df.columns:
        min_date = df[COL_TRADE_DATE].min()
        max_date = df[COL_TRADE_DATE].max()
        if pd.notna(min_date) and pd.notna(max_date):
            date_range = st.sidebar.date_input(
                "Date Range",
                value=(min_date, max_date),
                min_value=min_date,
                max_value=max_date,
            )
        else:
            date_range = None
    else:
        date_range = None

    return {
        "fg_upload": fg_upload,
        "hl_upload": hl_upload,
        "selected_sentiments": selected_sentiments,
        "selected_symbols": selected_symbols,
        "selected_trader": selected_trader,
        "date_range": date_range,
    }


def apply_filters(df: pd.DataFrame, filters: Dict) -> pd.DataFrame:
    """Apply sidebar filters to the DataFrame.

    Args:
        df: Enriched trade DataFrame.
        filters: Filter state from render_sidebar().

    Returns:
        Filtered DataFrame.
    """
    filtered = df.copy()

    # Sentiment filter
    if filters["selected_sentiments"]:
        filtered = filtered[filtered[COL_SENTIMENT_CATEGORY].isin(filters["selected_sentiments"])]

    # Symbol filter
    if filters["selected_symbols"]:
        filtered = filtered[filtered[COL_SYMBOL].isin(filters["selected_symbols"])]

    # Trader filter
    if filters["selected_trader"]:
        filtered = filtered[
            filtered[COL_ACCOUNT].str.startswith(filters["selected_trader"], na=False)
        ]

    # Date range filter
    date_range = filters.get("date_range")
    if date_range and len(date_range) == 2:
        start, end = pd.to_datetime(date_range[0]), pd.to_datetime(date_range[1])
        filtered = filtered[
            (filtered[COL_TRADE_DATE] >= start) & (filtered[COL_TRADE_DATE] <= end)
        ]

    return filtered


def render_kpis(df: pd.DataFrame):
    """Render KPI cards at the top of the dashboard."""
    col1, col2, col3, col4, col5, col6 = st.columns(6)

    total_pnl = df[COL_PNL].sum() if COL_PNL in df.columns else 0
    avg_pnl = df[COL_PNL].mean() if COL_PNL in df.columns else 0
    total_trades = len(df)
    win_rate = (df[COL_WIN_LOSS] == "Win").mean() * 100 if COL_WIN_LOSS in df.columns else 0
    avg_leverage = df[COL_ESTIMATED_LEVERAGE].mean() if COL_ESTIMATED_LEVERAGE in df.columns else 0
    unique_traders = df[COL_ACCOUNT].nunique() if COL_ACCOUNT in df.columns else 0

    col1.metric("Total PnL", f"${total_pnl:,.0f}")
    col2.metric("Avg PnL/Trade", f"${avg_pnl:,.2f}")
    col3.metric("Total Trades", f"{total_trades:,}")
    col4.metric("Win Rate", f"{win_rate:.1f}%")
    col5.metric("Avg Leverage", f"{avg_leverage:.2f}x")
    col6.metric("Unique Traders", f"{unique_traders:,}")


def render_eda_section(df: pd.DataFrame, fg_df: pd.DataFrame):
    """Render all EDA charts in the dashboard."""
    st.header("📈 Exploratory Data Analysis")

    # Row 1: Distribution plots
    st.subheader("Distributions")
    c1, c2, c3 = st.columns(3)
    with c1:
        st.plotly_chart(plot_sentiment_distribution(fg_df), use_container_width=True)
    with c2:
        st.plotly_chart(plot_pnl_distribution(df), use_container_width=True)
    with c3:
        st.plotly_chart(plot_leverage_histogram(df), use_container_width=True)

    # Row 2: Sentiment analysis
    st.subheader("Sentiment vs Performance")
    c1, c2 = st.columns(2)
    with c1:
        st.plotly_chart(plot_pnl_by_sentiment_box(df), use_container_width=True)
    with c2:
        st.plotly_chart(plot_leverage_by_sentiment_violin(df), use_container_width=True)

    # Row 3: Averages by sentiment
    c1, c2 = st.columns(2)
    with c1:
        st.plotly_chart(plot_avg_pnl_by_sentiment(df), use_container_width=True)
    with c2:
        st.plotly_chart(plot_avg_leverage_by_sentiment(df), use_container_width=True)

    # Row 4: Correlation & Scatter
    st.subheader("Correlation & Relationships")
    c1, c2 = st.columns(2)
    with c1:
        st.plotly_chart(plot_correlation_heatmap(df), use_container_width=True)
    with c2:
        st.plotly_chart(plot_leverage_vs_pnl_scatter(df), use_container_width=True)

    # Row 5: More scatter
    c1, c2 = st.columns(2)
    with c1:
        st.plotly_chart(plot_size_vs_pnl_scatter(df), use_container_width=True)
    with c2:
        st.plotly_chart(plot_event_distribution(df), use_container_width=True)

    # Row 6: Time series
    st.subheader("Time Series Analysis")
    c1, c2 = st.columns(2)
    with c1:
        st.plotly_chart(plot_daily_trade_counts(df), use_container_width=True)
    with c2:
        st.plotly_chart(plot_daily_pnl_trend(df), use_container_width=True)

    c1, c2 = st.columns(2)
    with c1:
        st.plotly_chart(plot_cumulative_pnl(df), use_container_width=True)
    with c2:
        st.plotly_chart(plot_rolling_7day_pnl(df), use_container_width=True)

    st.plotly_chart(plot_rolling_trade_volume(df), use_container_width=True)

    # Row 7: Traders
    st.subheader("Trader Leaderboards")
    c1, c2 = st.columns(2)
    with c1:
        st.plotly_chart(plot_top_profitable_traders(df), use_container_width=True)
    with c2:
        st.plotly_chart(plot_top_losing_traders(df), use_container_width=True)

    # Row 8: Symbol & Side
    c1, c2 = st.columns(2)
    with c1:
        st.plotly_chart(plot_symbol_pnl(df), use_container_width=True)
    with c2:
        st.plotly_chart(plot_side_analysis(df), use_container_width=True)


def render_statistical_section(stat_results: Dict):
    """Render statistical analysis results."""
    st.header("📐 Statistical Analysis")

    # Descriptive statistics table
    st.subheader("Descriptive Statistics by Sentiment")
    desc_df = stat_results["descriptive"][["mean", "50%", "std"]].copy()
    desc_df.columns = ["Mean PnL", "Median PnL", "Std Dev"]
    desc_df = desc_df.round(2)
    st.dataframe(desc_df, use_container_width=True)

    # Confidence intervals
    st.subheader("95% Confidence Intervals")
    ci_data = []
    for sentiment, (lower, upper) in stat_results["confidence_intervals"].items():
        ci_data.append({
            "Sentiment": sentiment,
            "Lower Bound": round(lower, 2),
            "Upper Bound": round(upper, 2),
            "Width": round(upper - lower, 2),
        })
    st.dataframe(pd.DataFrame(ci_data), use_container_width=True)

    # Correlations
    st.subheader("Correlation Analysis")
    col1, col2 = st.columns(2)
    with col1:
        p = stat_results["pearson"]
        st.metric("Pearson r", f"{p['r']:.4f}", f"p-value: {p['p_value']:.4e}")
    with col2:
        s = stat_results["spearman"]
        st.metric("Spearman rho", f"{s['rho']:.4f}", f"p-value: {s['p_value']:.4e}")

    # T-test
    ttest = stat_results.get("ttest_fear_greed")
    if ttest:
        st.subheader("T-Test: Fear vs Greed PnL")
        col1, col2, col3 = st.columns(3)
        col1.metric("t-statistic", f"{ttest['t_statistic']:.4f}")
        col2.metric("p-value", f"{ttest['p_value']:.4e}")
        col3.metric("Significant?", "Yes ✅" if ttest["significant"] else "No ❌")
        st.write(f"Fear mean PnL: ${ttest['fear_mean']:,.2f} (n={ttest['fear_n']:,})")
        st.write(f"Greed mean PnL: ${ttest['greed_mean']:,.2f} (n={ttest['greed_n']:,})")

    # ANOVA
    anova = stat_results.get("anova")
    if anova:
        st.subheader("ANOVA: PnL Across All Sentiment Classes")
        col1, col2, col3 = st.columns(3)
        col1.metric("F-statistic", f"{anova['f_statistic']:.4f}")
        col2.metric("p-value", f"{anova['p_value']:.4e}")
        col3.metric("Significant?", "Yes ✅" if anova["significant"] else "No ❌")

    # Full text results
    with st.expander("📋 Full Statistical Report"):
        st.text(format_statistical_results(stat_results))


def render_trader_section(tables: Dict[str, pd.DataFrame]):
    """Render trader analysis tables."""
    st.header("👤 Trader Analysis")

    tab_names = [
        "Best Traders", "Worst Traders", "Highest Leverage",
        "Lowest Leverage", "Highest Avg PnL", "Most Trades",
    ]
    tabs = st.tabs(tab_names)

    for i, (key, tab) in enumerate(zip(
        ["best_traders", "worst_traders", "highest_leverage",
         "lowest_leverage", "highest_avg_pnl", "most_trades"],
        tabs
    )):
        with tab:
            st.dataframe(tables[key], use_container_width=True, hide_index=True)


def render_insights_section(insights: List[str]):
    """Render market insights."""
    st.header("💡 Market Insights")
    for insight in insights:
        st.markdown(f"- {insight}")


def render_ml_section(ml_results: Dict):
    """Render machine learning results."""
    st.header("🤖 Machine Learning — Predict Profitable Trades")

    if "error" in ml_results:
        st.error(ml_results["error"])
        return

    # Metrics
    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("Accuracy", f"{ml_results['accuracy']:.4f}")
    col2.metric("Precision", f"{ml_results['precision']:.4f}")
    col3.metric("Recall", f"{ml_results['recall']:.4f}")
    col4.metric("F1 Score", f"{ml_results['f1']:.4f}")
    col5.metric("ROC AUC", f"{ml_results['roc_auc']:.4f}" if not np.isnan(ml_results['roc_auc']) else "N/A")

    # Confusion matrix
    st.subheader("Confusion Matrix")
    st.plotly_chart(ml_results["confusion_matrix_fig"], use_container_width=True)

    # Feature importance
    st.subheader("Feature Importance")
    st.plotly_chart(ml_results["feature_importance_fig"], use_container_width=True)

    # Classification report
    with st.expander("📋 Classification Report"):
        st.text(ml_results["classification_report"])


def render_export_section(df: pd.DataFrame, trader_tables: Dict, stat_results: Dict):
    """Render CSV export buttons."""
    st.header("📥 Export Data")

    col1, col2, col3 = st.columns(3)

    # Cleaned merged dataset
    with col1:
        csv_merged = df.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="💾 Download Cleaned Merged Dataset",
            data=csv_merged,
            file_name="merged_cleaned_data.csv",
            mime="text/csv",
        )

    # Trader summary
    with col2:
        trader_summary = trader_tables.get("best_traders", pd.DataFrame())
        if trader_summary.empty():
            # Combine all trader tables
            frames = []
            for name, tbl in trader_tables.items():
                tbl_copy = tbl.copy()
                tbl_copy["Category"] = name.replace("_", " ").title()
                frames.append(tbl_copy)
            if frames:
                trader_summary = pd.concat(frames, ignore_index=True)
        csv_trader = trader_summary.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="💾 Download Trader Summary",
            data=csv_trader,
            file_name="trader_summary.csv",
            mime="text/csv",
        )

    # Sentiment summary
    with col3:
        sent_summary = df.groupby(COL_SENTIMENT_CATEGORY).agg(
            avg_pnl=(COL_PNL, "mean"),
            median_pnl=(COL_PNL, "median"),
            total_pnl=(COL_PNL, "sum"),
            trade_count=(COL_PNL, "count"),
            avg_leverage=(COL_ESTIMATED_LEVERAGE, "mean"),
        ).reset_index()
        csv_sent = sent_summary.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="💾 Download Sentiment Summary",
            data=csv_sent,
            file_name="sentiment_summary.csv",
            mime="text/csv",
        )


# ============================================================================
# MAIN APPLICATION
# ============================================================================


@st.cache_data
def load_and_process_data(fg_path: str, hl_path: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Load both CSVs, engineer features, and return enriched DataFrames.

    This function is cached by Streamlit for performance.

    Args:
        fg_path: Path to Fear & Greed CSV.
        hl_path: Path to Hyperliquid historical data CSV.

    Returns:
        Tuple of (enriched_trades_df, fg_df).
    """
    fg_df = load_fear_greed_data(fg_path)
    hl_df = load_historical_data(hl_path)
    df = engineer_features(hl_df, fg_df)
    return df, fg_df


def main():
    """Main entry point for the Streamlit application."""
    init_streamlit_config()

    st.title("📊 Bitcoin Sentiment vs Hyperliquid Trader Performance")
    st.markdown(
        "An end-to-end analysis of the relationship between Bitcoin Fear & Greed "
        "Index sentiment and Hyperliquid trader performance."
    )

    # --- Determine data source ---
    # DEFAULT_FG_PATH / DEFAULT_HL_PATH may be None if no local files found anywhere
    local_files_available = DEFAULT_FG_PATH is not None and DEFAULT_HL_PATH is not None

    # Sidebar — upload section always visible
    st.sidebar.title("⚙️ Controls & Filters")
    st.sidebar.markdown("### 📁 Data Upload")

    if local_files_available:
        st.sidebar.success("✅ Local data files detected")
        st.sidebar.caption("Upload new files to override defaults")
    else:
        st.sidebar.warning("📤 No local data found — please upload both CSV files")
        st.sidebar.caption(
            "1. **fear_greed_index.csv** — columns: date, value, classification\n"
            "2. **historical_data.csv** — Hyperliquid trade data"
        )

    fg_upload = st.sidebar.file_uploader(
        "Upload Fear & Greed CSV", type=["csv"], key="fg_upload"
    )
    hl_upload = st.sidebar.file_uploader(
        "Upload Hyperliquid CSV", type=["csv"], key="hl_upload"
    )

    # Decide which data source to use
    use_uploads = fg_upload is not None and hl_upload is not None

    # If no local files and no uploads, show landing page
    if not local_files_available and not use_uploads:
        st.info(
            "👋 **Welcome!** To get started, upload your data files using the sidebar:\n\n"
            "1. **Fear & Greed Index CSV** — with columns: `date`, `value`, `classification`\n"
            "2. **Hyperliquid Historical Data CSV** — trade data with columns like "
            "`Account`, `Coin`, `Execution Price`, `Size USD`, `Side`, `Closed PnL`, `Timestamp IST`\n\n"
            "Both files are required to run the analysis."
        )
        # Show a sample layout even without data
        st.header("📊 Dashboard Preview")
        col1, col2, col3, col4, col5, col6 = st.columns(6)
        col1.metric("Total PnL", "—")
        col2.metric("Avg PnL/Trade", "—")
        col3.metric("Total Trades", "—")
        col4.metric("Win Rate", "—")
        col5.metric("Avg Leverage", "—")
        col6.metric("Unique Traders", "—")
        st.stop()

    # --- Load data ---
    with st.spinner("Loading and processing data... This may take a moment for large datasets."):
        try:
            if use_uploads:
                # Load from uploaded BytesIO objects
                fg_bytes = fg_upload.read()
                hl_bytes = hl_upload.read()
                fg_df = load_fear_greed_data(io.BytesIO(fg_bytes))
                hl_df = load_historical_data(io.BytesIO(hl_bytes))
                df = engineer_features(hl_df, fg_df)
            else:
                # Load from local file paths (cached)
                df, fg_df = load_and_process_data(DEFAULT_FG_PATH, DEFAULT_HL_PATH)
        except FileNotFoundError as e:
            st.error(
                f"Data file not found: {e}\n\n"
                "Please upload both CSV files using the sidebar."
            )
            st.stop()
        except Exception as e:
            st.error(f"Error loading data: {e}")
            st.stop()

    if df.empty:
        st.warning("No data available after processing. Please check your CSV files.")
        st.stop()

    # --- Now render the rest of the sidebar with filters ---
    st.sidebar.markdown("### 🔍 Filters")

    all_sentiments = sorted(df[COL_SENTIMENT_CATEGORY].unique().tolist())
    selected_sentiments = st.sidebar.multiselect(
        "Sentiment Filter", all_sentiments, default=all_sentiments
    )

    all_symbols = sorted(df[COL_SYMBOL].unique().tolist())
    top_symbols_default = all_symbols[:10] if len(all_symbols) > 10 else all_symbols
    selected_symbols = st.sidebar.multiselect(
        "Symbol Filter", all_symbols, default=top_symbols_default
    )

    all_accounts = df[COL_ACCOUNT].unique().tolist()
    selected_trader = st.sidebar.text_input(
        "Trader Filter (account prefix)", value=""
    )

    if COL_TRADE_DATE in df.columns:
        min_date = df[COL_TRADE_DATE].min()
        max_date = df[COL_TRADE_DATE].max()
        if pd.notna(min_date) and pd.notna(max_date):
            date_range = st.sidebar.date_input(
                "Date Range",
                value=(min_date, max_date),
                min_value=min_date,
                max_value=max_date,
            )
        else:
            date_range = None
    else:
        date_range = None

    filters = {
        "selected_sentiments": selected_sentiments,
        "selected_symbols": selected_symbols,
        "selected_trader": selected_trader,
        "date_range": date_range,
    }

    # --- Apply filters ---
    filtered_df = apply_filters(df, filters)

    if filtered_df.empty:
        st.warning("No data matches the selected filters. Please adjust your filter criteria.")
        st.stop()

    st.sidebar.markdown("---")
    st.sidebar.markdown(f"**Filtered records:** {len(filtered_df):,}")
    st.sidebar.markdown(f"**Total records:** {len(df):,}")

    # --- Render dashboard sections ---
    # KPIs
    st.header("📊 Key Performance Indicators")
    render_kpis(filtered_df)

    st.markdown("---")

    # EDA
    render_eda_section(filtered_df, fg_df)

    st.markdown("---")

    # Statistical Analysis
    stat_results = compute_statistical_analysis(filtered_df)
    render_statistical_section(stat_results)

    st.markdown("---")

    # Trader Analysis
    trader_tables = generate_trader_tables(filtered_df)
    render_trader_section(trader_tables)

    st.markdown("---")

    # Market Insights
    insights = generate_market_insights(filtered_df, stat_results)
    render_insights_section(insights)

    st.markdown("---")

    # Machine Learning
    ml_results = build_ml_model(filtered_df)
    render_ml_section(ml_results)

    st.markdown("---")

    # Export
    render_export_section(filtered_df, trader_tables, stat_results)


# ============================================================================
# ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    main()
