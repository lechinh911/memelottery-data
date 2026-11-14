"""Technical indicators used by the double top/bottom momentum strategy."""

from __future__ import annotations

import numpy as np
import pandas as pd


def ema(series: pd.Series, length: int) -> pd.Series:
    return series.ewm(span=length, adjust=False).mean()


def atr(df: pd.DataFrame, length: int = 14) -> pd.Series:
    high, low, close = df["high"], df["low"], df["close"]
    prev_close = close.shift(1)
    tr = pd.concat(
        [
            (high - low),
            (high - prev_close).abs(),
            (low - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    return tr.ewm(alpha=1 / length, adjust=False).mean()


def adx(df: pd.DataFrame, length: int = 14) -> pd.Series:
    high, low, close = df["high"], df["low"], df["close"]
    up = high.diff()
    down = -low.diff()
    plus_dm = np.where((up > down) & (up > 0), up, 0.0)
    minus_dm = np.where((down > up) & (down > 0), down, 0.0)
    tr = atr(df, length)  # smoothed TR via Wilder-like path below
    # Recalculate Wilder TR for ADX consistency
    prev_close = close.shift(1)
    true_range = pd.concat(
        [
            (high - low),
            (high - prev_close).abs(),
            (low - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    atr_w = true_range.ewm(alpha=1 / length, adjust=False).mean()
    plus_di = 100 * pd.Series(plus_dm, index=df.index).ewm(alpha=1 / length, adjust=False).mean() / atr_w
    minus_di = 100 * pd.Series(minus_dm, index=df.index).ewm(alpha=1 / length, adjust=False).mean() / atr_w
    dx = (100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)).fillna(0)
    return dx.ewm(alpha=1 / length, adjust=False).mean()


def pivot_high(high: pd.Series, left: int, right: int) -> pd.Series:
    """Confirmed pivot high at bar i-right when i is current (matches ta.pivothigh)."""
    n = len(high)
    out = np.full(n, np.nan)
    vals = high.to_numpy()
    for i in range(left + right, n):
        center = i - right
        window = vals[center - left : center + right + 1]
        if len(window) < left + right + 1:
            continue
        c = vals[center]
        if np.isnan(c):
            continue
        if c >= np.nanmax(window):
            # strict uniqueness: center must be unique max (TradingView allows ties as pivot)
            out[i] = c
    return pd.Series(out, index=high.index)


def pivot_low(low: pd.Series, left: int, right: int) -> pd.Series:
    n = len(low)
    out = np.full(n, np.nan)
    vals = low.to_numpy()
    for i in range(left + right, n):
        center = i - right
        window = vals[center - left : center + right + 1]
        if len(window) < left + right + 1:
            continue
        c = vals[center]
        if np.isnan(c):
            continue
        if c <= np.nanmin(window):
            out[i] = c
    return pd.Series(out, index=low.index)


def volume_sma(volume: pd.Series, length: int = 20) -> pd.Series:
    return volume.rolling(length).mean()


def rsi(series: pd.Series, length: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)
    avg_gain = gain.ewm(alpha=1 / length, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / length, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))
