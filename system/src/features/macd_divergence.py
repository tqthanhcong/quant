"""
Minimal MACD divergence feature.

This module exposes a single function that takes a DataFrame with columns
at least: [time, symbol, open, high, low, close, volume] (one symbol assumed),
and returns the same DataFrame with a new column indicating MACD divergence
signals at detected pivot endpoints:
  - 1 for bullish divergence
  - -1 for bearish divergence
  - 0 otherwise

No external I/O. No cross-file imports. Keep it simple.
"""

from __future__ import annotations

import pandas as pd
import numpy as np

__all__ = ["add_macd_divergence_column", "macd_divergence"]


def _ema(series: pd.Series, length: int) -> pd.Series:
    s = pd.Series(series, dtype="float64")
    return s.ewm(span=length, adjust=False, min_periods=1).mean()


def _macd(series: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> tuple[pd.Series, pd.Series, pd.Series]:
    ema_fast = _ema(series, fast)
    ema_slow = _ema(series, slow)
    macd = ema_fast - ema_slow
    macd_signal = _ema(macd, signal)
    hist = macd - macd_signal
    return macd, macd_signal, hist


def _local_extrema_indices(x: pd.Series, window: int = 3) -> tuple[list[int], list[int]]:
    """Return indices of simple local peaks (>0) and troughs (<0) using a small window."""
    peaks: list[int] = []
    troughs: list[int] = []
    n = len(x)
    if n == 0:
        return peaks, troughs
    w = max(1, int(window))
    for i in range(w, n - w):
        seg = x.iloc[i - w : i + w + 1]
        xi = x.iloc[i]
        if np.isfinite(xi):
            if xi > 0 and xi == seg.max():
                peaks.append(i)
            if xi < 0 and xi == seg.min():
                troughs.append(i)
    return peaks, troughs

def add_ema(df,fast=12,slow=26):
    df['ema_fast'] = _ema(df['close'], fast)
    df['ema_slow'] = _ema(df['close'], slow)
    return df

def add_macd(df,fast=12,slow=26,signal=9):
    macd, macd_signal, hist = _macd(df['close'], fast=fast, slow=slow, signal=signal)
    df['macd'] = macd
    df['macd_signal'] = macd_signal
    df['macd_hist'] = hist
    return df

def add_macd_local_extrema(df,local_window=3):
    peaks, troughs = _local_extrema_indices(df['macd_hist'], window=local_window)
    df['macd_peaks'] = 0
    df['macd_troughs'] = 0
    df.loc[peaks, 'macd_peaks'] = 1
    df.loc[troughs, 'macd_troughs'] = -1
    return df

def add_divergence_signals(df,price_col='close'):
    df['macd_divergence'] = 0
    # Bullish: price makes lower low, histogram makes higher low (troughs)
    troughs = df.index[df['macd_troughs'] == -1].tolist()
    for j in range(1, len(troughs)):
        i1, i2 = troughs[j - 1], troughs[j]
        if not (np.isfinite(df['macd_hist'].iloc[i1]) and np.isfinite(df['macd_hist'].iloc[i2])):
            continue
        if df[price_col].iloc[i2] < df[price_col].iloc[i1] and df['macd_hist'].iloc[i2] > df['macd_hist'].iloc[i1]:
            df.at[i2, 'macd_divergence'] = 1

    # Bearish: price makes higher high, histogram makes lower high (peaks)
    peaks = df.index[df['macd_peaks'] == 1].tolist()
    for j in range(1, len(peaks)):
        i1, i2 = peaks[j - 1], peaks[j]
        if not (np.isfinite(df['macd_hist'].iloc[i1]) and np.isfinite(df['macd_hist'].iloc[i2])):
            continue
        if df[price_col].iloc[i2] > df[price_col].iloc[i1] and df['macd_hist'].iloc[i2] < df['macd_hist'].iloc[i1]:
            # If both bullish and bearish hit same bar, prefer bearish (-1)
            df.at[i2, 'macd_divergence'] = -1
    return df

def add_macd_divergence_old(
    df: pd.DataFrame,
    *,
    price_col: str = "close",
    time_col: str = "time",
    symbol_col: str = "symbol",
    out_col: str = "macd_divergence",
    fast: int = 12,
    slow: int = 26,
    signal: int = 9,
    local_window: int = 3,
) -> pd.DataFrame:
    """Append a simple MACD divergence signal column to the given DataFrame.

    Assumptions:
    - Data contains only one symbol (this function will ignore/keep the symbol column as-is)
    - Input includes time column; rows will be sorted by time before processing
    - Output column values: -1 (bearish), 1 (bullish), 0 (none)
    """

    if price_col not in df.columns:
        raise ValueError(f"Missing price column '{price_col}' in input DataFrame")

    out = df.copy()

    # Ensure datetime and chronological order
    if time_col in out.columns:
        out[time_col] = pd.to_datetime(out[time_col])
        out = out.sort_values(time_col).reset_index(drop=True)

    # Compute MACD histogram (no intermediate columns kept)
    _, _, hist = _macd(out[price_col].astype(float), fast=fast, slow=slow, signal=signal)

    # Find simple local extrema on histogram
    peaks, troughs = _local_extrema_indices(hist, window=local_window)

    # Prepare signal column
    signal_arr = np.zeros(len(out), dtype=int)

    # Check consecutive extrema pairs for divergence
    # Bullish: price makes lower low, histogram makes higher low (troughs)
    for j in range(1, len(troughs)):
        i1, i2 = troughs[j - 1], troughs[j]
        if not (np.isfinite(hist.iloc[i1]) and np.isfinite(hist.iloc[i2])):
            continue
        if out[price_col].iloc[i2] < out[price_col].iloc[i1] and hist.iloc[i2] > hist.iloc[i1]:
            signal_arr[i2] = 1

    # Bearish: price makes higher high, histogram makes lower high (peaks)
    for j in range(1, len(peaks)):
        i1, i2 = peaks[j - 1], peaks[j]
        if not (np.isfinite(hist.iloc[i1]) and np.isfinite(hist.iloc[i2])):
            continue
        if out[price_col].iloc[i2] > out[price_col].iloc[i1] and hist.iloc[i2] < hist.iloc[i1]:
            # If both bullish and bearish hit same bar, prefer bearish (-1)
            signal_arr[i2] = -1

    out[out_col] = signal_arr
    return out