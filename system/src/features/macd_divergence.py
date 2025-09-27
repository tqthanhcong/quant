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


def add_macd_divergence_column(
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


# Short alias
def macd_divergence(df: pd.DataFrame, **kwargs) -> pd.DataFrame:
    return add_macd_divergence_column(df, **kwargs)
