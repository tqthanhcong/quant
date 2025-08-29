import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time
import requests
from datetime import datetime, timedelta, date
from vnstock import Listing, Quote

# --- Data Loading ---
def get_symbols_by_group(group_name):
    """Get stock symbols by group name."""
    return Listing().symbols_by_group(group_name)

def fetch_history_for_symbols(symbols, start='2010-01-01', end='2025-08-25', interval='1D'):
    """Fetch historical data for a list of symbols."""
    all_data = []
    for symbol in symbols:
        retry = 0
        while retry < 5:
            try:
                df = Quote(symbol=symbol, source='VCI').history(start=start, end=end, interval=interval)
                df['symbol'] = symbol
                all_data.append(df)
                break
            except Exception as e:
                if "rate limit" in str(e).lower():
                    print(f"Rate limit hit, waiting 25s... ({symbol})")
                    time.sleep(25)
                    retry += 1
                else:
                    print(f"Error with {symbol}: {e}")
                    break
        time.sleep(1)
    result = pd.concat(all_data, ignore_index=True)
    result['time'] = pd.to_datetime(result['time'])
    return result

def filter_data_by_symbol(symbol, df, start=None, end=None):
    """Filter dataframe by symbol and date range."""
    df.index = pd.to_datetime(df.index)
    if start is None:
        start = df.index.min()
    if end is None:
        end = df.index.max()
    return df[(df['symbol'] == symbol) & (df.index >= start) & (df.index <= end)]

# --- MACD Calculation ---
def ema_tv(series: pd.Series, length: int) -> pd.Series:
    """Calculate EMA (TradingView style)."""
    s = pd.Series(series, dtype='float64').copy()
    if s.first_valid_index() is None:
        return pd.Series(np.nan, index=s.index)
    alpha = 2.0 / (length + 1.0)
    ema = np.full(len(s), np.nan)
    i0 = np.where(~np.isnan(s.values))[0][0]
    ema[i0] = s.iloc[i0]
    for i in range(i0 + 1, len(s)):
        ema[i] = alpha * s.iloc[i] + (1 - alpha) * ema[i - 1]
    return pd.Series(ema, index=s.index)

def calculate_macd(series: pd.Series, fast=12, slow=26, signal=9):
    """Calculate MACD, signal, and histogram."""
    ema_fast = ema_tv(series, fast)
    ema_slow = ema_tv(series, slow)
    macd = ema_fast - ema_slow
    macd_signal = ema_tv(macd, signal)
    hist = macd - macd_signal
    return macd, macd_signal, hist

# --- Extrema Finder ---
def find_hist_extrema(hist: pd.Series, strict_window=5, local_window=1):
    """Find strict and local extrema in MACD histogram."""
    s = pd.Series(hist, dtype='float64').copy()
    peaks_strict, troughs_strict, peaks_local, troughs_local = [], [], [], []
    for i in range(len(s)):
        if i >= strict_window and i < len(s) - strict_window:
            window = s.iloc[i - strict_window:i + strict_window + 1]
            if s.iloc[i] == window.max() and s.iloc[i] > 0:
                peaks_strict.append(s.index[i])
            if s.iloc[i] == window.min() and s.iloc[i] < 0:
                troughs_strict.append(s.index[i])
        if i >= local_window and i < len(s) - local_window:
            window = s.iloc[i - local_window:i + local_window + 1]
            if s.iloc[i] == window.max() and s.iloc[i] > 0:
                peaks_local.append(s.index[i])
            if s.iloc[i] == window.min() and s.iloc[i] < 0:
                troughs_local.append(s.index[i])
    return {
        "peaks_strict": peaks_strict,
        "troughs_strict": troughs_strict,
        "peaks_local": peaks_local,
        "troughs_local": troughs_local
    }

# --- Divergence Finder ---
def find_macd_divergence(df, fast=12, slow=26, signal=9, window_min=15, window=45):
    """Find bullish and bearish MACD divergences."""
    df = df.sort_index().copy()
    df['macd'], df['signal'], df['hist'] = calculate_macd(df['close'], fast, slow, signal)
    hist_ext = find_hist_extrema(df['hist'])
    bullish, bearish = [], []
    for t1 in hist_ext["troughs_strict"]:
        for t2 in hist_ext["troughs_local"]:
            if t2 <= t1 or (t2 - t1).days > window or (t2 - t1).days < window_min:
                continue
            mask = (df.index > t1) & (df.index < t2)
            if not mask.any():
                continue
            if (df.loc[mask, 'hist'] > 0).any():
                if df.loc[t2, 'close'] < df.loc[t1, 'close'] and df.loc[t2, 'hist'] > df.loc[t1, 'hist']:
                    bullish.append((t1, t2))
    for p1 in hist_ext["peaks_strict"]:
        for p2 in hist_ext["peaks_local"]:
            if p2 <= p1 or (p2 - p1).days > window or (p2 - p1).days < window_min:
                continue
            mask = (df.index > p1) & (df.index < p2)
            if not mask.any():
                continue
            if (df.loc[mask, 'hist'] < 0).any():
                if df.loc[p2, 'close'] > df.loc[p1, 'close'] and df.loc[p2, 'hist'] < df.loc[p1, 'hist']:
                    bearish.append((p1, p2))
    def remove_dup(df):
        return df[~df.index.duplicated(keep='first')]
    bull_df = pd.DataFrame(bullish, columns=['t1', 't2']).set_index('t2') if bullish else pd.DataFrame(columns=['t1'])
    bear_df = pd.DataFrame(bearish, columns=['p1', 'p2']).set_index('p2') if bearish else pd.DataFrame(columns=['p1'])
    return remove_dup(bull_df), remove_dup(bear_df), df, hist_ext

# --- Notification ---
def notify_discord(divergence_signals, webhook_url):
    """Send divergence signals to Discord."""
    for signal in divergence_signals:
        for d in signal['bullish_dates']:
            msg = f"🟢 bullish - {signal['symbol']} - {str(d)[:10]}"
            requests.post(webhook_url, json={"content": msg})
        for d in signal['bearish_dates']:
            msg = f"🔴 bearish - {signal['symbol']} - {str(d)[:10]}"
            requests.post(webhook_url, json={"content": msg})

# --- Main Logic ---
def main():
    start = (date.today() - timedelta(days=60)).strftime('%Y-%m-%d')
    end = date.today().strftime('%Y-%m-%d')
    symbols = get_symbols_by_group('VN100')
    data_all = fetch_history_for_symbols(symbols, start=start, end=end)
    data_all = data_all.set_index('time')
    end_date = pd.to_datetime(data_all.index.max())
    start_date = end_date - timedelta(days=3)
    divergence_signals = []
    for symbol in data_all['symbol'].unique():
        df_symbol = data_all[data_all['symbol'] == symbol].copy()
        if df_symbol.empty:
            continue
        df_symbol.index = pd.to_datetime(df_symbol.index)
        bull_df, bear_df, _, _ = find_macd_divergence(df_symbol)
        if not isinstance(bull_df.index, pd.DatetimeIndex):
            bull_df.index = pd.to_datetime(bull_df.index)
        if not isinstance(bear_df.index, pd.DatetimeIndex):
            bear_df.index = pd.to_datetime(bear_df.index)
        recent_bull = bull_df[(bull_df.index >= start_date) & (bull_df.index <= end_date)]
        recent_bear = bear_df[(bear_df.index >= start_date) & (bear_df.index <= end_date)]
        if not recent_bull.empty or not recent_bear.empty:
            divergence_signals.append({
                'symbol': symbol,
                'bullish_dates': recent_bull.index.tolist(),
                'bearish_dates': recent_bear.index.tolist()
            })
    divergence_signals = sorted(divergence_signals, key=lambda x: (len(x['bearish_dates']), len(x['bullish_dates'])))
    DISCORD_WEBHOOK_URL = "https://discord.com/api/webhooks/1410620919147134996/8nl-iTmIJ0lfeWcAt4UdVIAXADOBZhK_wPXFzuSY_GAoKMz1GsAsLkrTiDp8BnSA3VrJ"
    notify_discord(divergence_signals, DISCORD_WEBHOOK_URL)

if __name__ == "__main__":
    main()