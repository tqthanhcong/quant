import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta, date
from statsmodels.tsa.stattools import adfuller
import statsmodels.api as sm
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta, date
from statsmodels.tsa.stattools import adfuller
import statsmodels.api as sm
from datetime import timedelta
import time
from vnstock import Listing, Quote, Company, Finance, Trading, Screener 
# load data
def get_symbols_by_group(group_name):
    listing = Listing()
    return listing.symbols_by_group(group_name)


def fetch_history_for_symbols(symbols, start='2010-01-01', end='2025-08-25', interval='1D'):
    all_data = []
    for i, symbol in enumerate(symbols):
        retry = 0
        while retry < 5:
            try:
                quote = Quote(symbol=symbol, source='VCI')
                df = quote.history(start=start, end=end, interval=interval)
                df['symbol'] = symbol
                all_data.append(df)
                break
            except Exception as e:
                if "rate limit" in str(e).lower():
                    print(f"Rate limit hit, waiting 25s... ({symbol})")
                    time.sleep(25)
                    retry += 1
                else:
                    print(f"Lỗi với {symbol}: {e}")
                    break
        time.sleep(1)  # Thêm delay nhỏ giữa các symbol
    result = pd.concat(all_data, ignore_index=True)
    result['time'] = pd.to_datetime(result['time'])
    return result

def save_to_csv(df, filepath):
    df.to_csv(filepath, index=False)
    print(f"Đã lưu dữ liệu vào {filepath}")

def filter_data_by_symbol(symbol, df, start=None, end=None):
    df.index = pd.to_datetime(df.index)
    if start is None:
        start = df.index.min()
    if end is None:
        end = df.index.max()
    return df[(df['symbol'] == symbol) & (df.index >= start) & (df.index <= end)]
start = (date.today() - timedelta(days=60)).strftime('%Y-%m-%d')
end = date.today().strftime('%Y-%m-%d')
symbols = get_symbols_by_group('VN100')
data_all = fetch_history_for_symbols(symbols, start=start, end=end)
data_all = data_all.set_index('time')
data_all.head()
# macd divergence
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ---------- EMA & MACD ----------
def ema_tv(series: pd.Series, length: int) -> pd.Series:
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
    ema_fast = ema_tv(series, fast)
    ema_slow = ema_tv(series, slow)
    macd = ema_fast - ema_slow
    macd_signal = ema_tv(macd, signal)
    hist = macd - macd_signal
    return macd, macd_signal, hist

# ---------- Extrema Finder ----------
def find_hist_extrema(hist: pd.Series, strict_window=5, local_window=1):
    s = pd.Series(hist, dtype='float64').copy()
    peaks_strict, troughs_strict = [], []
    peaks_local, troughs_local = [], []

    for i in range(len(s)):
        # strict extrema (t1/p1) = 5 nến trước + 5 nến sau
        if i >= strict_window and i < len(s) - strict_window:
            window = s.iloc[i - strict_window:i + strict_window + 1]
            if s.iloc[i] == window.max() and s.iloc[i] > 0:
                peaks_strict.append(s.index[i])
            if s.iloc[i] == window.min() and s.iloc[i] < 0:
                troughs_strict.append(s.index[i])

        # local extrema (t2/p2) = 1 nến trước + 1 nến sau
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

# ---------- Divergence Finder ----------
def find_macd_divergence(df, fast=12, slow=26, signal=9, window_min = 15, window=45):
    df = df.sort_index().copy()
    df['macd'], df['signal'], df['hist'] = calculate_macd(df['close'], fast, slow, signal)

    hist_ext = find_hist_extrema(df['hist'])
    bullish, bearish = [], []

    # bullish divergence: t1 strict, t2 local
    for t1 in hist_ext["troughs_strict"]:
        for t2 in hist_ext["troughs_local"]:
            if t2 <= t1: 
                continue
            if (t2 - t1).days > window: 
                continue
            if (t2 - t1).days < window_min:
                continue
            # boolean mask để lấy hist giữa t1 và t2
            mask = (df.index > t1) & (df.index < t2)
            if not mask.any():  # nếu không có cây nào giữa
                continue
            if (df.loc[mask, 'hist'] > 0).any():  # ít nhất 1 cây > 0 (cross above zero, per Elder)
                if df.loc[t2, 'close'] < df.loc[t1, 'close'] and df.loc[t2, 'hist'] > df.loc[t1, 'hist']:
                    bullish.append((t1, t2))

    # bearish divergence: p1 strict, p2 local
    for p1 in hist_ext["peaks_strict"]:
        for p2 in hist_ext["peaks_local"]:
            if p2 <= p1: 
                continue
            if (p2 - p1).days > window: 
                continue
            if (p2 - p1).days < window_min: 
                continue
            mask = (df.index > p1) & (df.index < p2)
            if not mask.any():
                continue
            if (df.loc[mask, 'hist'] < 0).any():  # ít nhất 1 cây < 0 (cross below zero, per Elder)
                if df.loc[p2, 'close'] > df.loc[p1, 'close'] and df.loc[p2, 'hist'] < df.loc[p1, 'hist']:
                    bearish.append((p1, p2))


    bull_df = pd.DataFrame(bullish, columns=['t1','t2']).set_index('t2') if bullish else pd.DataFrame(columns=['t1'])
    bear_df = pd.DataFrame(bearish, columns=['p1','p2']).set_index('p2') if bearish else pd.DataFrame(columns=['p1'])
    def remove_dup(df):
      df = df[~df.index.duplicated(keep='first')]
      return df
    bull_df = remove_dup(bull_df)
    bear_df = remove_dup(bear_df)

    return bull_df, bear_df, df, hist_ext

# ---------- Visualization ----------
def visualize_divergence(df, bullish, bearish, hist_ext, annotate=True):
    fig, (ax1, ax2) = plt.subplots(2,1, figsize=(14,10), sharex=True,
                                   gridspec_kw={'height_ratios':[2,1]})

    ax1.plot(df.index, df['close'], label='Close', color='black')
    ax2.bar(df.index, df['hist'], alpha=0.5, label='Histogram')
    ax2.plot(df.index, df['macd'], label='MACD')
    ax2.plot(df.index, df['signal'], label='Signal', linestyle='--')
    ax2.axhline(0, color='black', lw=0.7)

    # mark extrema
    ax2.scatter(hist_ext["peaks_strict"], df.loc[hist_ext["peaks_strict"], 'hist'], marker='^', color='red', s=80, label='Peak strict')
    ax2.scatter(hist_ext["troughs_strict"], df.loc[hist_ext["troughs_strict"], 'hist'], marker='v', color='green', s=80, label='Trough strict')
    ax2.scatter(hist_ext["peaks_local"], df.loc[hist_ext["peaks_local"], 'hist'], marker='^', color='orange', s=60, label='Peak local')
    ax2.scatter(hist_ext["troughs_local"], df.loc[hist_ext["troughs_local"], 'hist'], marker='v', color='blue', s=60, label='Trough local')

    # bullish
    for t2, row in bullish.iterrows():
        t1 = row['t1']
        ax1.plot([t1,t2], [df.loc[t1,'close'], df.loc[t2,'close']], color='green', lw=2)
        ax2.plot([t1,t2], [df.loc[t1,'hist'], df.loc[t2,'hist']], color='green', lw=2)
        if annotate: ax1.text(t2, df.loc[t2,'close'], 'Bull', color='green')

    # bearish
    for p2, row in bearish.iterrows():
        p1 = row['p1']
        ax1.plot([p1,p2], [df.loc[p1,'close'], df.loc[p2,'close']], color='red', lw=2)
        ax2.plot([p1,p2], [df.loc[p1,'hist'], df.loc[p2,'hist']], color='red', lw=2)
        if annotate: ax1.text(p2, df.loc[p2,'close'], 'Bear', color='red')

    ax1.legend()
    ax2.legend()
    plt.tight_layout()
    return fig, ax1, ax2


# noti
# Xác định khoảng thời gian 3 ngày gần nhất
end_date = pd.to_datetime(data_all.index.max())
start_date = end_date - timedelta(days=3)

symbols = data_all['symbol'].unique()
divergence_signals = []

for symbol in symbols:
    df_symbol = data_all[data_all['symbol'] == symbol]
    if df_symbol.empty:
        continue
    # Ensure the index is datetime for date arithmetic
    df_symbol = df_symbol.copy()
    df_symbol.index = pd.to_datetime(df_symbol.index)
    bull_df, bear_df, df_macd, hist_ext = find_macd_divergence(df_symbol)
    # Lọc các tín hiệu xuất hiện trong 7 ngày gần nhất
    recent_bull = bull_df[(bull_df.index >= start_date) & (bull_df.index <= end_date)]
    recent_bear = bear_df[(bear_df.index >= start_date) & (bear_df.index <= end_date)]
    if not recent_bull.empty or not recent_bear.empty:
        divergence_signals.append({
            'symbol': symbol,
            'bullish_dates': recent_bull.index.tolist(),
            'bearish_dates': recent_bear.index.tolist()
        })

# Hiển thị kết quả
for signal in divergence_signals:
    print(f"Symbol: {signal['symbol']}")
    if signal['bullish_dates']:
        print(f"  Bullish divergence at: {signal['bullish_dates']}")
    if signal['bearish_dates']:
        print(f"  Bearish divergence at: {signal['bearish_dates']}")
    print('-'*40)
import requests

DISCORD_WEBHOOK_URL = "https://discord.com/api/webhooks/1410620919147134996/8nl-iTmIJ0lfeWcAt4UdVIAXADOBZhK_wPXFzuSY_GAoKMz1GsAsLkrTiDp8BnSA3VrJ"  # Thay bằng webhook của bạn

for signal in divergence_signals:
    for d in signal['bullish_dates']:
        msg = f"🟢 bullish - {signal['symbol']} - {str(d)[:10]}"
        requests.post(DISCORD_WEBHOOK_URL, json={"content": msg})
    for d in signal['bearish_dates']:
        msg = f"🔴 bearish - {signal['symbol']} - {str(d)[:10]}"
        requests.post(DISCORD_WEBHOOK_URL, json={"content": msg})