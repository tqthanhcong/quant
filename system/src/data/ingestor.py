import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time
from datetime import date
from vnstock import Listing, Quote


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
    result.set_index('time', inplace=True)
    return result

def merge_data_to_csv(new_data, csv_path):
    """Merge new data with existing CSV data and save.
    Uniqueness key: (time index, symbol). 'time' is the index.
    """
    # Ensure 'time' is the index and named correctly
    if new_data.index.name != 'time':
        new_data = new_data.copy()
        new_data.index.name = 'time'

    try:
        existing_data = pd.read_csv(csv_path, parse_dates=['time'], index_col='time')
    except FileNotFoundError:
        combined_data = new_data.sort_index()
    else:
        combined_data = pd.concat([existing_data, new_data], axis=0)
        combined_data = (
            combined_data
            .reset_index()
            .drop_duplicates(subset=['time', 'symbol'], keep='last')
            .set_index('time')
            .sort_index()
        )

    combined_data.to_csv(csv_path, index=True)
    return combined_data

def get_data_by_symbols_and_save(symbols, csv_path, start=None, end=date.today().strftime('%Y-%m-%d')):
    """Fetch data for symbols, filter by date range, and save to CSV."""
    all_data = fetch_history_for_symbols(symbols, start=start, end=end)
    merged_data = merge_data_to_csv(all_data, csv_path)
    print(f"Data saved to {csv_path}. Total rows now: {len(merged_data)}.")