import pandas as pd
import numpy as np

def clean_data(df: pd.DataFrame, time_col = 'time') -> pd.DataFrame:
    """Clean data by removing duplicates and sorting by time."""
    df = df.drop_duplicates().reset_index(drop=True)
    df = df.sort_values(by=time_col).reset_index(drop=True)
    return df
