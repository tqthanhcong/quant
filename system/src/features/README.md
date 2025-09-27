Single vs. Multi-Symbol TA modules

- ta_single.py: Clean, single-symbol implementations. Each function assumes one symbol and focuses only on calculation.
- ta_multi.py: Thin wrappers that apply the single-symbol functions across symbols.

Core functions (single):
- clean_ohlcv(df)
- add_ema(df, length, price_col='close')
- add_macd(df, fast=12, slow=26, signal=9, price_col='close')
- add_macd_extrema(df, hist_col='macd_hist', strict_window=5, local_window=1)
- add_macd_divergence(df, ..., price_col='close', hist_col='macd_hist')
- compute_macd_features(df, ...)  # one-shot pipeline

Wrappers (multi): same names and signatures, operating on multi-symbol DataFrames.

Migration: prefer importing from `.ta_single` or `.ta_multi` instead of the legacy `macd_divergence.py`.
