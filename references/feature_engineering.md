# Feature Engineering for Regime Detection

Compute regime features from raw panda_data outputs. Focus on date alignment, rolling calculations, and avoiding look-ahead.

## Date Alignment

panda_data APIs return timestamps. Align all features to a common trading day index.

```python
def align_features(features_dict, trade_dates):
    """
    Align all features to trade_dates index.
    features_dict: {name: pd.Series or DataFrame} with mixed date indices
    trade_dates: pd.DatetimeIndex of all trading days
    Returns: pd.DataFrame with all features on common index
    """
    aligned = pd.DataFrame(index=trade_dates)
    for name, series in features_dict.items():
        # Forward fill macro data (data published today is available after publication)
        aligned[name] = series.reindex(trade_dates).ffill()
    return aligned
```

## Rolling Window Features

```python
def rolling_features(close_prices, windows=[5, 10, 21, 63, 125]):
    """
    Compute standard rolling features from closing prices.
    """
    log_ret = np.log(close_prices / close_prices.shift(1))
    features = pd.DataFrame(index=close_prices.index)
    
    for w in windows:
        features[f'ret_{w}d'] = close_prices.pct_change(w)
        features[f'vol_{w}d'] = log_ret.rolling(w).std() * np.sqrt(252)
        features[f'skew_{w}d'] = log_ret.rolling(w).skew()
        features[f'kurt_{w}d'] = log_ret.rolling(w).kurt()
    
    # Drawdown
    features['max_dd_63d'] = 1 - close_prices / close_prices.rolling(63).max()
    features['max_dd_252d'] = 1 - close_prices / close_prices.rolling(252).max()
    
    # Distance from moving average
    for w in windows:
        features[f'dist_ma_{w}'] = close_prices / close_prices.rolling(w).mean() - 1
    
    return features
```

## Term Structure Features

```python
def term_structure_features(term_df, contract_pairs):
    """
    Convert raw term structure data to features.
    term_df: DataFrame with columns [date, contract, close, settle, oi]
    contract_pairs: [(near_contract, far_contract), ...]
    Returns: DataFrame of basis features
    """
    features = pd.DataFrame()
    
    for near, far in contract_pairs:
        near_df = term_df[term_df['contract'] == near][['date', 'close']]
        far_df = term_df[term_df['contract'] == far][['date', 'close']]
        near_df = near_df.set_index('date')
        far_df = far_df.set_index('date')
        
        merged = near_df.join(far_df, how='inner', lsuffix='_near', rsuffix='_far')
        features[f'basis_{near}_{far}'] = (merged['close_near'] / merged['close_far'] - 1)
    
    return features
```

## Market Breadth & Sentiment Features

```python
def breadth_features(stock_daily_data_dict, trade_dates):
    """
    Compute market breadth from all-stock daily data.
    stock_daily_data_dict: {symbol: DataFrame with 'close', 'open', 'volume', 'high', 'low'}
    Returns: DataFrame of daily breadth features
    """
    # Build panel
    closes = pd.DataFrame({s: d['close'] for s, d in stock_daily_data_dict.items() if not d.empty})
    opens = pd.DataFrame({s: d['open'] for s, d in stock_daily_data_dict.items() if not d.empty})
    volumes = pd.DataFrame({s: d['volume'] for s, d in stock_daily_data_dict.items() if not d.empty})
    highs = pd.DataFrame({s: d['high'] for s, d in stock_daily_data_dict.items() if not d.empty})
    lows = pd.DataFrame({s: d['low'] for s, d in stock_daily_data_dict.items() if not d.empty})
    
    features = pd.DataFrame(index=closes.index)
    
    # Advanced/Decline ratio
    features['ad_ratio'] = (closes > opens).sum(axis=1) / closes.notna().sum(axis=1)
    
    # Above MA ratio
    for w in [5, 20, 60]:
        ma = closes.rolling(w).mean()
        features[f'above_ma_{w}'] = (closes > ma).sum(axis=1) / closes.notna().sum(axis=1)
    
    # Volume relative to MA
    vol_ma20 = volumes.rolling(20).mean()
    features['volume_ratio'] = (volumes.iloc[-1] if len(volumes) > 0 else 0).mean()
    
    # A-Share unique: ST stock ratio
    # Skipped - panda_data.get_stock_daily with st=True/False handles this
    
    return features
```

## Macro Feature Processing

```python
def macro_features(gdp_df, pmi_df, cpi_df, m2_df, trade_dates):
    """
    Process macro data into regime features.
    All macro data: forward-filled from publication date.
    """
    features = pd.DataFrame(index=trade_dates)
    
    # GDP: quarterly, forward-fill to daily
    if gdp_df is not None and 'gdp_yoy' in gdp_df.columns:
        features['gdp_yoy'] = gdp_df['gdp_yoy'].reindex(trade_dates).ffill()
        features['gdp_accel'] = features['gdp_yoy'].diff(4)  # YoY acceleration
    
    # PMI: monthly
    if pmi_df is not None and 'pmi' in pmi_df.columns:
        features['pmi'] = pmi_df['pmi'].reindex(trade_dates).ffill()
        features['pmi_above_50'] = (features['pmi'] > 50).astype(int)
    
    # CPI: monthly
    if cpi_df is not None and 'cpi_yoy' in cpi_df.columns:
        features['cpi_yoy'] = cpi_df['cpi_yoy'].reindex(trade_dates).ffill()
    
    # M2: monthly
    if m2_df is not None and 'm2_yoy' in m2_df.columns:
        features['m2_yoy'] = m2_df['m2_yoy'].reindex(trade_dates).ffill()
    
    return features
```

## Complete Feature Assembly

```python
def assemble_regime_features(index_close, index_data=None, term_data=None,
                              macro_data=None, stock_dict=None):
    """
    Assemble all regime features into a single aligned DataFrame.
    Returns: (features_df, feature_names)
    """
    trade_dates = index_close.index
    features = pd.DataFrame(index=trade_dates)
    
    # 1. Price-based features
    price_feats = rolling_features(index_close)
    features = features.join(price_feats, how='left')
    
    # 2. Valuation features (PE/PB percentile)
    if index_data is not None and 'pe' in index_data.columns:
        features['pe_percentile'] = index_data['pe'].rolling(1260).rank(pct=True)
    
    # 3. Term structure
    if term_data is not None:
        term_feats = term_structure_features(term_data)
        features = features.join(term_feats, how='left')
    
    # 4. Macro features
    if macro_data is not None:
        features = features.join(macro_data, how='left')
    
    # 5. Market breadth
    if stock_dict is not None and len(stock_dict) > 0:
        breadth = breadth_features(stock_dict)
        features = features.join(breadth, how='left')
    
    # Drop NaN rows (first 125 days for rolling calcs)
    features = features.dropna(how='all')
    
    # Select a clean subset for HMM (no NaN left)
    hmm_features = features.dropna()
    
    return features, hmm_features
```
