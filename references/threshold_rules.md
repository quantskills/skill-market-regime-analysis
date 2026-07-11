# Threshold Rules for Regime Detection

Simple, interpretable rules for classifying market regimes using hard thresholds on observable data.

## Price Trend Regime

**Recommended window:** 20 trading days (~1 month) for short-term, 125 days (~6 months) for medium-term.

```python
def classify_trend(index_close, short_window=20, long_window=125):
    """
    index_close: pd.Series of daily close prices, index=date
    Returns: Series of ['Bull', 'Bear', 'Sideways']
    """
    ret_short = index_close.pct_change(short_window)
    ma_long = index_close.rolling(long_window).mean()
    above_ma = index_close > ma_long
    
    regime = pd.Series('Sideways', index=index_close.index)
    regime[(ret_short > 0.05) & above_ma] = 'Bull'
    regime[(ret_short < -0.05) & ~above_ma] = 'Bear'
    return regime
```

### Variants

| Variant | Bull Condition | Bear Condition | Use Case |
|---------|---------------|----------------|----------|
| Strict | ret_20d > 8% AND MA60↑ | ret_20d < -8% AND MA60↓ | Major trends only |
| Moderate | ret_20d > 5% AND MA125↑ | ret_20d < -5% AND MA125↓ | Default |
| Loose | ret_10d > 3% | ret_10d < -3% | Short-term trading |
| Acceleration | ret_20d > ret_60d > ret_125d | ret_20d < ret_60d < ret_125d | Momentum-based |

## Volatility Regime

```python
def classify_vol(index_close, window=20, hist_low=0.2, hist_high=0.8):
    """
    Annualized volatility percentile regime.
    """
    log_ret = np.log(index_close / index_close.shift(1))
    vol = log_ret.rolling(window).std() * np.sqrt(252)
    
    # Use expanding window percentiles to avoid look-ahead
    low_threshold = vol.expanding().quantile(hist_low)
    high_threshold = vol.expanding().quantile(hist_high)
    
    regime = pd.Series('Normal Vol', index=vol.index)
    regime[vol > high_threshold] = 'High Vol'
    regime[vol < low_threshold] = 'Low Vol'
    return regime, vol
```

**A-share typical volatility levels:**
- 沪深300 High Vol: > 30% annualized
- 沪深300 Low Vol: < 12% annualized
- 沪深300 Normal: 12%-30%

## Macro Regime (Expansion / Contraction / Transition)

```python
def classify_macro(pmi, gdp_growth, m2_growth):
    """
    pmi: monthly PMI series
    gdp_growth: quarterly GDP YoY series (forward-filled to daily)
    m2_growth: monthly M2 YoY series
    """
    regime = pd.Series('Transition', index=pmi.index)
    
    expansion = (pmi > 50) & (gdp_growth > gdp_growth.shift(4)) & (m2_growth > 10)
    contraction = (pmi < 50) | (gdp_growth < gdp_growth.shift(4))
    
    regime[expansion] = 'Expansion'
    regime[contraction] = 'Contraction'
    return regime
```

**Note:** Macro data has publication lag (GDP ~1 month, PMI ~end of month). Always use `info_date` parameter where available, or shift data by 1 month to avoid look-ahead bias.

## Combined Regime

Combine multiple dimensions into a single regime label:

```python
def combined_regime(trend_regime, vol_regime, macro_regime):
    """Priority: Vol > Macro > Trend"""
    if vol_regime == 'High Vol':
        return 'Crisis' if trend_regime == 'Bear' else 'High Vol'
    if trend_regime == 'Bull' and vol_regime == 'Low Vol':
        return 'Goldilocks'
    if trend_regime == 'Bear':
        return 'Bear Market'
    if trend_regime == 'Sideways' and macro_regime == 'Contraction':
        return 'Stagflation Watch'
    if trend_regime == 'Sideways':
        return 'Range-Bound'
    if trend_regime == 'Bull':
        return 'Bull Market'
    return 'Normal'
```

## Validation

Always validate threshold rules by:
1. Plotting regime classification over time with shaded regions
2. Checking regime transition count (too many = overfitting, too few = insensitive)
3. Computing per-regime return distribution (mean, std, skew, VaR)
4. Regime persistence: average duration of each regime state
