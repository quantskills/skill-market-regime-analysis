# Conditional Analysis: Factors & Risk Per Regime

Evaluate how factors and risk metrics behave differently across regimes.

## Conditional Factor IC

```python
def conditional_ic(returns_df, factors_df, regimes):
    """
    Compute factor IC within each regime.
    
    Parameters
    ----------
    returns_df : pd.DataFrame, shape (T, N) — stock returns
    factors_df : pd.DataFrame, shape (T, N) — factor values (same stocks)
    regimes : pd.Series, shape (T,) — regime label per date
    
    Returns
    -------
    pd.DataFrame — {regime: {IC_mean, IC_std, IC_IR, hit_rate, n_days}}
    """
    results = {}
    
    for regime in regimes.unique():
        regime_dates = regimes[regimes == regime].index
        if len(regime_dates) < 20:  # minimum 20 days for reliable estimate
            continue
        
        ic_series = []
        for date in regime_dates:
            if date not in returns_df.index or date not in factors_df.index:
                continue
            ret_today = returns_df.loc[date]
            fac_today = factors_df.loc[date]
            valid = ret_today.notna() & fac_today.notna()
            if valid.sum() < 30:
                continue
            ic = ret_today[valid].rank().corr(fac_today[valid].rank())
            ic_series.append(ic)
        
        if not ic_series:
            continue
        
        ic_arr = np.array(ic_series)
        results[regime] = {
            'IC_mean': ic_arr.mean(),
            'IC_std': ic_arr.std(),
            'IC_IR': ic_arr.mean() / ic_arr.std() if ic_arr.std() > 0 else 0,
            't_stat': ic_arr.mean() / ic_arr.std() * np.sqrt(len(ic_arr)),
            'hit_rate': (ic_arr > 0).mean(),
            'n_days': len(ic_arr)
        }
    
    return pd.DataFrame(results).T
```

## Conditional Return & Risk

```python
def conditional_risk(index_returns, regimes, risk_free_rate=0.02):
    """
    Compute return/risk metrics per regime for an index or portfolio.
    
    index_returns: pd.Series of daily returns
    regimes: pd.Series of regime labels (same index)
    """
    joined = pd.DataFrame({'ret': index_returns, 'regime': regimes}).dropna()
    
    results = {}
    for regime, group in joined.groupby('regime'):
        daily_rets = group['ret']
        ann_ret = daily_rets.mean() * 252
        ann_vol = daily_rets.std() * np.sqrt(252)
        
        # Max drawdown within regime
        cum = (1 + daily_rets).cumprod()
        rolling_max = cum.expanding().max()
        drawdown = (cum / rolling_max - 1)
        
        results[regime] = {
            'n_days': len(daily_rets),
            'annualized_return': ann_ret,
            'annualized_vol': ann_vol,
            'sharpe': (ann_ret - risk_free_rate) / ann_vol if ann_vol > 0 else 0,
            'max_drawdown': drawdown.min(),
            'skew': daily_rets.skew(),
            'kurt': daily_rets.kurt(),
            'var_95': daily_rets.quantile(0.05),
            'cvar_95': daily_rets[daily_rets <= daily_rets.quantile(0.05)].mean(),
            'positive_ratio': (daily_rets > 0).mean()
        }
    
    return pd.DataFrame(results).T
```

## Regime Transition Matrix of Factors

```python
def factor_regime_dependency(factor_returns, regimes, n_lags=5):
    """
    Analyze if factor performance predicts regime transitions.
    factor_returns: pd.DataFrame {factor_name: daily_ret_series}
    regimes: pd.Series of regime labels
    
    Returns: Coef table of logistic regression
    """
    import statsmodels.api as sm
    
    # Create target: regime transition indicator
    regime_changed = (regimes != regimes.shift(1)).astype(int)
    
    results = {}
    for factor in factor_returns.columns:
        X = sm.add_constant(factor_returns[factor].shift(1).dropna())
        y = regime_changed.loc[X.index]
        
        model = sm.Logit(y, X).fit(disp=0)
        results[factor] = {
            'coef': model.params[factor],
            'p_value': model.pvalues[factor],
            'pseudo_r2': model.prsquared
        }
    
    return pd.DataFrame(results).T
```

## Conditional Correlation Matrix

```python
def conditional_corr(stock_returns, regimes, min_stocks=30):
    """
    Compute average pairwise stock correlation within each regime.
    Higher correlation = higher systemic risk.
    """
    results = {}
    
    for regime in regimes.unique():
        dates = regimes[regimes == regime].index
        regime_rets = stock_returns.loc[dates.intersection(stock_returns.index)]
        
        if len(regime_rets) < 20:
            continue
        
        corr_matrix = regime_rets.corr()
        upper_tri = corr_matrix.where(
            np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
        )
        avg_corr = upper_tri.stack().mean()
        
        results[regime] = {
            'avg_correlation': avg_corr,
            'n_stocks': regime_rets.shape[1],
            'n_days': regime_rets.shape[0]
        }
    
    return pd.DataFrame(results).T
```

## Output Report Template

```markdown
# Regime Conditional Analysis Report

## Regime Distribution
| Regime | Days | % of Total | Transition Count | Avg Duration |
|--------|------|-----------|-----------------|-------------|
| Bull   | 480  | 28.3%     | 8              | 60 days     |
| Bear   | 320  | 18.9%     | 6              | 53 days     |
| ...    | ...  | ...       | ...            | ...         |

## Factor IC by Regime
| Factor    | Overall IC | Bull IC | Bear IC | Sideways IC | High Vol IC |
|-----------|-----------|---------|---------|-------------|-------------|
| Momentum  | 0.034     | 0.062   | -0.021  | 0.028       | -0.045      |
| Value     | 0.028     | 0.015   | 0.058   | 0.032       | 0.041       |
| Low Vol   | 0.025     | 0.008   | 0.045   | 0.022       | 0.067       |
| Quality   | 0.031     | 0.022   | 0.038   | 0.035       | 0.025       |
| Growth    | 0.018     | 0.045   | -0.012  | 0.015       | -0.030      |

## Key Findings
- **Momentum** is only effective in Bull regimes, reverses in Bear
- **Value** and **Low Vol** are defensive — outperform in Bear and High Vol
- **Quality** is regime-agnostic — stable IC across all states
- High Vol regimes compress correlations (avg r rises from 0.25 to 0.55)
```
