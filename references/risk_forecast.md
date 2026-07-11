# Regime-Aware Risk Forecasting

Generate risk forecasts that adapt to the current market regime.

## Conditional Covariance Estimation

```python
def conditional_covariance(returns_df, regimes, current_regime, decay_halflife=60):
    """
    Estimate covariance matrix using only data from same regime.
    
    Parameters
    ----------
    returns_df : pd.DataFrame, shape (T, N) — asset returns
    regimes : pd.Series, shape (T,) — regime labels
    current_regime : str — the current regime label
    decay_halflife : int — exponential decay halflife in days
    
    Returns
    -------
    pd.DataFrame — conditional covariance matrix
    """
    regime_dates = regimes[regimes == current_regime].index
    regime_rets = returns_df.loc[regime_dates.intersection(returns_df.index)]
    
    if len(regime_rets) < 60:
        # Fall back to full-sample with regime weight adjustment
        return _weighted_covariance(returns_df, regimes, current_regime, decay_halflife)
    
    # Exponential weighted covariance on regime-specific data
    weights = np.exp(-np.log(2) * np.arange(len(regime_rets))[::-1] / decay_halflife)
    weights /= weights.sum()
    
    mean = (regime_rets * weights[:, np.newaxis]).sum(axis=0)
    centered = regime_rets - mean
    cov = (centered * weights[:, np.newaxis]).T @ centered
    cov /= 1 - (weights ** 2).sum()  # bias correction
    
    cov.index = returns_df.columns
    cov.columns = returns_df.columns
    return cov


def _weighted_covariance(returns_df, regimes, current_regime, halflife=60):
    """Use full sample but weight same-regime days more."""
    weights = pd.Series(0.0, index=returns_df.index)
    same_regime = (regimes == current_regime)
    weights[same_regime] = 1.0
    weights[~same_regime] = 0.3  # other regimes get 30% weight
    
    # Apply time decay
    time_weight = np.exp(-np.log(2) * np.arange(len(returns_df))[::-1] / halflife)
    weights = weights * time_weight
    weights /= weights.sum()
    
    mean = (returns_df * weights[:, np.newaxis]).sum(axis=0)
    centered = returns_df - mean
    cov = (centered * weights[:, np.newaxis]).T @ centered
    cov /= 1 - (weights ** 2).sum()
    
    return cov
```

## Conditional VaR / CVaR

```python
def conditional_var(portfolio_returns, regimes, alpha=0.05):
    """
    Estimate VaR and CVaR per regime using historical simulation.
    """
    joined = pd.DataFrame({'ret': portfolio_returns, 'regime': regimes}).dropna()
    
    results = {}
    for regime, group in joined.groupby('regime'):
        rets = group['ret'].values
        results[regime] = {
            'VaR_95': np.percentile(rets, 5),
            'CVaR_95': rets[rets <= np.percentile(rets, 5)].mean(),
            'VaR_99': np.percentile(rets, 1),
            'CVaR_99': rets[rets <= np.percentile(rets, 1)].mean(),
            'semi_std': rets[rets < 0].std(),  # downside deviation
            'n_days': len(rets)
        }
    
    return pd.DataFrame(results).T
```

## Regime-Aware Volatility Scaling

```python
def regime_vol_scaling(current_regime, target_vol=0.15, regime_vols=None):
    """
    Scale position sizes to target volatility given current regime.
    
    regime_vols: dict {regime: expected_annualized_vol}
    """
    if regime_vols is None:
        regime_vols = {
            'Bull': 0.18,
            'Bear': 0.35,
            'Sideways': 0.15,
            'High Vol': 0.45,
            'Low Vol': 0.10,
            'Crisis': 0.50,
            'Goldilocks': 0.12
        }
    
    expected_vol = regime_vols.get(current_regime, 0.20)
    scale = min(1.0, target_vol / expected_vol)
    
    return {
        'scale_factor': scale,
        'expected_vol': expected_vol,
        'target_vol': target_vol,
        'max_leverage': scale * 1.0
    }
```

## Stress Scenarios

Test current portfolio against historical regime episodes:

```python
def stress_scenarios(portfolio_returns, index_returns, beta):
    """
    Apply historical regime shocks to current portfolio.
    Returns worst-case return over each scenario.
    """
    scenarios = {
        '2015_Crash': {'index_return': -0.324},  # 2015H2 crisis
        '2018_FullYear': {'index_return': -0.253},  # 2018 bear
        '2020_COVID': {'index_return': -0.119},  # Feb 2020 COVID crash
        '2022_Gloom': {'index_return': -0.216},  # 2022 bear
    }
    
    results = {}
    for scenario, params in scenarios.items():
        # Simple linear: portfolio_return = beta * index_return
        scenario_return = beta * params['index_return']
        
        # More precise: use actual historical returns during the scenario
        results[scenario] = {
            'portfolio_impact': scenario_return,
            'index_return': params['index_return']
        }
    
    return pd.DataFrame(results).T
```

## Regime Transition Probability & Risk Timeline

```python
def regime_transition_risk(transition_matrix, current_state, forecast_days=[5, 20, 60]):
    """
    Compute probability of staying in current regime over forecast horizons.
    """
    tm = np.array(transition_matrix)
    n_states = tm.shape[0]
    
    # Initial state vector (one-hot)
    state_vec = np.zeros(n_states)
    state_vec[current_state] = 1.0
    
    results = {}
    for days in [1] + forecast_days:
        prob = state_vec @ np.linalg.matrix_power(tm, days)
        results[days] = {
            'stay_prob': prob[current_state],
            'exit_prob': 1 - prob[current_state],
            f'p_state_{current_state}': prob[current_state],
        }
    
    return pd.DataFrame(results).T
```
