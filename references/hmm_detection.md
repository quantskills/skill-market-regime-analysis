# HMM Regime Detection

Use Hidden Markov Models to discover latent market states directly from data, without pre-defined thresholds.

## Why HMM?

- **Data-driven:** no need to manually define what "Bull" means
- **Probabilistic:** outputs P(state|data), not hard classification
- **Temporal:** captures transition probabilities between states
- **Multi-signal:** naturally handles multi-dimensional features

## Implementation

```python
from hmmlearn import hmm
from sklearn.preprocessing import StandardScaler

def hmm_regime_detection(features, n_states=4, n_iter=500, n_init=10):
    """
    features: pd.DataFrame with columns [ret_5d, vol_20d, term_spread, turnover_chg]
              index = trading dates
    n_states: number of latent regimes to discover
    Returns: (state_series, model, transition_matrix)
    """
    # 1. Standardize features
    scaler = StandardScaler()
    X = scaler.fit_transform(features.values)
    
    # 2. Fit HMM
    model = hmm.GaussianHMM(
        n_components=n_states,
        covariance_type='full',
        n_iter=n_iter,
        n_init=n_init,
        random_state=42,
        tol=0.01
    )
    model.fit(X)
    
    # 3. Predict states
    states = model.predict(X)
    state_series = pd.Series(states, index=features.index, name='regime')
    
    # 4. Decode state means (for labeling)
    state_means = pd.DataFrame(
        scaler.inverse_transform(model.means_),
        columns=features.columns,
        index=[f'State_{i}' for i in range(n_states)]
    )
    
    return state_series, model, state_means
```

## State Labeling

After HMM discovers states, label them by their characteristics:

```python
def label_states(state_means, vol_col='vol_20d', ret_col='ret_5d'):
    """
    Assign human-readable names based on each state's mean feature values.
    """
    labels = {}
    for state in state_means.index:
        row = state_means.loc[state]
        vol = row[vol_col]  # annualized vol
        ret = row[ret_col]  # 5-day return (decimal)
        
        if vol > 0.35 and ret < -0.02:
            labels[state] = 'Crisis'
        elif vol > 0.30:
            labels[state] = 'High Vol'
        elif vol < 0.12 and ret > 0.01:
            labels[state] = 'Low Vol Rally'
        elif vol < 0.12 and ret < -0.005:
            labels[state] = 'Low Vol Drift'
        elif ret > 0.02:
            labels[state] = 'Strong Bull'
        elif ret < -0.015:
            labels[state] = 'Bear'
        else:
            labels[state] = 'Sideways'
    return labels
```

## Feature Engineering for HMM

### Core 4-dimension feature set

| Feature | Formula | Rationale |
|---------|---------|-----------|
| `ret_5d` | ln(P_t / P_{t-5}) ≈ 1-week return | Captures short-term momentum |
| `ret_21d` | ln(P_t / P_{t-21}) ≈ 1-month return | Medium-term trend |
| `vol_21d` | std(log_returns, 21d) × √252 | Annualized vol regime |
| `max_dd_63d` | 1 - P_t / max(P_{t-63:t}) | 3-month drawdown |
| `turnover_20d_ma` | 20-day MA of turnover ratio | Liquidity regime |
| `basis_spread` | 近月合约价 - 远月合约价 | 期货基差 → 情绪 |

### A-Share Specific Features

| Feature | Data Source | Notes |
|---------|-------------|-------|
| `pe_percentile` | `get_index_indicator('000300.SH', fields=['pe'])` | 沪深300 PE 5年分位 |
| `northbound_flow_20d` | `get_hsgt_hold()` → daily change | 北向资金净流入趋势 |
| `margin_change_5d` | `get_margin()` → margin balance change | 融资余额变化 |
| `advance_decline_ratio` | `get_stock_daily()` → count(close>open) / total | 涨跌比 |
| `new_high_ratio` | `get_stock_daily()` → count(close>MA20) / total | 创新高比例 |

**Compute advance/decline/new-high ratio efficiently:**
```python
def market_breadth(stock_data_dict, date, ma_window=20):
    """stock_data_dict: {symbol: DataFrame with 'close', 'open'}"""
    closes = pd.DataFrame({s: df['close'] for s, df in stock_data_dict.items()})
    opens = pd.DataFrame({s: df['open'] for s, df in stock_data_dict.items()})
    
    ad_ratio = (closes > opens).mean(axis=1)  # advance/decline
    ma = closes.rolling(ma_window).mean()
    new_high = (closes > ma).mean(axis=1)      # above MA ratio
    
    return ad_ratio, new_high
```

## Model Selection (Number of States)

```python
def select_n_states(features, max_states=6):
    """Select optimal number of HMM states using BIC."""
    bic_scores = {}
    for n in range(2, max_states + 1):
        model = hmm.GaussianHMM(n_components=n, n_iter=500, random_state=42)
        model.fit(features.values)
        
        n_params = n * features.shape[1] * 2  # means + covars per state
        n_params += n * n  # transition matrix
        log_lik = model.score(features.values)
        bic = -2 * log_lik + n_params * np.log(features.shape[0])
        bic_scores[n] = bic
    
    best_n = min(bic_scores, key=bic_scores.get)
    return best_n, bic_scores
```

## Transition Matrix Interpretation

After fitting, analyze the transition matrix:

```python
def analyze_transitions(model, state_labels):
    """Print state transition characteristics."""
    tm = pd.DataFrame(
        model.transmat_,
        index=[state_labels[i] for i in range(model.n_components)],
        columns=[state_labels[i] for i in range(model.n_components)]
    )
    
    # Persistence (probability of staying in same state)
    persistence = pd.Series(
        np.diag(model.transmat_), 
        index=[state_labels[i] for i in range(model.n_components)]
    )
    
    # Expected duration of each state
    expected_duration = 1 / (1 - np.diag(model.transmat_))
    
    return tm, persistence, expected_duration
```

## A-Share Typical Regime Patterns (for reference)

Based on 2010-2024 history, typical 4-state HMM on 沪深300 produces:

| State | Frequency | Avg Duration | Typical Periods |
|-------|-----------|-------------|-----------------|
| Bull | 25-30% | 40-60 days | 2014H2, 2017, 2019Q1, 2020H1 |
| Bear | 15-20% | 20-40 days | 2015H2, 2018, 2022, 2024Q1 |
| Sideways | 35-40% | 60-120 days | 2013, 2016, 2019H2, 2023 |
| High Vol | 10-15% | 10-20 days | 2015 crisis, 2020 COVID, 2022H1 |

## Pitfalls

1. **Look-ahead bias**: Never fit HMM on full dataset then predict all dates. Use expanding window fitting.
2. **Feature scaling**: Always standardize features. HMM covariances are sensitive to scale differences.
3. **State permutation**: HMM state indices are arbitrary (label "State 0" might be Bull in one run and Bear in another). Always label by mean features.
4. **Regime persistence validation**: If average state duration < 5 days, states are likely noise.
5. **Feature freshness**: Macro indicators and financial data have publication lag.
