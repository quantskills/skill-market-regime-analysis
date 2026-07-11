---
name: market-regime-analysis
description: "Analyze Chinese A-share market regimes by combining index data, macro indicators, futures term structure, and volatility clustering. Use when an agent needs to: (1) classify market states (bull/bear/sideways/high-vol/low-vol), (2) detect regime shifts using Hidden Markov Models or threshold rules, (3) evaluate factor performance conditional on regimes, (4) generate regime-aware risk forecasts, or (5) build regime-switching trading strategies using Pandadata APIs (get_index_daily, get_macro_*, get_future_term_structure, get_future_basis) and panda_backtest."
quantSkills:
  organization: https://github.com/quantskills
  repository: quantskills/skill-market-regime-analysis
  repository_url: https://github.com/quantskills/skill-market-regime-analysis
  project_type: skill
  collection: market-research
  license: GPL-3.0
  category: tooling
  tags: [market-regime, regime-detection, hidden-markov-model, volatility-clustering, macro-indicators, term-structure, a-share, factor-conditioning, risk-forecast, pandadata]
  platforms: [claude-code, codex, openclaw]
  language: zh-en
  status: draft
  validation_level: listed
  maintainer_type: community
  requires: []
  summary_zh: A股市场状态分析工具——结合指数数据、宏观指标、期货期限结构和波动率聚集特征，通过HMM或阈值规则划分市场状态（牛/熊/震荡/高波/低波），评估因子在各状态下的条件表现，生成状态感知的风险预测，构建状态切换策略
  summary_en: Chinese A-share market regime analysis toolkit — classify market states (bull/bear/sideways/high-vol/low-vol) using HMMs or threshold rules on index data, macro indicators, and futures term structure; evaluate conditional factor performance; generate regime-aware risk forecasts; and build regime-switching strategies.
---

# Market Regime Analysis

Analyze and classify Chinese A-share market regimes using multi-source data, then use regime information to improve factor timing, risk forecasting, and strategy design.

## Data Sources (panda_data)

This skill uses the following Pandadata APIs for regime detection:

| Data | API | Purpose |
|------|-----|---------|
| Index prices | `get_index_daily(symbol, start_date, end_date)` | Price trends, momentum, volatility (沪深300/中证500/创业板指 等) |
| Index valuations | `get_index_indicator(symbol, start_date, end_date, fields)` | PE/PB/股息率分位值 → 估值状态 |
| Macro (NA/CI/PI/FI/MB/IR/PM) | `get_macro_*(symbol, start_date, end_date)` | GDP/CPI/PPI/社融/M2/利率/PMI → 宏观状态 |
| Futures term structure | `get_future_term_structure(symbol, start_date, end_date)` | 基差/期限结构 → 市场情绪与预期 |
| Futures basis | `get_future_basis(underlying_symbol, start_date, end_date, fields)` | 期现价差 → 套保/投机情绪 |
| Stock daily | `get_stock_daily(start_date, end_date, symbol, indicator, st)` | 全市场量价数据 → 截面波动、相关性 |
| Trade calendar | `get_trade_cal(start_date, end_date)` | 交易日对齐 |

### Key Index Symbols

| Symbol | Index |
|--------|-------|
| `000300.SH` | 沪深300 (CSI 300) |
| `000905.SH` | 中证500 (CSI 500) |
| `000016.SH` | 上证50 (SSE 50) |
| `000688.SH` | 科创50 (STAR 50) |
| `399006.SZ` | 创业板指 (ChiNext) |
| `000001.SH` | 上证综合 (SSE Composite) |

### Key Macro Indicators

| API | Data |
|-----|------|
| `get_macro_na` | 国民经济核算 — GDP |
| `get_macro_ci` | 景气指数 — 中国经济景气 |
| `get_macro_pi` | 价格指数 — CPI/PPI |
| `get_macro_fi` | 财政 — 财政收入/支出 |
| `get_macro_mb` | 货币与银行 — M2/社融/信贷 |
| `get_macro_ir` | 利率 — 贷款/存款基准利率 |
| `get_macro_pm` | 采购经理人指数 — PMI |
| `get_macro_fe` | 对外经济 — 进出口 |
| `get_macro_na` | GDP — 季度GDP/累计GDP |
| `get_macro_dt` | 国内贸易 — 社会消费品零售 |
| `get_macro_fa` | 固定资产投资 |
| `get_macro_in` | 工业 — 工业增加值 |
| `get_macro_cal` | 宏观日历 — 经济数据发布时间 |

## Regime Detection Methods

### Method 1: Simple Threshold Rules (Quick & Interpretable)

Define regimes by cutting historical data at fixed thresholds. Best for explaining to non-technical stakeholders.

**Price trend regime:**
- Bull: 指数20日涨幅 > 5% AND 125日线向上
- Bear: 指数20日涨幅 < -5% AND 125日线向下
- Sideways: 其他

**Volatility regime:**
- High vol: 20日年化波动率 > 历史80%分位
- Low vol: 20日年化波动率 < 历史20%分位
- Normal vol: 中间

**Macro regime:**
- Expansion: PMI > 50 AND GDP增速加速 AND M2增速 > 10%
- Contraction: PMI < 50 AND GDP增速放缓
- Transition: 其他

### Method 2: Hidden Markov Model (HMM)

Use Gaussian HMM on multi-dimensional features to discover latent states. Supports 2-5 states.

**Standard feature set (4-dim):**
1. 指数对数收益率(5日)
2. 指数20日年化波动率
3. 期限结构斜率(近月-远月基差)
4. 全市场等权平均换手率变化

**Workflow:**
```
returns + vol + term_structure + turnover_change
                    │
              ┌─────┴─────┐
              │  StandardScaler │
              └─────┬─────┘
                    │
              ┌─────┴─────┐
              │  GaussianHMM   │  (fit 500 iter, 10 random init)
              │  n_components  │  (2-5 states, pick by BIC or AIC)
              └─────┬─────┘
                    │
        ┌───────────┼───────────┐
        ▼           ▼           ▼
  State 0      State 1      State 2 ...
   (Bull)      (Bear)      (Sideways)
```

### Method 3: Rolling Regime Classification

Update regime daily with a rolling window. The regime at date T is inferred from data up to T (no look-ahead).

## Factor Conditioning

After regimes are classified, test how factors perform differently across regimes:

```python
# Per-regime factor evaluation pattern
regime_stats = {}
for state in states:
    regime_stats[state] = {
        'avg_ic': factors[dates_in_state].corr(returns[dates_in_state]),
        'avg_ret': factors[dates_in_state].mean(),
        'hit_rate': (factors[dates_in_state] * returns[dates_in_state] > 0).mean(),
        'sharpe': ...,
        'n_days': len(dates_in_state)
    }
```

Key question: does a factor's IC vary significantly across regimes? If yes, regime-aware factor weighting can improve performance.

## Risk Forecasting per Regime

Each regime has distinct risk characteristics:

| Regime | Typical Vol | Correlation | Drawdown Risk |
|--------|------------|-------------|---------------|
| Bull | Low-Med | Low | Low (short-lived) |
| Bear | High | High (all stocks fall together) | High (prolonged) |
| Sideways | Low | Medium | Medium (range-bound) |
| High-Vol | Very High | Very High | Extreme |
| Low-Vol | Low | Low | Low |

For each regime, estimate conditional covariance matrix and VaR/CVaR. See `references/risk_forecast.md` for detailed method.

## Conditional Portfolio Construction

Use regime to adjust portfolio parameters:

| Parameter | Bull | Bear | Sideways | High Vol |
|-----------|------|------|----------|----------|
| Target volatility | 12% | 8% | 10% | 6% |
| Max drawdown limit | 15% | 10% | 12% | 8% |
| Factor tilt | Momentum+Growth | Value+LowVol | Quality+Reversal | Cash+Defensive |
| Leverage | 1.0x | 0.5x | 0.8x | 0.3x |
| Position sizing | Full | Half | Normal | Quarter |

## Workflow

### Phase 1: Data Collection

```python
import pandas as pd
import numpy as np

# Collect index data
index_data = get_index_daily(symbol='000300.SH', start_date='20200101', end_date='20250702')

# Collect macro indicators
gdp = get_macro_na(symbol='GDP_QUARTERLY', start_date='20200101', end_date='20250702')
cpi = get_macro_pi(symbol='CPI_YOY', start_date='20200101', end_date='20250702')
pmi = get_macro_pm(symbol='PMI', start_date='20200101', end_date='20250702')
m2 = get_macro_mb(symbol='M2_YOY', start_date='20200101', end_date='20250702')

# Collect term structure
term_structure = get_future_term_structure(symbol='IF', start_date='20200101', end_date='20250702')
```

### Phase 2: Feature Engineering

Compute regime features from raw data. See `references/feature_engineering.md`.

### Phase 3: Regime Classification

Choose a method. See `references/hmm_detection.md` and `references/threshold_rules.md`.

### Phase 4: Conditional Analysis

Evaluate factors and risk metrics per regime. See `references/conditional_analysis.md`.

### Phase 5: Regime-Aware Strategy

Build a strategy that adapts to detected regimes. See `examples/regime_switching_strategy.py`.

## References

| File | When to read |
|------|-------------|
| `references/threshold_rules.md` | When using simple threshold-based regime definitions (bull/bear/sideways, high/low vol) |
| `references/hmm_detection.md` | When applying Hidden Markov Models for latent state discovery |
| `references/feature_engineering.md` | When computing regime features from raw data (rolling stats, normalization, date alignment) |
| `references/conditional_analysis.md` | When evaluating factor performance or risk metrics per regime |
| `references/risk_forecast.md` | When generating regime-aware covariance, VaR, or drawdown forecasts |
| `examples/regime_switching_strategy.py` | When building a complete regime-dependent trading strategy with panda_backtest |
