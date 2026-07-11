# Market Regime Analysis

> Analyze Chinese A-share market regimes by combining index data, macro indicators, futures term structure, and volatility clustering — then use regime information to improve factor timing, risk forecasting, and strategy design.

[![Skill](https://img.shields.io/badge/AgentSkill-market--regime--analysis-blue)](#)
[![License](https://img.shields.io/badge/License-GPL--3.0-green)](#)
[![Language](https://img.shields.io/badge/lang-zh--en-yellow)](#)
[![Status](https://img.shields.io/badge/status-draft-lightgrey)](#)

---

## Overview

This skill provides an end-to-end workflow for Chinese A-share market regime analysis, covering **data collection → feature engineering → regime detection → conditional analysis → regime-aware strategy**.

### What It Does

| Capability | Description |
|------------|-------------|
| **Regime Classification** | Classify markets into bull/bear/sideways/high-vol/low-vol states |
| **Regime Detection** | Use Hidden Markov Models or threshold rules to detect shifts automatically |
| **Conditional Factor Analysis** | Evaluate factor IC, returns, and Sharpe ratios conditional on regimes |
| **Risk Forecasting** | Estimate conditional covariance, VaR, and CVaR per regime |
| **Regime-Switching Strategy** | Build strategies that adapt positioning and factor tilts to the current regime |

---

## Data Sources

Uses [Pandadata](https://github.com/pandadb/panda_data) APIs:

| Data | API | Purpose |
|------|-----|---------|
| Index daily | `get_index_daily` | Price trends, momentum, volatility |
| Index valuations | `get_index_indicator` | PE/PB/dividend yield percentiles |
| Macro indicators | `get_macro_*` (NA/CI/PI/FI/MB/IR/PM/FE/DT/FA/IN/CAL) | GDP/CPI/PPI/Social Financing/M2/rates/PMI |
| Futures term structure | `get_future_term_structure` | Basis and market sentiment |
| Futures basis | `get_future_basis` | Hedging/speculation sentiment |
| Stock daily | `get_stock_daily` | Cross-sectional volatility, correlations |
| Trade calendar | `get_trade_cal` | Date alignment |

---

## Regime Detection Methods

### Method 1: Threshold Rules (Simple & Interpretable)

```python
# Price trend: 20d return > 5% → Bull, < -5% → Bear, else → Sideways
# Volatility: 20d annualized vol above 80th percentile → High Vol
# Macro: PMI > 50 + GDP accelerating + M2 > 10% → Expansion
```

Great for quick prototypes or explaining to non-technical stakeholders.

### Method 2: Hidden Markov Model (HMM)

Gaussian HMM on a 4-dimensional feature vector for unsupervised state discovery:

```
log_return(5d) + annualized_vol(20d) + term_structure_slope + turnover_change
        │
   StandardScaler
        │
   GaussianHMM (n_components=2~5, select by BIC/AIC)
        │
   State 0 (Bull)   State 1 (Bear)   State 2 (Sideways) ...
```

### Method 3: Rolling Classification

Infer today's regime using only data up to today — no look-ahead bias.

---

## Quick Start

```python
# 1. Collect data
index_data = get_index_daily(symbol='000300.SH', start_date='20200101', end_date='20250702')
pmi = get_macro_pm(symbol='PMI', start_date='20200101', end_date='20250702')
term_structure = get_future_term_structure(symbol='IF', start_date='20200101', end_date='20250702')

# 2. Feature engineering → see references/feature_engineering.md
# 3. Regime detection → see references/hmm_detection.md or references/threshold_rules.md
# 4. Conditional analysis → see references/conditional_analysis.md
# 5. Build strategy → see examples/regime_switching_strategy.py
```

---

## File Structure

```
skill-market-regime-analysis/
├── SKILL.md                              # Main entry point (full API documentation)
├── README.md                             # English README (this file)
├── README.zh-CN.md                       # Chinese README
├── references/
│   ├── feature_engineering.md            # Feature engineering deep-dive
│   ├── hmm_detection.md                  # HMM regime detection
│   ├── threshold_rules.md                # Threshold-based rules
│   ├── conditional_analysis.md           # Conditional analysis
│   └── risk_forecast.md                  # Risk forecasting
├── examples/
│   └── regime_switching_strategy.py      # Complete strategy example
└── agents/
    └── openai.yaml                       # Agent configuration
```

---

## Supported Agent Platforms

- **Claude Code** (Anthropic)
- **Codex** (OpenAI)
- **OpenClaw**

---

## License

GPL-3.0 License. See [LICENSE](LICENSE).
