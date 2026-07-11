# Market Regime Analysis (A股市场状态分析)

> 结合指数数据、宏观指标、期货期限结构和波动率聚集特征，对 A 股市场进行状态划分与状态感知的策略构建。

[![Skill](https://img.shields.io/badge/AgentSkill-market--regime--analysis-blue)](#)
[![License](https://img.shields.io/badge/License-GPL--3.0-green)](#)
[![Language](https://img.shields.io/badge/lang-zh--en-yellow)](#)
[![Status](https://img.shields.io/badge/status-draft-lightgrey)](#)

---

## 概述

本 Skill 提供了一套完整的 A 股市场状态分析工作流，覆盖**数据采集 → 特征工程 → 状态检测 → 条件分析 → 状态感知策略**全链路。

### 能做什么

| 功能 | 说明 |
|------|------|
| **状态分类** | 将市场划分为牛市/熊市/震荡/高波/低波等状态 |
| **状态检测** | 通过 HMM（隐马尔可夫模型）或阈值规则自动识别状态切换 |
| **因子条件分析** | 评估同一因子在不同市场状态下的表现差异（IC、收益、夏普比） |
| **风险预测** | 基于当前状态生成条件协方差矩阵、VaR/CVaR |
| **状态切换策略** | 构建随市场状态自动调整仓位、因子偏好的策略 |

---

## 数据来源

基于 [Pandadata](https://github.com/pandadb/panda_data) API，使用以下数据：

| 数据 | API | 用途 |
|------|-----|------|
| 指数日线 | `get_index_daily` | 价格趋势、动量、波动率 |
| 指数估值 | `get_index_indicator` | PE/PB/股息率分位值 |
| 宏观指标 | `get_macro_na/ci/pi/fi/mb/ir/pm/fe/dt/fa/in/cal` | GDP/CPI/PPI/社融/M2/利率/PMI |
| 期货期限结构 | `get_future_term_structure` | 基差与市场情绪 |
| 期现价差 | `get_future_basis` | 套保/投机情绪 |
| 全市场行情 | `get_stock_daily` | 截面波动、相关性 |
| 交易日历 | `get_trade_cal` | 日期对齐 |

---

## 状态检测方法

### 方法一：阈值规则（简单、可解释）

```python
# 价格趋势：20日涨幅 > 5% → 牛，< -5% → 熊，其他 → 震荡
# 波动率：20日年化波动率高于历史80%分位 → 高波
# 宏观：PMI > 50 + GDP加速 + M2 > 10% → 扩张
```

适合快速原型或向非技术背景的读者解释。

### 方法二：隐马尔可夫模型（HMM）

使用 Gaussian HMM 对 4 维特征序列进行无监督状态发现：

```
对数收益率(5日) + 年化波动率(20日) + 期限结构斜率 + 换手率变化
        │
   StandardScaler
        │
   GaussianHMM (n_components=2~5, BIC/AIC 选最优)
        │
   状态0(牛)   状态1(熊)   状态2(震荡) ...
```

### 方法三：滚动分类

每日用截至当天的数据推断当日状态，无前视偏差。

---

## 快速开始

```python
# 1. 收集数据
index_data = get_index_daily(symbol='000300.SH', start_date='20200101', end_date='20250702')
pmi = get_macro_pm(symbol='PMI', start_date='20200101', end_date='20250702')
term_structure = get_future_term_structure(symbol='IF', start_date='20200101', end_date='20250702')

# 2. 特征工程 → 参考 references/feature_engineering.md
# 3. 状态检测 → 参考 references/hmm_detection.md 或 references/threshold_rules.md
# 4. 条件分析 → 参考 references/conditional_analysis.md
# 5. 策略构建 → 参考 examples/regime_switching_strategy.py
```

---

## 文档结构

```
skill-market-regime-analysis/
├── SKILL.md                              # 主入口（完整 API 文档）
├── README.zh-CN.md                       # 中文说明（本文件）
├── README.md                             # English README
├── references/
│   ├── feature_engineering.md            # 特征工程详解
│   ├── hmm_detection.md                  # HMM 状态检测
│   ├── threshold_rules.md                # 阈值规则
│   ├── conditional_analysis.md           # 条件分析
│   └── risk_forecast.md                  # 风险预测
├── examples/
│   └── regime_switching_strategy.py      # 完整策略示例
└── agents/
    └── openai.yaml                       # Agent 配置
```

---
## 如何一句话触发这个skill
直接说：

"做一下A股市场状态分析"

或者更具体的：

"分析当前A股的市场状态（牛/熊/震荡）和波动率状态"

只要任务涉及 市场状态划分、HMM识别状态切换、因子条件分析、状态感知的风险预测、构建状态切换策略 这些关键词，技能就会自动触发。

## 适用的 Agent 平台

- **Claude Code** (Anthropic)
- **Codex** (OpenAI)
- **OpenClaw**

---

## 许可

GPL-3.0 License. 详见 [LICENSE](LICENSE)。
