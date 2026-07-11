"""
Regime-Switching Trading Strategy for Panda Backtest
====================================================
Detects market regime using rolling features and switches between
momentum (bull) and defensive (bear) positioning.

Strategy logic:
- Bull regime (20d ret > 5%, price above MA60): Long IF主力合约, 2手
- Bear regime (20d ret < -3%, price below MA60): Long AU主力合约, 1手 (避险)
- Sideways/Other: 空仓

Data sources used:
- get_index_daily: 沪深300指数日线
- get_future_dominant: 主力合约映射
- get_future_list: 合约乘数
- get_stock_daily: A股行情

This example requires panda_backtest >= 0.1.0 and panda_data >= 0.1.0.
"""

from panda_backtest.api.api import *
import panda_data
import numpy as np

# =====================================================================
# Config
# =====================================================================
MODE = 'backtest'

def initialize(context):
    context.account = '5588'
    context.mode = MODE
    
    # Parameters
    context.products = ['IF', 'AU']  # 股指期货 + 黄金期货
    context.index_symbol = '000300.SH'
    context.ma_window = 60
    context.trend_window = 20
    context.bull_threshold = 0.05
    context.bear_threshold = -0.03
    
    # State
    context.today_dominant = {}
    context.contract_mul = {}
    context.regime_history = []
    
    # Preload
    if context.mode == 'backtest':
        _preload_all_data(context)

def _preload_all_data(context):
    """
    Preload dominant contract map and multiplier.
    Only used in backtest mode for performance.
    """
    context._dominant_map = {}
    context._mul_map = {}
    
    for product in context.products:
        try:
            dom_df = panda_data.get_future_dominant(
                underlying_symbol=[product],
                start_date='20200101',
                end_date='20250701'
            )
            if dom_df is not None and not dom_df.empty:
                for _, row in dom_df.iterrows():
                    date_key = str(row['nature_date']).replace('-', '')[:8]
                    context._dominant_map[(product, date_key)] = row['symbol']
        except Exception:
            pass
    
    for product in context.products:
        try:
            dom_df = panda_data.get_future_dominant(
                underlying_symbol=[product],
                start_date='20250601',
                end_date='20250701'
            )
            if dom_df is not None and not dom_df.empty:
                symbols = dom_df['symbol'].unique().tolist()
                mul_df = panda_data.get_future_list(
                    symbol=symbols,
                    fields=['symbol', 'contract_multiplier']
                )
                if mul_df is not None and not mul_df.empty:
                    for _, row in mul_df.iterrows():
                        context._mul_map[row['symbol']] = float(row['contract_multiplier'])
        except Exception:
            pass

def before_trading(context):
    """Update dominant contracts and check index regime."""
    today = str(context.now)
    context.today_dominant = {}
    context.contract_mul = {}
    
    if context.mode == 'backtest':
        for product in context.products:
            symbol = context._dominant_map.get((product, today))
            if symbol:
                context.today_dominant[product] = symbol
                context.contract_mul[symbol] = context._mul_map.get(symbol, 1.0)
    else:
        try:
            dom_df = panda_data.get_future_dominant(
                underlying_symbol=context.products,
                start_date=today,
                end_date=today
            )
            if dom_df is not None and not dom_df.empty:
                for _, row in dom_df.iterrows():
                    context.today_dominant[row['underlying_symbol']] = row['symbol']
        except Exception:
            pass
        
        symbols = list(context.today_dominant.values())
        if symbols:
            try:
                mul_df = panda_data.get_future_list(
                    symbol=symbols,
                    fields=['symbol', 'contract_multiplier']
                )
                if mul_df is not None and not mul_df.empty:
                    for _, row in mul_df.iterrows():
                        context.contract_mul[row['symbol']] = float(row['contract_multiplier'])
            except Exception:
                pass
    
    symbols = list(context.today_dominant.values())
    if symbols:
        sub_future_symbol(symbols)
    
    # ---- Detect regime using rolling index data ----
    _detect_regime(context)

def _detect_regime(context):
    """Classify current market regime."""
    try:
        # Get last 125 days of index data
        end = str(context.now)
        idx_df = panda_data.get_index_daily(
            symbol=context.index_symbol,
            start_date='20200101',
            end_date=end
        )
        if idx_df is None or idx_df.empty:
            context.current_regime = 'Unknown'
            return
        
        close = idx_df['close'].values
        if len(close) < context.ma_window:
            context.current_regime = 'Unknown'
            return
        
        # Calculate features
        ret_20d = close[-1] / close[-context.trend_window] - 1
        ma60 = np.mean(close[-context.ma_window:])
        price = close[-1]
        above_ma = price > ma60
        
        # Classify
        if ret_20d > context.bull_threshold and above_ma:
            context.current_regime = 'Bull'
        elif ret_20d < context.bear_threshold and not above_ma:
            context.current_regime = 'Bear'
        else:
            context.current_regime = 'Sideways'
        
        context.regime_history.append((context.now, context.current_regime))
        
    except Exception:
        context.current_regime = 'Unknown'

def handle_data(context, data):
    """Execute regime-dependent trading."""
    futures_account = context.future_account_dict.get(context.account)
    if not futures_account:
        return
    
    regime = getattr(context, 'current_regime', 'Unknown')
    
    for product, symbol in context.today_dominant.items():
        try:
            bar = data[symbol]
        except Exception:
            continue
        if not bar or bar.close <= 0:
            continue
        
        positions = futures_account.positions
        current_qty = 0
        if symbol in list(positions.keys()):
            current_qty = positions[symbol].buy_quantity
        
        mul = context.contract_mul.get(symbol, 1.0)
        target_qty = 0
        
        if regime == 'Bull':
            if product == 'IF':
                target_qty = 2  # Long 2 lots IF
            else:
                target_qty = 0
        elif regime == 'Bear':
            if product == 'AU':
                target_qty = 1  # Long 1 lot AU (safe haven)
            else:
                target_qty = 0
        else:
            target_qty = 0
        
        # Rebalance
        if target_qty > current_qty:
            buy_open(context.account, symbol, target_qty - current_qty, style=MarketOrderStyle)
        elif target_qty < current_qty:
            sell_close(context.account, symbol, current_qty - target_qty, style=MarketOrderStyle)

def after_trading(context):
    """Log daily summary."""
    futures_account = context.future_account_dict.get(context.account)
    if futures_account:
        print(f"[{context.now}] Regime={getattr(context, 'current_regime', '?')} "
              f"Equity={futures_account.total_value:.0f} "
              f"P&L={futures_account.holding_pnl:.0f}")
