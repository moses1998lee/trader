# Forex Trading Strategies Backtesting Framework

This project provides a comprehensive backtesting framework for forex trading strategies, designed to achieve >=3% monthly returns while minimizing drawdown. The framework includes several proven algorithmic trading strategies and tools for backtesting, optimization, and performance analysis.

## Strategies Implemented

1. **Moving Average Crossover with Trend Filter**
   - Uses multiple moving averages to identify trends and generate entry/exit signals
   - Fast MA crosses above/below Slow MA for entry signals
   - Trend MA filters out trades against the main trend
   - Suitable for trending markets

2. **Bollinger Band Mean Reversion**
   - Trades when prices move to extremes (bands) and then revert to the mean
   - Uses RSI to confirm overbought/oversold conditions
   - Effective in range-bound markets
   - Aims to capture price reversions to the mean

3. **RSI with ADX Filter**
   - Combines Relative Strength Index (RSI) for overbought/oversold conditions
   - Average Directional Index (ADX) filters for strong trends
   - Reduces false signals in choppy markets
   - Balances trend and reversal trading

4. **MACD with Support/Resistance**
   - Uses Moving Average Convergence Divergence (MACD) for momentum signals
   - Support/Resistance levels provide optimal entry points
   - Combines momentum and price level analysis
   - Effective in both trending and ranging markets

5. **Dual Timeframe Momentum Strategy**
   - Combines momentum indicators from two timeframes
   - Fast RSI for entry signals, Slow RSI for trend confirmation
   - EMA filter for additional trend confirmation
   - Reduces false positives by requiring multiple confirmations

## Files

- `forex_strategy_backtest.py`: The main backtesting framework with strategy implementations
- `run_backtest.py`: Script to run backtests on different currency pairs and strategies
- `README_trading_strategies.md`: This documentation file

## How to Use

### Running Backtests

1. **Test a Single Strategy on a Single Currency Pair**

```python
from run_backtest import run_single_strategy_backtest
from forex_strategy_backtest import ma_crossover_strategy

data_path = "data/raw/oanda/eur_usd/eur_usd_m30_01122024_01052025.csv"
run_single_strategy_backtest(
    data_path=data_path,
    strategy_func=ma_crossover_strategy,
    strategy_params={
        "fast_ma": 10,
        "slow_ma": 50,
        "trend_ma": 200,
        "stop_loss_pips": 20,
        "take_profit_pips": 40,
    },
    strategy_name="Moving Average Crossover"
)
```

2. **Test All Strategies on a Single Currency Pair**

```python
from run_backtest import run_all_strategies_on_pair

data_path = "data/raw/oanda/eur_usd/eur_usd_m30_01122024_01052025.csv"
run_all_strategies_on_pair(data_path)
```

3. **Optimize Strategy Parameters**

```python
from run_backtest import optimize_strategy_parameters
from forex_strategy_backtest import ma_crossover_strategy

data_path = "data/raw/oanda/eur_usd/eur_usd_m30_01122024_01052025.csv"
param_grid = {
    "fast_ma": [5, 10, 15],
    "slow_ma": [30, 50, 70],
    "trend_ma": [150, 200, 250],
    "stop_loss_pips": [15, 20, 25],
    "take_profit_pips": [30, 40, 50],
}
optimize_strategy_parameters(
    data_path=data_path,
    strategy_func=ma_crossover_strategy,
    param_grid=param_grid,
    strategy_name="Moving Average Crossover",
    metric="sharpe_ratio"
)
```

4. **Run All Strategies on All Available Currency Pairs**

```python
python run_backtest.py
```

### Customizing Strategies

You can customize the strategy parameters to optimize performance for different market conditions:

1. **Moving Average Crossover**
   - `fast_ma`: Period for the fast moving average (default: 10)
   - `slow_ma`: Period for the slow moving average (default: 50)
   - `trend_ma`: Period for the trend filter moving average (default: 200)
   - `stop_loss_pips`: Stop loss distance in pips (default: 20)
   - `take_profit_pips`: Take profit distance in pips (default: 40)

2. **Bollinger Band Mean Reversion**
   - `ma_period`: Period for the moving average (default: 20)
   - `std_dev`: Standard deviation multiplier for bands (default: 2.0)
   - `rsi_period`: RSI period (default: 14)
   - `rsi_overbought`: RSI overbought threshold (default: 70)
   - `rsi_oversold`: RSI oversold threshold (default: 30)
   - `stop_loss_pips`: Stop loss distance in pips (default: 15)
   - `take_profit_pips`: Take profit distance in pips (default: 30)

3. **RSI with ADX Filter**
   - `rsi_period`: RSI period (default: 14)
   - `rsi_overbought`: RSI overbought threshold (default: 70)
   - `rsi_oversold`: RSI oversold threshold (default: 30)
   - `adx_period`: ADX period (default: 14)
   - `adx_threshold`: ADX threshold for trend strength (default: 25)
   - `stop_loss_pips`: Stop loss distance in pips (default: 25)
   - `take_profit_pips`: Take profit distance in pips (default: 50)

4. **MACD with Support/Resistance**
   - `fast_period`: Fast EMA period for MACD (default: 12)
   - `slow_period`: Slow EMA period for MACD (default: 26)
   - `signal_period`: Signal line period for MACD (default: 9)
   - `sr_period`: Period for support/resistance calculation (default: 20)
   - `stop_loss_pips`: Stop loss distance in pips (default: 30)
   - `take_profit_pips`: Take profit distance in pips (default: 60)

5. **Dual Timeframe Momentum**
   - `fast_rsi_period`: Fast RSI period (default: 7)
   - `slow_rsi_period`: Slow RSI period (default: 21)
   - `rsi_overbought`: RSI overbought threshold (default: 70)
   - `rsi_oversold`: RSI oversold threshold (default: 30)
   - `ema_period`: EMA period for trend filter (default: 50)
   - `stop_loss_pips`: Stop loss distance in pips (default: 25)
   - `take_profit_pips`: Take profit distance in pips (default: 50)

## Performance Metrics

The backtesting framework calculates and reports the following performance metrics:

- **Total Return**: Overall return on investment
- **Annual Return**: Annualized return
- **Monthly Return**: Monthly return (target: >=3%)
- **Max Drawdown**: Maximum peak-to-trough decline (lower is better)
- **Sharpe Ratio**: Risk-adjusted return (higher is better)
- **Win Rate**: Percentage of winning trades
- **Profit Factor**: Ratio of gross profits to gross losses
- **Total Trades**: Number of trades executed

## Risk Management

The framework implements proper risk management:

- **Position Sizing**: Based on a percentage of capital at risk (default: 2%)
- **Stop Loss**: Automatic stop loss for every trade
- **Take Profit**: Automatic take profit for every trade
- **Transaction Costs**: Accounts for spread, slippage, and commission

## Tips for Achieving >=3% Monthly Returns

1. **Strategy Selection**:
   - Different strategies perform better in different market conditions
   - Trending strategies (MA Crossover, MACD) work best in trending markets
   - Mean reversion strategies (Bollinger Bands) work best in range-bound markets
   - Use the strategy comparison tool to identify the best strategy for each pair

2. **Parameter Optimization**:
   - Use the parameter optimization tool to find optimal parameters
   - Optimize for Sharpe ratio to balance returns and risk
   - Consider different optimization metrics (monthly return, drawdown, etc.)

3. **Risk Management**:
   - Adjust risk per trade (default: 2%) based on strategy performance
   - Consider reducing risk for strategies with lower win rates
   - Adjust stop loss and take profit levels based on market volatility

4. **Currency Pair Selection**:
   - Some strategies perform better on specific currency pairs
   - Compare performance across different pairs
   - Focus on pairs with higher liquidity and lower spreads

5. **Timeframe Considerations**:
   - The default timeframe is 30 minutes (M30)
   - Consider testing on different timeframes (H1, H4, etc.)
   - Higher timeframes often have less noise but fewer trading opportunities

## Extending the Framework

You can extend the framework by:

1. **Adding New Strategies**:
   - Create a new strategy function in `forex_strategy_backtest.py`
   - Follow the pattern of existing strategies
   - Add the strategy to the list in `run_backtest.py`

2. **Adding New Performance Metrics**:
   - Modify the `backtest_strategy` method in the `ForexBacktester` class
   - Add new metrics to the results dictionary

3. **Implementing Portfolio Backtesting**:
   - Modify the framework to trade multiple currency pairs simultaneously
   - Implement portfolio-level risk management

## Conclusion

This backtesting framework provides a comprehensive set of tools for developing, testing, and optimizing forex trading strategies. By carefully selecting and optimizing strategies for different market conditions and currency pairs, it's possible to achieve the target of >=3% monthly returns while minimizing drawdown.

Remember that past performance is not indicative of future results, and all trading strategies should be thoroughly tested before being used in live trading.
