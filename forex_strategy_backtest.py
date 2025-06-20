import glob
import os
from typing import Callable, Dict, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Constants
RISK_PER_TRADE = 0.02  # 2% risk per trade
SPREAD_COST = 0.0001  # 1 pip spread cost (adjust based on actual spread)
COMMISSION = 0.0  # Commission per trade (if any)
SLIPPAGE = 0.0001  # 1 pip slippage (adjust based on expected slippage)


class ForexBacktester:
    def __init__(
        self,
        data_path: str,
        initial_capital: float = 10000.0,
        risk_per_trade: float = RISK_PER_TRADE,
        spread_cost: float = SPREAD_COST,
        commission: float = COMMISSION,
        slippage: float = SLIPPAGE,
    ):
        """
        Initialize the backtester with data and parameters.

        Args:
            data_path: Path to the CSV file containing OHLCV data
            initial_capital: Initial capital for backtesting
            risk_per_trade: Percentage of capital to risk per trade
            spread_cost: Cost of spread in decimal (e.g., 0.0001 for 1 pip in 4-decimal pairs)
            commission: Commission per trade in decimal
            slippage: Expected slippage in decimal
        """
        self.data_path = data_path
        self.initial_capital = initial_capital
        self.risk_per_trade = risk_per_trade
        self.spread_cost = spread_cost
        self.commission = commission
        self.slippage = slippage

        # Load and prepare data
        self.data = self._load_data()

        # Initialize results storage
        self.results = None
        self.trades = []
        self.equity_curve = []

    def _load_data(self) -> pd.DataFrame:
        """Load and prepare the data for backtesting."""
        df = pd.read_csv(self.data_path)

        # Convert time column to datetime
        df["time"] = pd.to_datetime(df["time"])

        # Set time as index
        df.set_index("time", inplace=True)

        # Calculate mid price
        df["mid"] = (df["bid"] + df["ask"]) / 2

        # Calculate returns
        df["returns"] = df["mid"].pct_change()

        return df

    def _calculate_position_size(
        self, capital: float, risk_amount: float, stop_loss_pips: float
    ) -> float:
        """
        Calculate the position size based on risk parameters.

        Args:
            capital: Current capital
            risk_amount: Amount to risk (percentage of capital)
            stop_loss_pips: Stop loss distance in pips

        Returns:
            Position size in standard lots (100,000 units)
        """
        risk_capital = capital * risk_amount
        pip_value = 10  # For a standard lot, 1 pip = $10 for most USD pairs

        # Calculate position size in standard lots
        position_size = risk_capital / (stop_loss_pips * pip_value)

        return position_size

    def _calculate_trade_result(
        self,
        entry_price: float,
        exit_price: float,
        position_size: float,
        direction: str,
        entry_time: pd.Timestamp,
        exit_time: pd.Timestamp,
    ) -> Dict:
        """
        Calculate the result of a trade.

        Args:
            entry_price: Entry price
            exit_price: Exit price
            position_size: Position size in standard lots
            direction: 'long' or 'short'
            entry_time: Entry timestamp
            exit_time: Exit timestamp

        Returns:
            Dictionary with trade details
        """
        pip_value = 10  # For a standard lot, 1 pip = $10 for most USD pairs

        if direction == "long":
            price_diff = exit_price - entry_price
            transaction_costs = (
                (self.spread_cost + self.slippage)
                * 2
                * position_size
                * pip_value
                * 10000
            )
            pnl = (
                price_diff * position_size * pip_value * 10000
                - transaction_costs
                - self.commission
            )
        else:  # short
            price_diff = entry_price - exit_price
            transaction_costs = (
                (self.spread_cost + self.slippage)
                * 2
                * position_size
                * pip_value
                * 10000
            )
            pnl = (
                price_diff * position_size * pip_value * 10000
                - transaction_costs
                - self.commission
            )

        return {
            "entry_time": entry_time,
            "exit_time": exit_time,
            "entry_price": entry_price,
            "exit_price": exit_price,
            "direction": direction,
            "position_size": position_size,
            "pnl": pnl,
            "pnl_pct": pnl / self.initial_capital * 100,
        }

    def backtest_strategy(self, strategy_func: Callable, strategy_params: Dict) -> Dict:
        """
        Backtest a trading strategy.

        Args:
            strategy_func: Function that implements the strategy
            strategy_params: Parameters for the strategy

        Returns:
            Dictionary with backtest results
        """
        # Apply strategy to get signals
        signals = strategy_func(self.data.copy(), **strategy_params)

        # Initialize variables for tracking performance
        capital = self.initial_capital
        max_capital = capital
        max_drawdown = 0
        in_position = False
        position_direction = None
        entry_price = 0
        entry_time = None
        position_size = 0

        # Track equity curve
        equity_curve = [{"time": self.data.index[0], "equity": capital}]

        # Iterate through data points
        for i in range(1, len(signals)):
            current_time = signals.index[i]

            # If not in a position, check for entry signals
            if not in_position:
                if signals["signal"][i] == 1:  # Long signal
                    entry_price = signals["ask"][i] + self.slippage
                    stop_loss_pips = strategy_params.get("stop_loss_pips", 20)
                    position_size = self._calculate_position_size(
                        capital, self.risk_per_trade, stop_loss_pips
                    )
                    in_position = True
                    position_direction = "long"
                    entry_time = current_time
                elif signals["signal"][i] == -1:  # Short signal
                    entry_price = signals["bid"][i] - self.slippage
                    stop_loss_pips = strategy_params.get("stop_loss_pips", 20)
                    position_size = self._calculate_position_size(
                        capital, self.risk_per_trade, stop_loss_pips
                    )
                    in_position = True
                    position_direction = "short"
                    entry_time = current_time

            # If in a position, check for exit signals
            else:
                exit_signal = False
                exit_price = 0

                if position_direction == "long":
                    # Check for stop loss
                    if (
                        signals["bid"][i]
                        <= entry_price
                        - strategy_params.get("stop_loss_pips", 20) * 0.0001
                    ):
                        exit_signal = True
                        exit_price = signals["bid"][i] - self.slippage
                    # Check for take profit
                    elif (
                        signals["bid"][i]
                        >= entry_price
                        + strategy_params.get("take_profit_pips", 40) * 0.0001
                    ):
                        exit_signal = True
                        exit_price = signals["bid"][i] - self.slippage
                    # Check for exit signal
                    elif signals["signal"][i] == -1:
                        exit_signal = True
                        exit_price = signals["bid"][i] - self.slippage

                elif position_direction == "short":
                    # Check for stop loss
                    if (
                        signals["ask"][i]
                        >= entry_price
                        + strategy_params.get("stop_loss_pips", 20) * 0.0001
                    ):
                        exit_signal = True
                        exit_price = signals["ask"][i] + self.slippage
                    # Check for take profit
                    elif (
                        signals["ask"][i]
                        <= entry_price
                        - strategy_params.get("take_profit_pips", 40) * 0.0001
                    ):
                        exit_signal = True
                        exit_price = signals["ask"][i] + self.slippage
                    # Check for exit signal
                    elif signals["signal"][i] == 1:
                        exit_signal = True
                        exit_price = signals["ask"][i] + self.slippage

                # Process exit
                if exit_signal:
                    trade_result = self._calculate_trade_result(
                        entry_price,
                        exit_price,
                        position_size,
                        position_direction,
                        entry_time,
                        current_time,
                    )
                    self.trades.append(trade_result)

                    # Update capital
                    capital += trade_result["pnl"]

                    # Update max capital and drawdown
                    if capital > max_capital:
                        max_capital = capital
                    else:
                        drawdown = (max_capital - capital) / max_capital
                        if drawdown > max_drawdown:
                            max_drawdown = drawdown

                    # Reset position flags
                    in_position = False
                    position_direction = None

            # Update equity curve
            equity_curve.append({"time": current_time, "equity": capital})

        # Convert equity curve to DataFrame
        equity_df = pd.DataFrame(equity_curve)
        equity_df.set_index("time", inplace=True)

        # Calculate performance metrics
        total_trades = len(self.trades)
        winning_trades = sum(1 for trade in self.trades if trade["pnl"] > 0)
        losing_trades = total_trades - winning_trades

        if total_trades > 0:
            win_rate = winning_trades / total_trades
            avg_win = (
                sum(trade["pnl"] for trade in self.trades if trade["pnl"] > 0)
                / winning_trades
                if winning_trades > 0
                else 0
            )
            avg_loss = (
                sum(trade["pnl"] for trade in self.trades if trade["pnl"] <= 0)
                / losing_trades
                if losing_trades > 0
                else 0
            )
            profit_factor = (
                abs(
                    sum(trade["pnl"] for trade in self.trades if trade["pnl"] > 0)
                    / sum(trade["pnl"] for trade in self.trades if trade["pnl"] <= 0)
                )
                if sum(trade["pnl"] for trade in self.trades if trade["pnl"] <= 0) != 0
                else float("inf")
            )
        else:
            win_rate = 0
            avg_win = 0
            avg_loss = 0
            profit_factor = 0

        # Calculate returns
        total_return = (capital - self.initial_capital) / self.initial_capital

        # Calculate annualized return
        days = (self.data.index[-1] - self.data.index[0]).days
        annual_return = (1 + total_return) ** (365 / days) - 1 if days > 0 else 0

        # Calculate monthly return (approximate)
        monthly_return = (1 + total_return) ** (30 / days) - 1 if days > 0 else 0

        # Calculate Sharpe ratio (assuming risk-free rate of 0)
        if len(equity_df) > 1:
            equity_returns = equity_df["equity"].pct_change().dropna()
            sharpe_ratio = (
                np.sqrt(252) * equity_returns.mean() / equity_returns.std()
                if equity_returns.std() != 0
                else 0
            )
        else:
            sharpe_ratio = 0

        # Store results
        self.results = {
            "initial_capital": self.initial_capital,
            "final_capital": capital,
            "total_return": total_return,
            "annual_return": annual_return,
            "monthly_return": monthly_return,
            "max_drawdown": max_drawdown,
            "sharpe_ratio": sharpe_ratio,
            "total_trades": total_trades,
            "winning_trades": winning_trades,
            "losing_trades": losing_trades,
            "win_rate": win_rate,
            "avg_win": avg_win,
            "avg_loss": avg_loss,
            "profit_factor": profit_factor,
            "equity_curve": equity_df,
        }

        return self.results

    def plot_results(self, save_path: Optional[str] = None):
        """
        Plot the backtest results.

        Args:
            save_path: Path to save the plot (optional)
        """
        if self.results is None:
            print("No backtest results to plot. Run backtest_strategy first.")
            return

        fig, (ax1, ax2) = plt.subplots(
            2, 1, figsize=(12, 10), gridspec_kw={"height_ratios": [3, 1]}
        )

        # Plot equity curve
        equity_curve = self.results["equity_curve"]
        ax1.plot(equity_curve.index, equity_curve["equity"])
        ax1.set_title("Equity Curve")
        ax1.set_ylabel("Capital")
        ax1.grid(True)

        # Calculate drawdown
        equity = equity_curve["equity"].values
        peak = np.maximum.accumulate(equity)
        drawdown = (peak - equity) / peak

        # Plot drawdown
        ax2.fill_between(equity_curve.index, 0, drawdown, alpha=0.3, color="red")
        ax2.set_title("Drawdown")
        ax2.set_ylabel("Drawdown")
        ax2.set_xlabel("Date")
        ax2.grid(True)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path)

        plt.show()

    def print_results(self):
        """Print the backtest results."""
        if self.results is None:
            print("No backtest results to print. Run backtest_strategy first.")
            return

        print("\n===== BACKTEST RESULTS =====")
        print(f"Initial Capital: ${self.results['initial_capital']:.2f}")
        print(f"Final Capital: ${self.results['final_capital']:.2f}")
        print(f"Total Return: {self.results['total_return']:.2%}")
        print(f"Annual Return: {self.results['annual_return']:.2%}")
        print(f"Monthly Return: {self.results['monthly_return']:.2%}")
        print(f"Max Drawdown: {self.results['max_drawdown']:.2%}")
        print(f"Sharpe Ratio: {self.results['sharpe_ratio']:.2f}")
        print(f"Total Trades: {self.results['total_trades']}")
        print(
            f"Winning Trades: {self.results['winning_trades']} ({self.results['win_rate']:.2%})"
        )
        print(
            f"Losing Trades: {self.results['losing_trades']} ({1 - self.results['win_rate']:.2%})"
        )
        print(f"Average Win: ${self.results['avg_win']:.2f}")
        print(f"Average Loss: ${self.results['avg_loss']:.2f}")
        print(f"Profit Factor: {self.results['profit_factor']:.2f}")
        print("============================\n")


# Strategy 1: Moving Average Crossover with Trend Filter
def ma_crossover_strategy(
    data: pd.DataFrame,
    fast_ma: int = 10,
    slow_ma: int = 50,
    trend_ma: int = 200,
    stop_loss_pips: int = 20,
    take_profit_pips: int = 40,
) -> pd.DataFrame:
    """
    Moving Average Crossover strategy with trend filter.

    Args:
        data: DataFrame with price data
        fast_ma: Fast moving average period
        slow_ma: Slow moving average period
        trend_ma: Trend moving average period
        stop_loss_pips: Stop loss in pips
        take_profit_pips: Take profit in pips

    Returns:
        DataFrame with signals
    """
    df = data.copy()

    # Calculate moving averages
    df["fast_ma"] = df["mid"].rolling(window=fast_ma).mean()
    df["slow_ma"] = df["mid"].rolling(window=slow_ma).mean()
    df["trend_ma"] = df["mid"].rolling(window=trend_ma).mean()

    # Initialize signal column
    df["signal"] = 0

    # Generate signals
    for i in range(1, len(df)):
        # Long signal: Fast MA crosses above Slow MA and price is above Trend MA
        if (
            df["fast_ma"][i - 1] <= df["slow_ma"][i - 1]
            and df["fast_ma"][i] > df["slow_ma"][i]
            and df["mid"][i] > df["trend_ma"][i]
        ):
            df.loc[df.index[i], "signal"] = 1

        # Short signal: Fast MA crosses below Slow MA and price is below Trend MA
        elif (
            df["fast_ma"][i - 1] >= df["slow_ma"][i - 1]
            and df["fast_ma"][i] < df["slow_ma"][i]
            and df["mid"][i] < df["trend_ma"][i]
        ):
            df.loc[df.index[i], "signal"] = -1

    return df


# Strategy 2: Bollinger Band Mean Reversion
def bollinger_band_strategy(
    data: pd.DataFrame,
    ma_period: int = 20,
    std_dev: float = 2.0,
    rsi_period: int = 14,
    rsi_overbought: int = 70,
    rsi_oversold: int = 30,
    stop_loss_pips: int = 15,
    take_profit_pips: int = 30,
) -> pd.DataFrame:
    """
    Bollinger Band Mean Reversion strategy.

    Args:
        data: DataFrame with price data
        ma_period: Moving average period for Bollinger Bands
        std_dev: Standard deviation multiplier for bands
        rsi_period: RSI period
        rsi_overbought: RSI overbought threshold
        rsi_oversold: RSI oversold threshold
        stop_loss_pips: Stop loss in pips
        take_profit_pips: Take profit in pips

    Returns:
        DataFrame with signals
    """
    df = data.copy()

    # Calculate Bollinger Bands
    df["ma"] = df["mid"].rolling(window=ma_period).mean()
    df["std"] = df["mid"].rolling(window=ma_period).std()
    df["upper_band"] = df["ma"] + std_dev * df["std"]
    df["lower_band"] = df["ma"] - std_dev * df["std"]

    # Calculate RSI
    delta = df["mid"].diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)

    avg_gain = gain.rolling(window=rsi_period).mean()
    avg_loss = loss.rolling(window=rsi_period).mean()

    rs = avg_gain / avg_loss
    df["rsi"] = 100 - (100 / (1 + rs))

    # Initialize signal column
    df["signal"] = 0

    # Generate signals
    for i in range(1, len(df)):
        # Long signal: Price below lower band and RSI oversold
        if df["mid"][i] < df["lower_band"][i] and df["rsi"][i] < rsi_oversold:
            df.loc[df.index[i], "signal"] = 1

        # Short signal: Price above upper band and RSI overbought
        elif df["mid"][i] > df["upper_band"][i] and df["rsi"][i] > rsi_overbought:
            df.loc[df.index[i], "signal"] = -1

    return df


# Strategy 3: RSI with ADX Filter
def rsi_adx_strategy(
    data: pd.DataFrame,
    rsi_period: int = 14,
    rsi_overbought: int = 70,
    rsi_oversold: int = 30,
    adx_period: int = 14,
    adx_threshold: int = 25,
    stop_loss_pips: int = 25,
    take_profit_pips: int = 50,
) -> pd.DataFrame:
    """
    RSI with ADX Filter strategy.

    Args:
        data: DataFrame with price data
        rsi_period: RSI period
        rsi_overbought: RSI overbought threshold
        rsi_oversold: RSI oversold threshold
        adx_period: ADX period
        adx_threshold: ADX threshold for trend strength
        stop_loss_pips: Stop loss in pips
        take_profit_pips: Take profit in pips

    Returns:
        DataFrame with signals
    """
    df = data.copy()

    # Calculate RSI
    delta = df["mid"].diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)

    avg_gain = gain.rolling(window=rsi_period).mean()
    avg_loss = loss.rolling(window=rsi_period).mean()

    rs = avg_gain / avg_loss
    df["rsi"] = 100 - (100 / (1 + rs))

    # Calculate ADX
    high_low = df["ask"] - df["bid"]
    high_close = np.abs(df["ask"] - df["mid"].shift())
    low_close = np.abs(df["bid"] - df["mid"].shift())

    ranges = pd.concat([high_low, high_close, low_close], axis=1)
    true_range = ranges.max(axis=1)

    df["tr"] = true_range
    df["atr"] = df["tr"].rolling(window=adx_period).mean()

    df["up_move"] = df["ask"].diff()
    df["down_move"] = -df["bid"].diff()

    df["plus_dm"] = np.where(
        (df["up_move"] > df["down_move"]) & (df["up_move"] > 0), df["up_move"], 0
    )
    df["minus_dm"] = np.where(
        (df["down_move"] > df["up_move"]) & (df["down_move"] > 0), df["down_move"], 0
    )

    df["plus_di"] = 100 * (df["plus_dm"].rolling(window=adx_period).mean() / df["atr"])
    df["minus_di"] = 100 * (
        df["minus_dm"].rolling(window=adx_period).mean() / df["atr"]
    )

    df["dx"] = 100 * np.abs(
        (df["plus_di"] - df["minus_di"]) / (df["plus_di"] + df["minus_di"])
    )
    df["adx"] = df["dx"].rolling(window=adx_period).mean()

    # Initialize signal column
    df["signal"] = 0

    # Generate signals
    for i in range(1, len(df)):
        # Long signal: RSI crosses above oversold and ADX above threshold
        if (
            df["rsi"][i - 1] <= rsi_oversold
            and df["rsi"][i] > rsi_oversold
            and df["adx"][i] > adx_threshold
        ):
            df.loc[df.index[i], "signal"] = 1

        # Short signal: RSI crosses below overbought and ADX above threshold
        elif (
            df["rsi"][i - 1] >= rsi_overbought
            and df["rsi"][i] < rsi_overbought
            and df["adx"][i] > adx_threshold
        ):
            df.loc[df.index[i], "signal"] = -1

    return df


# Strategy 4: MACD with Support/Resistance
def macd_sr_strategy(
    data: pd.DataFrame,
    fast_period: int = 12,
    slow_period: int = 26,
    signal_period: int = 9,
    sr_period: int = 20,
    stop_loss_pips: int = 30,
    take_profit_pips: int = 60,
) -> pd.DataFrame:
    """
    MACD with Support/Resistance strategy.

    Args:
        data: DataFrame with price data
        fast_period: Fast EMA period for MACD
        slow_period: Slow EMA period for MACD
        signal_period: Signal line period for MACD
        sr_period: Period for support/resistance calculation
        stop_loss_pips: Stop loss in pips
        take_profit_pips: Take profit in pips

    Returns:
        DataFrame with signals
    """
    df = data.copy()

    # Calculate MACD
    df["ema_fast"] = df["mid"].ewm(span=fast_period, adjust=False).mean()
    df["ema_slow"] = df["mid"].ewm(span=slow_period, adjust=False).mean()
    df["macd"] = df["ema_fast"] - df["ema_slow"]
    df["macd_signal"] = df["macd"].ewm(span=signal_period, adjust=False).mean()
    df["macd_hist"] = df["macd"] - df["macd_signal"]

    # Calculate support and resistance
    df["support"] = df["bid"].rolling(window=sr_period).min()
    df["resistance"] = df["ask"].rolling(window=sr_period).max()

    # Initialize signal column
    df["signal"] = 0

    # Generate signals
    for i in range(1, len(df)):
        # Long signal: MACD crosses above signal line and price near support
        if (
            df["macd"][i - 1] <= df["macd_signal"][i - 1]
            and df["macd"][i] > df["macd_signal"][i]
            and df["mid"][i] < df["support"][i] * 1.001
        ):
            df.loc[df.index[i], "signal"] = 1

        # Short signal: MACD crosses below signal line and price near resistance
        elif (
            df["macd"][i - 1] >= df["macd_signal"][i - 1]
            and df["macd"][i] < df["macd_signal"][i]
            and df["mid"][i] > df["resistance"][i] * 0.999
        ):
            df.loc[df.index[i], "signal"] = -1

    return df


# Strategy 5: Dual Timeframe Momentum Strategy
def dual_timeframe_momentum_strategy(
    data: pd.DataFrame,
    fast_rsi_period: int = 7,
    slow_rsi_period: int = 21,
    rsi_overbought: int = 70,
    rsi_oversold: int = 30,
    ema_period: int = 50,
    stop_loss_pips: int = 25,
    take_profit_pips: int = 50,
) -> pd.DataFrame:
    """
    Dual Timeframe Momentum strategy.

    Args:
        data: DataFrame with price data
        fast_rsi_period: Fast RSI period
        slow_rsi_period: Slow RSI period
        rsi_overbought: RSI overbought threshold
        rsi_oversold: RSI oversold threshold
        ema_period: EMA period for trend filter
        stop_loss_pips: Stop loss in pips
        take_profit_pips: Take profit in pips

    Returns:
        DataFrame with signals
    """
    df = data.copy()

    # Calculate fast RSI
    delta_fast = df["mid"].diff()
    gain_fast = delta_fast.where(delta_fast > 0, 0)
    loss_fast = -delta_fast.where(delta_fast < 0, 0)

    avg_gain_fast = gain_fast.rolling(window=fast_rsi_period).mean()
    avg_loss_fast = loss_fast.rolling(window=fast_rsi_period).mean()

    rs_fast = avg_gain_fast / avg_loss_fast
    df["rsi_fast"] = 100 - (100 / (1 + rs_fast))

    # Calculate slow RSI
    delta_slow = df["mid"].diff()
    gain_slow = delta_slow.where(delta_slow > 0, 0)
    loss_slow = -delta_slow.where(delta_slow < 0, 0)

    avg_gain_slow = gain_slow.rolling(window=slow_rsi_period).mean()
    avg_loss_slow = loss_slow.rolling(window=slow_rsi_period).mean()

    rs_slow = avg_gain_slow / avg_loss_slow
    df["rsi_slow"] = 100 - (100 / (1 + rs_slow))

    # Calculate EMA for trend filter
    df["ema"] = df["mid"].ewm(span=ema_period, adjust=False).mean()

    # Initialize signal column
    df["signal"] = 0

    # Generate signals
    for i in range(1, len(df)):
        # Long signal: Fast RSI crosses above oversold, Slow RSI below 50, and price above EMA
        if (
            df["rsi_fast"][i - 1] <= rsi_oversold
            and df["rsi_fast"][i] > rsi_oversold
            and df["rsi_slow"][i] < 50
            and df["mid"][i] > df["ema"][i]
        ):
            df.loc[df.index[i], "signal"] = 1

        # Short signal: Fast RSI crosses below overbought, Slow RSI above 50, and price below EMA
        elif (
            df["rsi_fast"][i - 1] >= rsi_overbought
            and df["rsi_fast"][i] < rsi_overbought
            and df["rsi_slow"][i] > 50
            and df["mid"][i] < df["ema"][i]
        ):
            df.loc[df.index[i], "signal"] = -1

    return df


def main():
    """Run backtests for all strategies on all available currency pairs."""
    # Define currency pairs to test
    currency_pairs = ["eur_usd", "gbp_usd", "aud_usd", "aud_chf"]

    # Define strategies to test
    strategies = [
        {
            "name": "Moving Average Crossover",
            "function": ma_crossover_strategy,
            "params": {
                "fast_ma": 10,
                "slow_ma": 50,
                "trend_ma": 200,
                "stop_loss_pips": 20,
                "take_profit_pips": 40,
            },
        },
        {
            "name": "Bollinger Band Mean Reversion",
            "function": bollinger_band_strategy,
            "params": {
                "ma_period": 20,
                "std_dev": 2.0,
                "rsi_period": 14,
                "rsi_overbought": 70,
                "rsi_oversold": 30,
                "stop_loss_pips": 15,
                "take_profit_pips": 30,
            },
        },
        {
            "name": "RSI with ADX Filter",
            "function": rsi_adx_strategy,
            "params": {
                "rsi_period": 14,
                "rsi_overbought": 70,
                "rsi_oversold": 30,
                "adx_period": 14,
                "adx_threshold": 25,
                "stop_loss_pips": 25,
                "take_profit_pips": 50,
            },
        },
        {
            "name": "MACD with Support/Resistance",
            "function": macd_sr_strategy,
            "params": {
                "fast_period": 12,
                "slow_period": 26,
                "signal_period": 9,
                "sr_period": 20,
                "stop_loss_pips": 30,
                "take_profit_pips": 60,
            },
        },
        {
            "name": "Dual Timeframe Momentum",
            "function": dual_timeframe_momentum_strategy,
            "params": {
                "fast_rsi_period": 7,
                "slow_rsi_period": 21,
                "rsi_overbought": 70,
                "rsi_oversold": 30,
                "ema_period": 50,
                "stop_loss_pips": 25,
                "take_profit_pips": 50,
            },
        },
    ]

    # Create results directory
    results_dir = "backtest_results"
    os.makedirs(results_dir, exist_ok=True)

    # Run backtests for each currency pair and strategy
    for pair in currency_pairs:
        print(f"\n===== Testing strategies on {pair} =====")

        # Find the most recent data file for this pair
        data_path = f"data/raw/oanda/{pair}"
        if not os.path.exists(data_path):
            print(f"No data found for {pair}. Skipping...")
            continue

        # Get list of m30 data files
        data_files = glob.glob(f"{data_path}/{pair}_m30_*.csv")
        if not data_files:
            print(f"No m30 data files found for {pair}. Skipping...")
            continue

        # Sort by modification time (most recent first)
        data_files.sort(key=os.path.getmtime, reverse=True)
        latest_file = data_files[0]

        print(f"Using data file: {latest_file}")

        # Initialize backtester
        backtester = ForexBacktester(
            data_path=latest_file, initial_capital=10000.0, risk_per_trade=0.02
        )

        # Run backtests for each strategy
        for strategy in strategies:
            print(f"\nTesting {strategy['name']} on {pair}...")

            # Run backtest
            results = backtester.backtest_strategy(
                strategy_func=strategy["function"], strategy_params=strategy["params"]
            )

            # Print results
            backtester.print_results()

            # Plot and save results
            plot_path = f"{results_dir}/{pair}_{strategy['name'].replace('/', '_')}.png"
            backtester.plot_results(save_path=plot_path)

            # Reset backtester for next strategy
            backtester.trades = []
            backtester.results = None


if __name__ == "__main__":
    main()
