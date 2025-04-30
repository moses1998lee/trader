import warnings
from datetime import datetime, timedelta

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
import yfinance as yf
from statsmodels.tsa.stattools import adfuller, coint

warnings.filterwarnings("ignore")


class EnhancedPairsTrader:
    def __init__(
        self,
        lookback_period=252,  # Default to 1 trading year
        min_half_life=5,  # Minimum mean-reversion half-life (days)
        max_half_life=42,  # Maximum mean-reversion half-life (days)
        entry_threshold=2.0,  # Z-score entry threshold
        exit_threshold=0.0,  # Z-score exit threshold
        stop_loss_threshold=4.0,
    ):  # Stop-loss z-score threshold
        self.lookback_period = lookback_period
        self.min_half_life = min_half_life
        self.max_half_life = max_half_life
        self.entry_threshold = entry_threshold
        self.exit_threshold = exit_threshold
        self.stop_loss_threshold = stop_loss_threshold

    def download_data(self, tickers, start_date, end_date):
        """Download price data for a list of tickers"""
        print(f"Downloading data for {len(tickers)} tickers...")
        data = yf.download(tickers, start=start_date, end=end_date)
        try:
            data = data["Adj Close"]
        except:
            data = data["Close"]
        return data

    def find_cointegrated_pairs(self, data, significance=0.05):
        """Find cointegrated pairs among the provided tickers"""
        n = data.shape[1]
        pvalue_matrix = np.ones((n, n))
        keys = data.columns
        pairs = []

        print("Testing pairs for cointegration...")
        for i in range(n):
            for j in range(i + 1, n):
                stock1 = keys[i]
                stock2 = keys[j]
                result = coint(data[stock1], data[stock2])
                pvalue = result[1]
                pvalue_matrix[i, j] = pvalue

                if pvalue < significance:
                    pairs.append((stock1, stock2, pvalue))

        # Sort by p-value
        return sorted(pairs, key=lambda x: x[2])

    def calculate_half_life(self, spread):
        """Calculate the half-life of mean reversion for a spread"""
        spread_lag = spread.shift(1)
        spread_lag.iloc[0] = spread_lag.iloc[1]
        spread_ret = spread - spread_lag
        spread_ret.iloc[0] = spread_ret.iloc[1]
        spread_lag2 = sm.add_constant(spread_lag)

        model = sm.OLS(spread_ret, spread_lag2)
        res = model.fit()

        # Calculate half-life
        print(f"PARAMS:: {res.params}")
        half_life = -np.log(2) / res.params[1]
        return half_life

    def calculate_hedge_ratio(self, stock1_prices, stock2_prices, window=None):
        """Calculate the optimal hedge ratio using OLS regression"""
        if window is None:
            # Static hedge ratio using all data
            stock1_prices = sm.add_constant(stock1_prices)
            model = sm.OLS(stock2_prices, stock1_prices).fit()
            return model.params[1]
        else:
            # Rolling hedge ratio
            result = np.zeros(len(stock1_prices))
            for t in range(window, len(stock1_prices)):
                s1 = stock1_prices[t - window : t]
                s2 = stock2_prices[t - window : t]
                s1 = sm.add_constant(s1)
                model = sm.OLS(s2, s1).fit()
                result[t] = model.params[1]
            return result

    def calculate_spread(self, stock1_prices, stock2_prices, hedge_ratio):
        """Calculate the spread between two stocks using the hedge ratio"""
        if isinstance(hedge_ratio, (int, float)):
            # Static hedge ratio
            return stock2_prices - hedge_ratio * stock1_prices
        else:
            # Dynamic hedge ratio
            spread = np.zeros(len(stock1_prices))
            for i in range(len(stock1_prices)):
                if not np.isnan(hedge_ratio[i]):
                    spread[i] = stock2_prices[i] - hedge_ratio[i] * stock1_prices[i]
            return spread

    def calculate_zscore(self, spread, window=None):
        """Calculate the z-score of the spread"""
        if window is None:
            # Static z-score
            return (spread - spread.mean()) / spread.std()
        else:
            # Rolling z-score
            roll_mean = spread.rolling(window=window).mean()
            roll_std = spread.rolling(window=window).std()
            return (spread - roll_mean) / roll_std

    def generate_signals(self, zscore):
        """Generate trading signals based on z-score thresholds"""
        signals = np.zeros(len(zscore))

        # Long entry
        signals[zscore <= -self.entry_threshold] = 1
        # Short entry
        signals[zscore >= self.entry_threshold] = -1

        # Find exit points
        long_exit = (zscore >= self.exit_threshold) & (
            zscore.shift(1) < self.exit_threshold
        )
        short_exit = (zscore <= self.exit_threshold) & (
            zscore.shift(1) > self.exit_threshold
        )
        stop_loss = np.abs(zscore) >= self.stop_loss_threshold

        # Apply exits
        position = 0
        for i in range(1, len(signals)):
            # Carry forward previous position unless we have a reason to exit
            position = signals[i] if signals[i] != 0 else position

            # Exit if crossing threshold or stop loss
            if (
                (long_exit[i] and position > 0)
                or (short_exit[i] and position < 0)
                or stop_loss[i]
            ):
                position = 0

            signals[i] = position

        return signals

    def calculate_returns(self, stock1_prices, stock2_prices, signals, hedge_ratio):
        """Calculate strategy returns"""
        # Daily price changes
        stock1_returns = stock1_prices.pct_change()
        stock2_returns = stock2_prices.pct_change()

        # Make sure hedge ratio is the correct shape
        if isinstance(hedge_ratio, (int, float)):
            hedge_ratio = np.ones(len(signals)) * hedge_ratio

        # Strategy returns: Long stock2, short stock1 when signal = 1
        # Short stock2, long stock1 when signal = -1
        returns = np.zeros(len(signals))
        for i in range(1, len(signals)):
            if signals[i - 1] == 1:  # Long the spread
                returns[i] = stock2_returns[i] - hedge_ratio[i - 1] * stock1_returns[i]
            elif signals[i - 1] == -1:  # Short the spread
                returns[i] = -stock2_returns[i] + hedge_ratio[i - 1] * stock1_returns[i]

        # Assume 0.1% transaction cost when position changes
        transaction_cost = 0.001
        position_changes = np.diff(np.append(0, signals)) != 0
        returns[position_changes] -= transaction_cost

        return returns

    def backtest_pair(self, stock1, stock2, data, dynamic_hedge=True, window=63):
        """Backtest a pair trading strategy"""
        print(f"Backtesting pair: {stock1} - {stock2}")

        # Get price data
        stock1_prices = data[stock1]
        stock2_prices = data[stock2]

        # Calculate hedge ratio
        if dynamic_hedge:
            hedge_ratio = self.calculate_hedge_ratio(
                stock1_prices, stock2_prices, window=window
            )
            # Fill NaN values in the beginning
            hedge_ratio[:window] = hedge_ratio[window]
        else:
            hedge_ratio = self.calculate_hedge_ratio(stock1_prices, stock2_prices)

        # Calculate spread
        spread = self.calculate_spread(stock1_prices, stock2_prices, hedge_ratio)
        spread_series = pd.Series(spread, index=data.index)

        # Test for stationarity
        adf_result = adfuller(spread_series.dropna())
        is_stationary = adf_result[1] < 0.05

        # Calculate half-life
        half_life = self.calculate_half_life(spread_series.dropna())

        # Skip pairs that don't meet our criteria
        if (
            not is_stationary
            or half_life < self.min_half_life
            or half_life > self.max_half_life
        ):
            print(
                f"Pair {stock1}-{stock2} rejected: stationary={is_stationary}, half_life={half_life:.2f} days"
            )
            return None

        print(f"Pair {stock1}-{stock2} accepted: half_life={half_life:.2f} days")

        # Calculate z-score
        zscore = self.calculate_zscore(spread_series, window=window)

        # Generate signals
        signals = self.generate_signals(zscore)

        # Calculate returns
        returns = self.calculate_returns(
            stock1_prices, stock2_prices, signals, hedge_ratio
        )
        returns_series = pd.Series(returns, index=data.index)

        # Create performance metrics
        total_return = np.expm1(np.log1p(returns).sum())
        annualized_return = np.expm1(np.log1p(returns).sum() * (252 / len(returns)))
        annualized_vol = np.std(returns) * np.sqrt(252)
        sharpe_ratio = annualized_return / annualized_vol if annualized_vol > 0 else 0

        # Calculate drawdown
        cum_returns = (1 + returns_series).cumprod()
        running_max = cum_returns.cummax()
        drawdown = (cum_returns / running_max) - 1
        max_drawdown = drawdown.min()

        results = {
            "pair": f"{stock1}-{stock2}",
            "is_stationary": is_stationary,
            "half_life": half_life,
            "total_return": total_return,
            "annualized_return": annualized_return,
            "annualized_vol": annualized_vol,
            "sharpe_ratio": sharpe_ratio,
            "max_drawdown": max_drawdown,
            "returns": returns_series,
            "zscore": zscore,
            "signals": signals,
            "spread": spread_series,
        }

        return results

    def backtest_all_pairs(self, data, min_pairs=5, max_pairs=10):
        """Backtest all possible cointegrated pairs"""
        # Find cointegrated pairs
        pairs = self.find_cointegrated_pairs(data)
        print(f"Found {len(pairs)} cointegrated pairs")

        if len(pairs) == 0:
            print("No cointegrated pairs found. Try increasing the significance level.")
            return None

        # Limit to top pairs
        pairs = pairs[: min(max_pairs, len(pairs))]

        # Backtest each pair
        results = []
        for stock1, stock2, pvalue in pairs:
            result = self.backtest_pair(stock1, stock2, data)
            if result is not None:
                results.append(result)

            # Stop if we have enough successful pairs
            if len(results) >= min_pairs:
                break

        return results if results else None

    def run_portfolio_backtest(self, tickers, start_date, end_date):
        """Run a portfolio backtest using multiple pairs"""
        # Download data
        data = self.download_data(tickers, start_date, end_date)

        # Remove tickers with too many NaN values
        missing_threshold = len(data) * 0.05  # Allow 5% missing values
        valid_columns = [
            col for col in data.columns if data[col].isna().sum() < missing_threshold
        ]
        data = data[valid_columns]

        # Fill any remaining NaN values
        data = data.fillna(method="ffill").fillna(method="bfill")

        # Run backtest on all pairs
        results = self.backtest_all_pairs(data)

        if results is None:
            print("No viable pairs found. Try different parameters or more tickers.")
            return None

        # Create a portfolio of pairs
        portfolio_returns = pd.DataFrame(
            {result["pair"]: result["returns"] for result in results}
        )

        # Equal weight portfolio
        portfolio_returns["portfolio"] = portfolio_returns.mean(axis=1)

        # Calculate portfolio performance metrics
        returns = portfolio_returns["portfolio"]
        cum_returns = (1 + returns).cumprod()

        annualized_return = np.expm1(np.log1p(returns).sum() * (252 / len(returns)))
        annualized_vol = returns.std() * np.sqrt(252)
        sharpe_ratio = annualized_return / annualized_vol if annualized_vol > 0 else 0

        # Calculate drawdowns
        running_max = cum_returns.cummax()
        drawdown = (cum_returns / running_max) - 1
        max_drawdown = drawdown.min()

        print("\nPortfolio Performance:")
        print(f"Annualized Return: {annualized_return:.2%}")
        print(f"Annualized Volatility: {annualized_vol:.2%}")
        print(f"Sharpe Ratio: {sharpe_ratio:.2f}")
        print(f"Max Drawdown: {max_drawdown:.2%}")

        # Plot portfolio performance
        self.plot_portfolio_performance(results, cum_returns, drawdown)

        return {
            "results": results,
            "portfolio_returns": portfolio_returns,
            "annualized_return": annualized_return,
            "annualized_vol": annualized_vol,
            "sharpe_ratio": sharpe_ratio,
            "max_drawdown": max_drawdown,
        }

    def plot_portfolio_performance(self, pair_results, cum_returns, drawdown):
        """Plot portfolio performance charts"""
        fig, axes = plt.subplots(3, 1, figsize=(12, 18))

        # Plot cumulative returns
        axes[0].plot(cum_returns.index, cum_returns, "b-", linewidth=2)
        axes[0].set_title("Portfolio Cumulative Returns", fontsize=14)
        axes[0].set_ylabel("Cumulative Return", fontsize=12)
        axes[0].set_xlabel("Date", fontsize=12)
        axes[0].grid(True)

        # Plot drawdowns
        axes[1].fill_between(drawdown.index, drawdown, 0, color="r", alpha=0.3)
        axes[1].set_title("Portfolio Drawdowns", fontsize=14)
        axes[1].set_ylabel("Drawdown", fontsize=12)
        axes[1].set_xlabel("Date", fontsize=12)
        axes[1].grid(True)

        # Plot example pair spread and signals
        if pair_results:
            best_pair = max(pair_results, key=lambda x: x["sharpe_ratio"])
            spread = best_pair["spread"]
            zscore = best_pair["zscore"]
            signals = best_pair["signals"]

            # Plot Z-score with entry/exit thresholds
            axes[2].plot(zscore.index, zscore, "k-", linewidth=1)
            axes[2].axhline(
                y=self.entry_threshold, color="r", linestyle="--", alpha=0.5
            )
            axes[2].axhline(
                y=-self.entry_threshold, color="g", linestyle="--", alpha=0.5
            )
            axes[2].axhline(y=self.exit_threshold, color="b", linestyle=":", alpha=0.5)
            axes[2].axhline(y=-self.exit_threshold, color="b", linestyle=":", alpha=0.5)

            # Plot entry and exit points
            long_entries = zscore.index[signals == 1]
            short_entries = zscore.index[signals == -1]
            exits = zscore.index[(signals == 0) & (signals.shift(1) != 0)]

            axes[2].scatter(
                long_entries,
                zscore[long_entries],
                color="g",
                marker="^",
                s=100,
                label="Long Entry",
            )
            axes[2].scatter(
                short_entries,
                zscore[short_entries],
                color="r",
                marker="v",
                s=100,
                label="Short Entry",
            )
            axes[2].scatter(
                exits, zscore[exits], color="b", marker="o", s=70, label="Exit"
            )

            axes[2].set_title(
                f"Best Pair: {best_pair['pair']} - Z-Score and Trading Signals",
                fontsize=14,
            )
            axes[2].set_ylabel("Z-Score", fontsize=12)
            axes[2].set_xlabel("Date", fontsize=12)
            axes[2].legend()
            axes[2].grid(True)

        plt.tight_layout()
        plt.show()

    def optimize_parameters(self, tickers, start_date, end_date):
        """Simple grid search to optimize strategy parameters"""
        # Download data
        data = self.download_data(tickers, start_date, end_date)
        data = data.fillna(method="ffill").fillna(method="bfill")

        # Find top pairs
        pairs = self.find_cointegrated_pairs(data)[
            :5
        ]  # Use top 5 pairs for optimization

        if len(pairs) == 0:
            print("No cointegrated pairs found. Cannot optimize parameters.")
            return

        # Grid search parameters
        entry_thresholds = [1.5, 2.0, 2.5]
        exit_thresholds = [0.0, 0.5]

        best_sharpe = -np.inf
        best_params = {}

        print("\nOptimizing parameters...")
        for entry in entry_thresholds:
            for exit in exit_thresholds:
                # Update parameters
                self.entry_threshold = entry
                self.exit_threshold = exit

                # Test parameters on first pair
                stock1, stock2, _ = pairs[0]
                result = self.backtest_pair(stock1, stock2, data)

                if result and result["sharpe_ratio"] > best_sharpe:
                    best_sharpe = result["sharpe_ratio"]
                    best_params = {
                        "entry_threshold": entry,
                        "exit_threshold": exit,
                        "sharpe_ratio": best_sharpe,
                    }

        print(
            f"Best parameters found: Entry={best_params['entry_threshold']}, Exit={best_params['exit_threshold']}"
        )
        print(f"Best Sharpe ratio: {best_params['sharpe_ratio']:.2f}")

        # Restore best parameters
        self.entry_threshold = best_params["entry_threshold"]
        self.exit_threshold = best_params["exit_threshold"]

        return best_params


# Example usage of the enhanced pairs trading strategy
def run_strategy_test():
    # Define sectors and tickers for testing
    sectors = {
        "Energy": [
            "XOM",
            "CVX",
            "COP",
            "EOG",
            "SLB",
            "OXY",
            "MPC",
            "PSX",
            "VLO",
        ],
        "Technology": [
            "AAPL",
            "MSFT",
            "GOOGL",
            "AMZN",
            "META",
            "NVDA",
            "ADBE",
            "CRM",
            "INTC",
            "CSCO",
        ],
        "Financial": ["JPM", "BAC", "WFC", "C", "GS", "MS", "BLK", "AXP", "V", "MA"],
        "Healthcare": [
            "UNH",
            "JNJ",
            "PFE",
            "MRK",
            "ABBV",
            "LLY",
            "TMO",
            "ABT",
            "DHR",
            "BMY",
        ],
    }

    # Initialize the trader
    trader = EnhancedPairsTrader(
        lookback_period=252,
        min_half_life=7,
        max_half_life=60,
        entry_threshold=2.0,
        exit_threshold=0.0,
    )

    # Choose time period for backtesting
    end_date = datetime.now()
    start_date = end_date - timedelta(days=365 * 3)  # 3 years of data

    # Run strategy on each sector
    best_sector = None
    best_sharpe = -np.inf

    for sector_name, tickers in sectors.items():
        print(f"\n=== Testing {sector_name} Sector ===")

        # Optimize parameters
        trader.optimize_parameters(tickers, start_date, end_date - timedelta(days=365))

        # Run backtest with optimized parameters
        results = trader.run_portfolio_backtest(tickers, start_date, end_date)

        if results and results["sharpe_ratio"] > best_sharpe:
            best_sharpe = results["sharpe_ratio"]
            best_sector = sector_name

    print(
        f"\nBest performing sector: {best_sector} with Sharpe Ratio: {best_sharpe:.2f}"
    )

    return trader


# Run the strategy test
if __name__ == "__main__":
    trader = run_strategy_test()
