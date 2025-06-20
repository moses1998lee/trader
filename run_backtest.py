import os

import matplotlib.pyplot as plt

from forex_strategy_backtest import (
    ForexBacktester,
    bollinger_band_strategy,
    dual_timeframe_momentum_strategy,
    ma_crossover_strategy,
    macd_sr_strategy,
    rsi_adx_strategy,
)


def run_single_strategy_backtest(
    data_path,
    strategy_func,
    strategy_params,
    strategy_name,
    initial_capital=10000.0,
    risk_per_trade=0.02,
    save_results=True,
    results_dir="backtest_results",
):
    """
    Run a backtest for a single strategy on a single currency pair.

    Args:
        data_path: Path to the data file
        strategy_func: Strategy function to use
        strategy_params: Parameters for the strategy
        strategy_name: Name of the strategy
        initial_capital: Initial capital for backtesting
        risk_per_trade: Risk per trade as a percentage of capital
        save_results: Whether to save the results
        results_dir: Directory to save results to
    """
    print(f"\n===== Testing {strategy_name} =====")
    print(f"Using data file: {data_path}")

    # Initialize backtester
    backtester = ForexBacktester(
        data_path=data_path,
        initial_capital=initial_capital,
        risk_per_trade=risk_per_trade,
    )

    # Run backtest
    results = backtester.backtest_strategy(
        strategy_func=strategy_func, strategy_params=strategy_params
    )

    # Print results
    backtester.print_results()

    # Plot and save results if requested
    if save_results:
        os.makedirs(results_dir, exist_ok=True)
        pair_name = os.path.basename(data_path).split("_")[0]
        plot_path = f"{results_dir}/{pair_name}_{strategy_name.replace('/', '_')}.png"
        backtester.plot_results(save_path=plot_path)
    else:
        backtester.plot_results()

    return results


def run_all_strategies_on_pair(
    data_path,
    initial_capital=10000.0,
    risk_per_trade=0.02,
    save_results=True,
    results_dir="backtest_results",
):
    """
    Run all strategies on a single currency pair.

    Args:
        data_path: Path to the data file
        initial_capital: Initial capital for backtesting
        risk_per_trade: Risk per trade as a percentage of capital
        save_results: Whether to save the results
        results_dir: Directory to save results to
    """
    pair_name = os.path.basename(data_path).split("_")[0]
    print(f"\n===== Testing all strategies on {pair_name} =====")

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

    # Store results for comparison
    all_results = {}

    # Run backtests for each strategy
    for strategy in strategies:
        backtester = ForexBacktester(
            data_path=data_path,
            initial_capital=initial_capital,
            risk_per_trade=risk_per_trade,
        )

        print(f"\nTesting {strategy['name']}...")

        # Run backtest
        results = backtester.backtest_strategy(
            strategy_func=strategy["function"], strategy_params=strategy["params"]
        )

        # Print results
        backtester.print_results()

        # Plot and save results if requested
        if save_results:
            os.makedirs(results_dir, exist_ok=True)
            plot_path = (
                f"{results_dir}/{pair_name}_{strategy['name'].replace('/', '_')}.png"
            )
            backtester.plot_results(save_path=plot_path)

        # Store results for comparison
        all_results[strategy["name"]] = {
            "monthly_return": results["monthly_return"],
            "max_drawdown": results["max_drawdown"],
            "sharpe_ratio": results["sharpe_ratio"],
            "win_rate": results["win_rate"],
            "profit_factor": results["profit_factor"],
            "total_trades": results["total_trades"],
        }

    # Compare strategies
    compare_strategies(all_results, pair_name, save_results, results_dir)

    return all_results


def compare_strategies(
    results, pair_name, save_results=True, results_dir="backtest_results"
):
    """
    Compare the performance of different strategies.

    Args:
        results: Dictionary of strategy results
        pair_name: Name of the currency pair
        save_results: Whether to save the comparison plot
        results_dir: Directory to save results to
    """
    # Create figure for comparison
    fig, axs = plt.subplots(2, 2, figsize=(14, 10))

    # Extract strategy names and metrics
    strategies = list(results.keys())
    monthly_returns = [results[s]["monthly_return"] * 100 for s in strategies]
    max_drawdowns = [results[s]["max_drawdown"] * 100 for s in strategies]
    sharpe_ratios = [results[s]["sharpe_ratio"] for s in strategies]
    win_rates = [results[s]["win_rate"] * 100 for s in strategies]

    # Plot monthly returns
    axs[0, 0].bar(strategies, monthly_returns)
    axs[0, 0].set_title("Monthly Return (%)")
    axs[0, 0].set_xticklabels(strategies, rotation=45, ha="right")
    axs[0, 0].grid(True)

    # Plot max drawdowns
    axs[0, 1].bar(strategies, max_drawdowns, color="red")
    axs[0, 1].set_title("Max Drawdown (%)")
    axs[0, 1].set_xticklabels(strategies, rotation=45, ha="right")
    axs[0, 1].grid(True)

    # Plot Sharpe ratios
    axs[1, 0].bar(strategies, sharpe_ratios, color="green")
    axs[1, 0].set_title("Sharpe Ratio")
    axs[1, 0].set_xticklabels(strategies, rotation=45, ha="right")
    axs[1, 0].grid(True)

    # Plot win rates
    axs[1, 1].bar(strategies, win_rates, color="purple")
    axs[1, 1].set_title("Win Rate (%)")
    axs[1, 1].set_xticklabels(strategies, rotation=45, ha="right")
    axs[1, 1].grid(True)

    plt.tight_layout()

    # Save comparison plot if requested
    if save_results:
        os.makedirs(results_dir, exist_ok=True)
        plt.savefig(f"{results_dir}/{pair_name}_strategy_comparison.png")

    plt.show()

    # Print strategy ranking based on monthly return
    print("\n===== Strategy Ranking (by Monthly Return) =====")
    sorted_strategies = sorted(
        results.items(), key=lambda x: x[1]["monthly_return"], reverse=True
    )
    for i, (strategy, metrics) in enumerate(sorted_strategies):
        print(
            f"{i + 1}. {strategy}: {metrics['monthly_return']:.2%} monthly return, {metrics['max_drawdown']:.2%} max drawdown, {metrics['sharpe_ratio']:.2f} Sharpe ratio"
        )


def optimize_strategy_parameters(
    data_path,
    strategy_func,
    param_grid,
    strategy_name,
    initial_capital=10000.0,
    risk_per_trade=0.02,
    metric="monthly_return",
    save_results=True,
    results_dir="backtest_results",
):
    """
    Optimize strategy parameters using grid search.

    Args:
        data_path: Path to the data file
        strategy_func: Strategy function to optimize
        param_grid: Dictionary of parameter grids to search
        strategy_name: Name of the strategy
        initial_capital: Initial capital for backtesting
        risk_per_trade: Risk per trade as a percentage of capital
        metric: Metric to optimize ('monthly_return', 'sharpe_ratio', etc.)
        save_results: Whether to save the results
        results_dir: Directory to save results to
    """
    pair_name = os.path.basename(data_path).split("_")[0]
    print(f"\n===== Optimizing {strategy_name} for {pair_name} =====")

    # Generate all parameter combinations
    import itertools

    param_keys = list(param_grid.keys())
    param_values = list(param_grid.values())
    param_combinations = list(itertools.product(*param_values))

    # Store results
    optimization_results = []

    # Test each parameter combination
    for i, combination in enumerate(param_combinations):
        params = dict(zip(param_keys, combination))

        print(f"\nTesting combination {i + 1}/{len(param_combinations)}: {params}")

        # Initialize backtester
        backtester = ForexBacktester(
            data_path=data_path,
            initial_capital=initial_capital,
            risk_per_trade=risk_per_trade,
        )

        # Run backtest
        results = backtester.backtest_strategy(
            strategy_func=strategy_func, strategy_params=params
        )

        # Store results
        optimization_results.append(
            {
                "params": params,
                "monthly_return": results["monthly_return"],
                "max_drawdown": results["max_drawdown"],
                "sharpe_ratio": results["sharpe_ratio"],
                "win_rate": results["win_rate"],
                "profit_factor": results["profit_factor"],
                "total_trades": results["total_trades"],
            }
        )

    # Sort results by the specified metric
    optimization_results.sort(key=lambda x: x[metric], reverse=True)

    # Print top results
    print("\n===== Top Parameter Combinations =====")
    for i, result in enumerate(optimization_results[:5]):
        print(f"{i + 1}. Parameters: {result['params']}")
        print(f"   Monthly Return: {result['monthly_return']:.2%}")
        print(f"   Max Drawdown: {result['max_drawdown']:.2%}")
        print(f"   Sharpe Ratio: {result['sharpe_ratio']:.2f}")
        print(f"   Win Rate: {result['win_rate']:.2%}")
        print(f"   Profit Factor: {result['profit_factor']:.2f}")
        print(f"   Total Trades: {result['total_trades']}")

    # Run backtest with best parameters and plot results
    best_params = optimization_results[0]["params"]
    print("\n===== Running backtest with best parameters =====")
    print(f"Best parameters: {best_params}")

    backtester = ForexBacktester(
        data_path=data_path,
        initial_capital=initial_capital,
        risk_per_trade=risk_per_trade,
    )

    results = backtester.backtest_strategy(
        strategy_func=strategy_func, strategy_params=best_params
    )

    backtester.print_results()

    # Plot and save results if requested
    if save_results:
        os.makedirs(results_dir, exist_ok=True)
        plot_path = f"{results_dir}/{pair_name}_{strategy_name}_optimized.png"
        backtester.plot_results(save_path=plot_path)
    else:
        backtester.plot_results()

    return optimization_results


if __name__ == "__main__":
    # Example usage:

    # 1. Run a single strategy backtest
    # data_path = "data/raw/oanda/eur_usd/eur_usd_m30_01122024_01052025.csv"
    # run_single_strategy_backtest(
    #     data_path=data_path,
    #     strategy_func=ma_crossover_strategy,
    #     strategy_params={
    #         "fast_ma": 10,
    #         "slow_ma": 50,
    #         "trend_ma": 200,
    #         "stop_loss_pips": 20,
    #         "take_profit_pips": 40,
    #     },
    #     strategy_name="Moving Average Crossover"
    # )

    # 2. Run all strategies on a single pair
    # data_path = "data/raw/oanda/eur_usd/eur_usd_m30_01122024_01052025.csv"
    # run_all_strategies_on_pair(data_path)

    # 3. Optimize strategy parameters
    # data_path = "data/raw/oanda/eur_usd/eur_usd_m30_01122024_01052025.csv"
    # param_grid = {
    #     "fast_ma": [5, 10, 15],
    #     "slow_ma": [30, 50, 70],
    #     "trend_ma": [150, 200, 250],
    #     "stop_loss_pips": [15, 20, 25],
    #     "take_profit_pips": [30, 40, 50],
    # }
    # optimize_strategy_parameters(
    #     data_path=data_path,
    #     strategy_func=ma_crossover_strategy,
    #     param_grid=param_grid,
    #     strategy_name="Moving Average Crossover",
    #     metric="sharpe_ratio"
    # )

    # Run all strategies on all available currency pairs
    currency_pairs = ["eur_usd", "gbp_usd", "aud_usd", "aud_chf"]

    for pair in currency_pairs:
        # Find the most recent data file for this pair
        data_path = f"data/raw/oanda/{pair}"
        if not os.path.exists(data_path):
            print(f"No data found for {pair}. Skipping...")
            continue

        # Get list of m30 data files
        import glob

        data_files = glob.glob(f"{data_path}/{pair}_m30_*.csv")
        if not data_files:
            print(f"No m30 data files found for {pair}. Skipping...")
            continue

        # Sort by modification time (most recent first)
        data_files.sort(key=os.path.getmtime, reverse=True)
        latest_file = data_files[0]

        # Run all strategies on this pair
        run_all_strategies_on_pair(latest_file)
