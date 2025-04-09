"""
Module: hyperparameter_tuning_with_save_results

This module implements hyperparameter tuning on simulation parameters for a mean reversion
trading strategy and saves the final account capital results for each hyperparameter
combination into CSV and text files.

The results for each simulation run include:
    - The hyperparameters used.
    - The final account capital achieved.
    - The currency symbol.

Data is saved in the 'results' directory with one CSV and one TXT file per currency.

Assumptions:
    - The simulation classes Account, MeanReversion, and Simulator are available.
    - Simulator.simulate returns a numeric value representing the final account capital.
    - Price data for each currency is stored in a CSV file following the naming convention:
      f"data/raw/{{currency}}/{{currency}}_{{granularity}}_{{dataset_date_str}}.csv".
"""

import itertools
import os
from typing import Any, Dict, List

import pandas as pd

from src.simulator import Account, Simulator
from src.strategy import MeanReversion

# -----------------------------------------------------------------------------
# Hyperparameter Grid Definition
# -----------------------------------------------------------------------------
PARAM_GRIDS: Dict[str, List[Any]] = {
    "max_risk": [0.005, 0.01, 0.02],
    "risk_reward": [(1, 0.8), (1, 1.0), (1, 1.2), (1, 1.5), (1, 2), (1, 3)],
    "mid_reversion_window": [
        10,
        15,
        20,
        25,
        30,
        35,
        40,
        45,
        50,
        55,
        60,
        65,
        70,
        75,
        80,
        85,
        90,
        95,
        100,
        105,
        110,
        115,
        120,
    ],
    "mid_std_threshold": [0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5],
    "vol_reversion_window": [
        25,
        30,
        35,
        40,
        45,
        50,
        55,
        60,
        65,
        70,
        75,
        80,
        85,
        90,
        95,
        100,
        105,
        110,
        115,
        120,
    ],
    "vol_std_threshold": [0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5],
    "vol_spike_window": [
        5,
        10,
        15,
        20,
        25,
        30,
    ],
    "stoploss_percent_std": [2, 4, 6, 8, 10, 12, 14, 16],
}


def tune_currency_and_save_results(
    currency: str, df: pd.DataFrame, start: str, end: str, starting_capital: float
) -> pd.DataFrame:
    """
    Performs hyperparameter tuning on a single currency simulation and records each result.

    For every possible combination of hyperparameters (as defined in PARAM_GRIDS), the function
    runs the simulation, records the parameter values along with the final account capital achieved,
    and returns a DataFrame with all results.

    Args:
        currency (str): The currency symbol (e.g., "EUR_USD").
        df (pd.DataFrame): Historical price data for the currency.
        start (str): Start date for the simulation in ddmmyyyy format.
        end (str): End date for the simulation in ddmmyyyy format.
        starting_capital (float): The initial account capital.

    Returns:
        pd.DataFrame: A DataFrame containing each hyperparameter combination and its corresponding
                      final account capital.
    """
    results: List[Dict[str, Any]] = []
    keys = list(PARAM_GRIDS.keys())
    all_combinations = itertools.product(*(PARAM_GRIDS[key] for key in keys))

    for combo in all_combinations:
        params = dict(zip(keys, combo))

        # Instantiate simulation objects with current hyperparameters.
        account = Account(starting_capital)
        strategy = MeanReversion(
            mid_reversion_window=params["mid_reversion_window"],
            vol_reversion_window=params["vol_reversion_window"],
            mid_reversion_std=params["mid_std_threshold"],
            vol_reversion_std=params["vol_std_threshold"],
            vol_spike_window=params["vol_spike_window"],
            stoploss_percent_std=params["stoploss_percent_std"],
        )
        simulator = Simulator(
            account=account,
            max_risk=params["max_risk"],
            risk_reward=params["risk_reward"],
            strategy=strategy,
            df=df,
        )
        # Run simulation (simulate is assumed to return the final capital as a float).
        simulator.simulate(start, end)
        final_capital = simulator.account.capital
        initial_capital = simulator.initial_capital

        # Record the simulation result.
        record = params.copy()
        record["capital_gains"] = (final_capital - initial_capital) / initial_capital
        record["currency"] = currency
        results.append(record)

    return pd.DataFrame(results)


def tune_all_currencies_and_save(
    currencies: List[str],
    dataset_date_str: str,
    granularity: str,
    start: str,
    end: str,
    starting_capital: float,
    results_dir: str = "results",
) -> None:
    """
    Performs hyperparameter tuning on multiple currencies and saves the results.

    For each provided currency, the function loads its dataset, runs the hyperparameter tuning
    (saving each result to a DataFrame), and then writes the results to both a CSV file and a
    plain text file in the specified directory.

    Args:
        currencies (List[str]): List of currency symbols (e.g., ["EUR_USD", "USD_JPY"]).
        dataset_date_str (str): String representing the date range in the dataset filename.
        granularity (str): Data resolution (e.g., "M1" for one-minute data).
        start (str): Simulation start date (format: ddmmyyyy).
        end (str): Simulation end date (format: ddmmyyyy).
        starting_capital (float): The initial account capital.
        results_dir (str, optional): Directory to save the result files. Defaults to "results".
    """
    # Create the results directory if it does not exist.
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)

    for cur in currencies:
        filepath = f"data/raw/{cur}/{cur}_{granularity}_{dataset_date_str}.csv"
        df = pd.read_csv(filepath, index_col=0, parse_dates=True)

        print(f"Running hyperparameter tuning for {cur}...")
        result_df = tune_currency_and_save_results(
            cur, df, start, end, starting_capital
        )

        date_dir = os.path.join(results_dir, f"{start_date}-{end_date}")
        os.makedirs(date_dir, exist_ok=True)

        sorted_df = result_df.sort_values(by="capital_gains", ascending=False)
        csv_filepath = os.path.join(
            date_dir, f"{cur}_{granularity}_{start}-{end}_results.csv"
        )

        sorted_df.to_csv(csv_filepath, index=False)

        # Plain Text
        txt_filepath = os.path.join(
            date_dir, f"{cur}_{granularity}_{start}-{end}_results.txt"
        )
        with open(txt_filepath, "w") as file:
            file.write(sorted_df.to_string(index=False))

        print(
            f"Results for {cur} saved as:\n  CSV: {csv_filepath}\n  TXT: {txt_filepath}"
        )


if __name__ == "__main__":
    """
    Main entry point for hyperparameter tuning with result saving.

    This script loads simulation data for each currency, runs a grid search over the defined
    hyperparameter ranges, and saves the results (hyperparameter settings and final capital)
    in both CSV and TXT formats.
    """
    CURRENCIES = ["EUR_JPY", "EUR_USD", "USD_CAD", "USD_JPY"]
    DATASET_DATE_STR = "01012010_31122024"
    GRANULARITY = "M1"
    STARTING_CAPITAL = 1000

    for year in range(2010, 2024 + 1):
        start_date = f"0101{year}"
        end_date = f"3112{year}"
        tune_all_currencies_and_save(
            CURRENCIES,
            DATASET_DATE_STR,
            GRANULARITY,
            start_date,
            end_date,
            STARTING_CAPITAL,
        )
