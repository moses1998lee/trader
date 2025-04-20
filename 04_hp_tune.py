"""
Module: optuna_hyperparameter_tuning

This module replaces the exhaustive grid search with Optuna to optimize and prune
hyperparameters for a mean reversion trading strategy. It loops over currencies and years,
runs an Optuna study for each combination, and saves all trial results (ranked by final performance)
in CSV and TXT formats.

Key Features:
    - Automated hyperparameter search (Bayesian/categorical sampling).
    - Pruning of unpromising trials based on intermediate evaluations (MedianPruner).
    - Full record of trials (best to worst) exported for analysis.

Usage:
    python optuna_hyperparameter_tuning.py
"""

import os
from typing import Any, Dict

import optuna
import pandas as pd

from src.simulator import Account, Simulator
from src.strategy import MeanReversion


def simulate_for_trial(
    trial: optuna.Trial,
    start_year: int,
    end_year: int,
    df: pd.DataFrame,
    starting_capital: float,
    simulate_verbose: bool,
) -> float:
    """
    Defines how Optuna will run a single simulation trial, suggesting hyperparameters
    and returning the final capital gains ratio as an objective to maximize.

    Args:
        trial (optuna.Trial): The Optuna trial object for sampling hyperparameters and reporting results.
        currency (str): The currency symbol (e.g., "EUR_USD").
        df (pd.DataFrame): Historical price data for the specified currency.
        start_date (str): Simulation start date (ddmmyyyy).
        end_date (str): Simulation end date (ddmmyyyy).
        starting_capital (float): Initial capital for the simulation.

    Returns:
        float: The capital gains ratio from the start to the end of the simulation.
               A higher ratio is considered better, so it is the direct objective for Optuna.
    """
    # ---- Suggest hyperparameters via Optuna ----
    max_risk = trial.suggest_float("max_risk", 0.01, 0.05, step=0.01)
    # You can also try making risk_reward a float, but here we'll keep it categorical
    # with discrete pairs:
    reward_factor = trial.suggest_float("reward_factor", 0.8, 3.0, step=0.2)

    mid_reversion_window = trial.suggest_int("mid_reversion_window", 10, 120, step=5)
    mid_std_threshold = trial.suggest_float("mid_std_threshold", 0.5, 5.0, step=0.5)
    vol_reversion_window = trial.suggest_int("vol_reversion_window", 25, 120, step=5)
    vol_std_threshold = trial.suggest_float("vol_std_threshold", 0.5, 5.0, step=0.5)
    vol_spike_window = trial.suggest_int("vol_spike_window", 5, 30, step=5)
    stoploss_percent_std = trial.suggest_int("stoploss_percent_std", 2, 16, step=2)

    # ---- Run the simulation using the suggested hyperparameters ----
    yearly_gains = []
    for year in range(start_year, end_year + 1):
        start_date = f"0101{year}"
        end_date = f"3112{year}"

        account = Account(starting_capital)
        strategy = MeanReversion(
            mid_reversion_window=mid_reversion_window,
            vol_reversion_window=vol_reversion_window,
            mid_reversion_std=mid_std_threshold,
            vol_reversion_std=vol_std_threshold,
            vol_spike_window=vol_spike_window,
            stoploss_percent_std=stoploss_percent_std,
        )

        simulator = Simulator(
            account=account,
            max_risk=max_risk,
            risk_reward=(1, reward_factor),
            strategy=strategy,
            df=df,
            verbose=simulate_verbose,
        )

        print(f"CAPITAL: {simulator.account.capital}")
        # Run the simulation for [start_date, end_date].
        simulator.simulate(start_date, end_date)
        print(f"CAPITAL: {simulator.account.capital}")

        final_capital = simulator.account.capital
        initial_capital = simulator.initial_capital

        gains_ratio = (final_capital - initial_capital) / initial_capital

        yearly_gains.append(gains_ratio)

    # Maximize worst gain
    worst_gain = min(yearly_gains)

    # ---- Pruning Example (Optional) ----
    # If you can track intermediate results, you can do:
    # trial.report(worst_gain, step=step_number)
    # if trial.should_prune():
    #     raise optuna.exceptions.TrialPruned()

    # Since we want to maximize the gains ratio, we directly return it (Optuna will handle the rest).
    return worst_gain


def run_optuna(
    currency: str,
    df: pd.DataFrame,
    start_year: int,
    end_year: int,
    starting_capital: float,
    n_trials: int = 50,
    simulate_verbose: bool = False,
) -> pd.DataFrame:
    """
    Runs an Optuna study for a given currency and year range, returning a DataFrame
    of all trials (sorted by best -> worst).

    Args:
        currency (str): The currency symbol.
        df (pd.DataFrame): Historical data for that currency.
        start_date (str): Simulation start date (ddmmyyyy).
        end_date (str): Simulation end date (ddmmyyyy).
        starting_capital (float): Initial capital for the simulation.
        n_trials (int, optional): Number of trials for Optuna. Default is 50.

    Returns:
        pd.DataFrame: A DataFrame with each trial's hyperparameters, final gains,
                      plus any other info relevant to the study.
    """

    def objective(trial: optuna.Trial) -> float:
        return simulate_for_trial(
            trial, start_year, end_year, df, starting_capital, simulate_verbose
        )

    study = optuna.create_study(
        direction="maximize", pruner=optuna.pruners.MedianPruner(n_warmup_steps=5)
    )

    # Run optimization for n_trials.
    study.optimize(objective, n_trials=n_trials)

    # Gather all trial results in a DataFrame.
    records = []
    for t in study.trials:
        row: Dict[str, Any] = {
            "trial_id": t.number,
            "worst_capital_gain": t.value,  # This is the final objective value (gains ratio).
        }
        row.update(t.params)  # t.params is a dict of hyperparameter: value
        records.append(row)

    df_results = pd.DataFrame(records)
    df_sorted = df_results.sort_values(
        by="worst_capital_gain", ascending=False
    ).reset_index(drop=True)
    return df_sorted


def tune_all_currencies_and_save_optuna(
    currencies: list[str],
    dataset_date_str: str,
    granularity: str,
    start_year: int,
    end_year: int,
    starting_capital: float,
    n_trials: int = 50,
    results_dir: str = "results_optuna",
    simulate_verbose: bool = False,
) -> None:
    """
    Loops through each currency and each year, running an Optuna study to tune hyperparameters,
    saving the study's trial results to both CSV and TXT files for analysis.

    Args:
        currencies (list[str]): List of currencies to tune (e.g. ["EUR_USD", "USD_JPY"]).
        dataset_date_str (str): String representing the date range in the dataset filename.
        granularity (str): Data resolution (e.g., "M1").
        start_year (int): The first year in the range for simulation runs.
        end_year (int): The last year in the range for simulation runs.
        starting_capital (float): The initial account capital for each simulation.
        n_trials (int): Number of Optuna trials per currency-year combination.
        results_dir (str): Directory in which to store the results.
    """
    os.makedirs(results_dir, exist_ok=True)

    # for year in range(start_year, end_year + 1):
    #     start_date = f"0101{year}"
    #     end_date = f"3112{year}"
    #     date_dir = os.path.join(results_dir, f"{start_date}-{end_date}")
    #     os.makedirs(date_dir, exist_ok=True)

    for cur in currencies:
        # Load the dataset for the specific currency.
        filepath = f"data/raw/{cur}/{cur}_{granularity}_{dataset_date_str}.csv"
        df = pd.read_csv(filepath, index_col=0, parse_dates=True)

        print(f"Optuna tuning for {cur} from {start_year} to {end_year} ...")

        # Run the study and get sorted results (best -> worst).
        df_sorted = run_optuna(
            currency=cur,
            df=df,
            start_year=start_year,
            end_year=end_year,
            starting_capital=starting_capital,
            n_trials=n_trials,
            simulate_verbose=simulate_verbose,
        )

        # Save results to CSV (sorted).
        csv_path = os.path.join(
            results_dir, f"{cur}_{granularity}_{start_year}-{end_year}_optuna.csv"
        )
        df_sorted.to_csv(csv_path, index=False)

        # Also save to TXT (sorted).
        txt_path = os.path.join(
            results_dir, f"{cur}_{granularity}_{start_year}-{end_year}_optuna.txt"
        )
        with open(txt_path, "w") as txt_file:
            txt_file.write(df_sorted.to_string(index=False))

        print(
            f"Saved Optuna results for {cur}, {start_year}-{end_year}:\n  {csv_path}\n  {txt_path}"
        )


if __name__ == "__main__":
    """
    Main entry point for Optuna-based hyperparameter tuning with pruning.
    This code:
      1) Iterates over years from 2010 to 2024.
      2) Iterates over the provided currencies.
      3) Runs Optuna to find optimal hyperparameters for each (currency, year) subset.
      4) Saves a sorted record of all trials in CSV and TXT so you can see best-to-worst runs.
    """
    CURRENCIES = ["EUR_JPY", "EUR_USD", "USD_CAD", "USD_JPY"]
    DATASET_DATE_STR = "01012010_31122024"
    GRANULARITY = "M1"
    STARTING_CAPITAL = 1000
    N_TRIALS = 100
    SIMULATE_VERBOSE = False

    tune_all_currencies_and_save_optuna(
        currencies=CURRENCIES,
        dataset_date_str=DATASET_DATE_STR,
        granularity=GRANULARITY,
        start_year=2010,
        end_year=2024,
        starting_capital=STARTING_CAPITAL,
        n_trials=N_TRIALS,
        results_dir="results_optuna",
        simulate_verbose=SIMULATE_VERBOSE,
    )
