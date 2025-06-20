#%%
import numpy as np
import pandas as pd


def backtest_coil_percentile_volume(
    df, window=15, M=3, hold=5, spread_q=0.25, vol_q=0.5
):
    """
    Backtest coil-breakout with percentile-based coil thresholds and volume filter.

    Parameters:
    - window: rolling window for coil metrics
    - M: consecutive bars for stability
    - hold: hold time in minutes
    - spread_q: quantile for spread & price_range coil threshold (e.g., 0.25)
    - vol_q: volume quantile threshold (e.g., 0.5 for above-median)
    """
    df = df.copy().reset_index(drop=True)
    df["mid"] = (df["bid"] + df["ask"]) / 2
    df["spread"] = df["ask"] - df["bid"]

    # Rolling coil metrics
    df["spread_std"] = df["spread"].rolling(window).std()
    df["price_range"] = (
        df["mid"].rolling(window).max() - df["mid"].rolling(window).min()
    )

    # Percentile thresholds
    df["spread_thr"] = df["spread_std"].rolling(window).quantile(spread_q)
    df["range_thr"] = df["price_range"].rolling(window).quantile(spread_q)
    df["vol_thr"] = df["volume"].rolling(window).quantile(vol_q)

    # Coil & Volume readiness
    df["coil_ready"] = (df["spread_std"] <= df["spread_thr"]) & (
        df["price_range"] <= df["range_thr"]
    )
    df["coil_stable"] = df["coil_ready"].rolling(M).sum() == M
    df["vol_ready"] = df["volume"] >= df["vol_thr"]

    # Breakout condition
    df["prior_high"] = df["mid"].rolling(window).max().shift(1)
    df["breakout"] = df["mid"] > df["prior_high"]

    # Entry signal: coil + volume + breakout
    df["entry"] = df["coil_stable"].shift(1) & df["vol_ready"] & df["breakout"]

    # Collect trade returns
    returns = []
    for pos in df.index[df["entry"]]:
        if pos + hold < len(df):
            entry_price = df.at[pos, "ask"]
            exit_price = df.at[pos + hold, "bid"]
            returns.append((exit_price - entry_price) / entry_price)

    if not returns:
        return None

    ret = np.array(returns)
    equity = np.cumprod(1 + ret)
    metrics = {
        "total_return": equity[-1] - 1,
        "num_trades": len(ret),
        "win_rate": float((ret > 0).sum()) / len(ret),
        "avg_return": float(ret.mean()),
        "std_return": float(ret.std(ddof=1)),
        "sharpe": float(ret.mean() / ret.std(ddof=1))
        if ret.std(ddof=1) != 0
        else np.nan,
        "max_drawdown": float(
            (
                (equity - np.maximum.accumulate(equity)) / np.maximum.accumulate(equity)
            ).min()
        ),
    }
    return metrics


# Example usage on EUR/USD dataset
df_eurusd = pd.read_csv(
    "data/raw/oanda/eur_usd/eur_usd_m1_01042025_01052025.csv", parse_dates=["time"]
)
results_eurusd = backtest_coil_percentile_volume(
    df_eurusd, window=15, M=3, hold=5, spread_q=0.25, vol_q=0.5
)

print("EUR/USD metrics with percentile coil & volume filter:", results_eurusd)
#%%