# %%
"""
tick_whisper_analysis_with_bid_ask.py
=======================================

This module implements the Tick-Whisper Analysis algorithm on forex tick data
stored in a Pandas DataFrame containing bid, ask, and optionally volume columns.
It computes the midprice from bid and ask prices and applies a custom wavelet
denoising procedure with iterative soft thresholding to isolate high-frequency
liquidity signals.

**Disclaimer:** This code is for educational and experimental purposes only.
Actual trading systems would require extensive testing, optimization, and risk controls.

Example:
    To run the analysis on simulated forex tick data:
        python tick_whisper_analysis_with_bid_ask.py
"""

from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pywt


def soft_threshold(data: np.ndarray, threshold: float) -> np.ndarray:
    """
    Apply soft thresholding to the input data.

    Soft thresholding shrinks each coefficient towards zero by the threshold value,
    reducing noise while preserving significant features.

    :param data: Numpy array representing detail coefficients.
    :param threshold: The threshold value.
    :return: Thresholded coefficients as a numpy array.
    """
    return np.sign(data) * np.maximum(np.abs(data) - threshold, 0)


def iterative_thresholding(
    detail_coeffs: np.ndarray, max_iter: int = 10, tol: float = 1e-4
) -> np.ndarray:
    """
    Iteratively refine detail coefficients using soft thresholding until convergence.

    The threshold is updated dynamically using the median absolute deviation.

    :param detail_coeffs: Numpy array of detail coefficients.
    :param max_iter: Maximum number of iterations.
    :param tol: Convergence tolerance.
    :return: Refined detail coefficients.
    """
    coeffs = detail_coeffs.copy()
    for _ in range(max_iter):
        prev = coeffs.copy()
        med = np.median(np.abs(coeffs))
        threshold = med / 0.6745  # Similar to Donoho's threshold estimation
        coeffs = soft_threshold(coeffs, threshold)
        if np.linalg.norm(coeffs - prev) < tol:
            break
    return coeffs


def wavelet_denoise_series(
    data: np.ndarray, wavelet: str = "db4", level: int = 3
) -> np.ndarray:
    """
    Denoise a 1D time series using discrete wavelet transform (DWT) and iterative thresholding.

    This function decomposes the series, applies iterative thresholding to detail coefficients,
    and reconstructs the denoised signal.

    :param data: 1D numpy array of the signal (e.g., midprice).
    :param wavelet: Wavelet type to use (default 'db4').
    :param level: Decomposition level (default 3).
    :return: The reconstructed denoised signal.
    """
    coeffs = pywt.wavedec(data, wavelet, level=level)
    denoised_coeffs = [coeffs[0]]  # Keep approximation coefficients as is
    for detail in coeffs[1:]:
        refined_detail = iterative_thresholding(detail, max_iter=10, tol=1e-4)
        denoised_coeffs.append(refined_detail)
    return pywt.waverec(denoised_coeffs, wavelet)


def extract_high_frequency_component(
    series: np.ndarray, wavelet: str = "db4", level: int = 3
) -> np.ndarray:
    """
    Extract the high-frequency component from a time series.

    The high-frequency component is obtained by subtracting the denoised (smooth) signal
    from the original series.

    :param series: 1D numpy array (e.g., midprice).
    :param wavelet: Wavelet type (default 'db4').
    :param level: Decomposition level (default 3).
    :return: High-frequency component as a numpy array.
    """
    denoised = wavelet_denoise_series(series, wavelet, level)
    # Adjust length if needed
    high_freq = series - denoised[: len(series)]
    return high_freq


def process_forex_tick_data(
    tick_df: pd.DataFrame, wavelet: str = "db4", level: int = 3
) -> pd.DataFrame:
    """
    Process forex tick data stored in a DataFrame with 'bid' and 'ask' columns.

    This function computes the midprice, applies wavelet denoising, and extracts the high-frequency
    liquidity component.

    :param tick_df: Pandas DataFrame with at least 'bid' and 'ask' columns.
    :param wavelet: Wavelet type (default 'db4').
    :param level: Decomposition level (default 3).
    :return: DataFrame with added columns: 'midprice', 'denoised_midprice', 'high_freq_component'.
    """
    # Compute midprice: (bid + ask) / 2
    tick_df["midprice"] = (tick_df["bid"] + tick_df["ask"]) / 2.0

    # Convert the midprice column to a numpy array
    midprice_series = tick_df["midprice"].to_numpy()

    # Apply wavelet denoising to obtain a smooth version of the midprice
    denoised_midprice = wavelet_denoise_series(midprice_series, wavelet, level)

    # Extract the high-frequency component which may represent fleeting liquidity signals
    high_freq_component = midprice_series - denoised_midprice[: len(midprice_series)]

    # Add results to the DataFrame
    tick_df["denoised_midprice"] = denoised_midprice[: len(tick_df)]
    tick_df["high_freq_component"] = high_freq_component

    return tick_df


def simulate_forex_tick_dataframe(
    num_ticks: int = 1024, seed: int = 42
) -> pd.DataFrame:
    """
    Simulate a Pandas DataFrame of forex tick data including 'bid', 'ask', and 'volume'.

    This function creates a synthetic dataset that mimics tick-level fluctuations for a forex pair,
    for example, EUR/USD.

    :param num_ticks: Number of ticks to simulate (default 1024).
    :param seed: Seed for reproducibility (default 42).
    :return: A DataFrame with columns 'bid', 'ask', and 'volume'.
    """
    np.random.seed(seed)
    # Generate a random walk for the midprice, then simulate bid-ask spread around it.
    midprice = np.cumsum(0.00005 + np.random.randn(num_ticks) * 0.0001) + 1.2
    spread = np.random.uniform(0.00005, 0.0001, size=num_ticks)
    bid = midprice - spread / 2
    ask = midprice + spread / 2
    volume = np.random.randint(1, 100, size=num_ticks)  # Simple volume simulation

    data = {"bid": bid, "ask": ask, "volume": volume}
    return pd.DataFrame(data)


def plot_processed_data(processed_df: pd.DataFrame) -> None:
    """
    Visualize the processed forex tick data from the Tick-Whisper Analysis.

    This function creates a three-panel plot:
      1. The original midprice vs. time.
      2. The denoised midprice (smooth trend) vs. time.
      3. The high-frequency component vs. time.

    Expected observations:
      - The **midprice plot** shows the raw price evolution calculated as (bid+ask)/2.
      - The **denoised midprice plot** should smooth out the noise present in the midprice,
        revealing the underlying trend. It may lag slightly due to the reconstruction process.
      - The **high-frequency component plot** represents the difference between the raw midprice
        and the denoised signal. You should observe that:
            • It fluctuates around zero, indicating that the denoising effectively extracted
              the trend.
            • Spikes or bursts in this component may indicate moments of abrupt liquidity changes
              or microstructure events (the “heartbeat” of liquidity).
    """
    plt.figure(figsize=(14, 10))

    # Plot 1: Original Midprice
    plt.subplot(3, 1, 1)
    plt.plot(
        processed_df.index, processed_df["midprice"], color="blue", label="Midprice"
    )
    plt.title("Forex Tick Data - Midprice")
    plt.xlabel("Tick")
    plt.ylabel("Price")
    plt.legend()

    # Plot 2: Denoised Midprice (Smooth Trend)
    plt.subplot(3, 1, 2)
    plt.plot(
        processed_df.index,
        processed_df["denoised_midprice"],
        color="green",
        label="Denoised Midprice",
    )
    plt.title("Denoised Midprice (Trend)")
    plt.xlabel("Tick")
    plt.ylabel("Price")
    plt.legend()

    # Plot 3: High-Frequency Component
    plt.subplot(3, 1, 3)
    plt.plot(
        processed_df.index,
        processed_df["high_freq_component"],
        color="red",
        label="High-Frequency Component",
    )
    plt.title("High-Frequency Component (Liquidity Pulses)")
    plt.xlabel("Tick")
    plt.ylabel("Residual")
    plt.legend()

    plt.tight_layout()
    plt.show()


# Example usage (assuming you've already processed your DataFrame):
# processed_df = process_forex_tick_data(your_tick_dataframe)
# plot_processed_data(processed_df)


def main() -> None:
    """
    Main function to demonstrate the Tick-Whisper Analysis on forex tick data from a DataFrame.

    The function simulates forex tick data, processes it to compute the midprice,
    and extracts the denoised midprice and high-frequency liquidity component.
    """
    start, end = "02012024", "03012024"
    # Simulate a DataFrame with forex tick data
    tick_df = pd.read_csv(
        "data/raw/EUR_USD/EUR_USD_M1_01012024_31122024.csv",
        index_col=0,
        parse_dates=True,
    )
    # tick_df = simulate_forex_tick_dataframe(num_ticks=1024)
    filtered_df = tick_df.loc[
        datetime.strptime(start, "%d%m%Y") : datetime.strptime(end, "%d%m%Y")
    ]
    processed_df = process_forex_tick_data(filtered_df, wavelet="db4", level=3)

    # Print a sample of the DataFrame showing the computed columns
    print("Processed Forex Tick Data (first 10 rows):")
    print(
        processed_df.head(10)[
            ["bid", "ask", "midprice", "denoised_midprice", "high_freq_component"]
        ]
    )
    plot_processed_data(processed_df)


if __name__ == "__main__":
    main()

# %%
