# %%
import os
import re

import pandas as pd

from src.preprocessing import compute_lag_features
from src.utils import configs

CONFIGS = configs()
RAW_DATAFOLDER = CONFIGS["folder_paths"]["raw"]
PREPROCESSED_DATAFOLDER = CONFIGS["folder_paths"]["preprocessed"]

INCLUDE_INSTRUMENTS = "ALL"  # "ALL" or list of instruments
PREPROCESSING_CONFIGS = CONFIGS["preprocessing"]


def main():
    for instrument in os.listdir(RAW_DATAFOLDER):
        directory_path = os.path.join(RAW_DATAFOLDER, instrument)
        print(instrument)

        if INCLUDE_INSTRUMENTS != "ALL":
            if instrument not in INCLUDE_INSTRUMENTS:
                continue  # Skip if not included instrument

        for file in os.listdir(directory_path):
            print(f"Preprocessing for {file}")
            file_path = os.path.join(directory_path, file)

            df = pd.read_csv(file_path)
            # df = calculate_emas(df, PREPROCESSING_CONFIGS["ema"], "close")
            # df = calculate_smas(df, PREPROCESSING_CONFIGS["sma"], "close")
            # df = calculate_rsis(df, PREPROCESSING_CONFIGS["rsi"], "close")
            df["time"] = pd.to_datetime(df["time"])
            df.set_index("time", inplace=True)

            for n, lag_features in enumerate(PREPROCESSING_CONFIGS["lag_features"]):
                for lag in range(1, PREPROCESSING_CONFIGS["n_lags"] + 1):
                    first, second = re.split("-", lag_features)
                    feature = compute_lag_features(df, first, second, lag)
                    df[f"feature_{n}_{lag}"] = feature

            # df = df.dropna(axis=0)
            # df = df.drop(columns=["open", "high", "low"])
            # df = df.pct_change()

            print("Preprocessing completed!")

            preprocessed_instrument_folder = os.path.join(
                PREPROCESSED_DATAFOLDER, instrument
            )
            os.makedirs(preprocessed_instrument_folder, exist_ok=True)

            preprocessed_file_path = os.path.join(preprocessed_instrument_folder, file)
            df.to_csv(preprocessed_file_path, index=False)
            print(f"Preprocessed file saved to: {preprocessed_file_path}")


if __name__ == "__main__":
    main()
