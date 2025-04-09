import os

from src.oanda_utils import fetch_historical
from src.utils import configs

CONFIGS = configs()
RAW_DATAFOLDER = CONFIGS["folder_paths"]["raw"]

INSTRUMENTS = ["EUR_USD", "USD_CAD", "USD_JPY", "EUR_JPY"]
GRANULARITY = "M1"
PRICE_TYPE = "BA"  # bid-ask, change to "M" for mid-price
START_STR = "01012010"
END_STR = "31122024"


def main():
    for instrument in INSTRUMENTS:
        print(
            f"FETCHING DATA FOR {instrument}, {GRANULARITY} for period {START_STR}-{END_STR}"
        )
        hist_df = fetch_historical(
            instrument, GRANULARITY, PRICE_TYPE, START_STR, END_STR
        )
        instrument_folder = os.path.join(RAW_DATAFOLDER, instrument)
        os.makedirs(instrument_folder, exist_ok=True)

        file_name = f"{instrument}_{GRANULARITY}_{START_STR}_{END_STR}.csv"
        full_path = os.path.join(instrument_folder, file_name)
        hist_df.to_csv(full_path)
        print(f"File saved to {full_path}")


if __name__ == "__main__":
    main()
