import os

from src.oanda_utils import fetch_historical
from src.utils import configs

CONFIGS = configs()

OANDA_RAW_DATAFOLDER = os.path.join(
    CONFIGS.folder_paths.data_root,
    CONFIGS.folder_paths.raw_folder,
    CONFIGS.folder_paths.sub_folders.oanda,
)

INSTRUMENTS = ["eur_usd", "usd_cad", "usd_jpy", "eur_jpy"]
GRANULARITY = "m1"
PRICE_TYPE = "BA"  # bid-ask, change to "M" for mid-price
START_STR = "01012024"
END_STR = "16042025"


def main():
    for instrument in INSTRUMENTS:
        print(
            f"FETCHING DATA FOR {instrument}, {GRANULARITY} for period {START_STR}-{END_STR}"
        )
        hist_df = fetch_historical(
            instrument, GRANULARITY, PRICE_TYPE, START_STR, END_STR
        )
        instrument_folder = os.path.join(OANDA_RAW_DATAFOLDER, instrument)
        os.makedirs(instrument_folder, exist_ok=True)

        file_name = f"{instrument}_{GRANULARITY}_{START_STR}_{END_STR}.csv"
        full_path = os.path.join(instrument_folder, file_name)
        hist_df.to_csv(full_path)
        print(f"File saved to {full_path}")


if __name__ == "__main__":
    main()
