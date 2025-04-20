import os
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Optional

import pandas as pd
import pandas_datareader.data as web

from .entry_strat import BaseEntry
from .exit_strat import BaseExit
from .utils import configs, extract_base_string, extract_data_by_date, load_csv

CONFIGS = configs()
FILE_EXT = ".csv"  # Can currently only support .csv file types


@dataclass
class Configurator:
    data_provider: str  # currently only works for 'dukascopy' or 'oanda'
    granularity: str  # 'tick' or minute or hour 'M1', 'H1'
    currencies: list[str]  # list of currencies to simulate
    start_end_str: tuple[str, str]  # start and end str of simulation

    initial_capital: float
    allowed_leverage: float
    max_risk: float
    entry_strategy: BaseEntry
    exit_strategy: BaseExit
    risk_free_rate_per_min: Optional[float] = None
    data: Optional[dict[str, pd.DataFrame]] = None

    def __post_init__(self):
        if self.data is None:
            self.data = self.get_data()

        else:
            raise Warning(
                "Beware as you have loaded your own data directly which should not be done!"
            )

        if self.risk_free_rate_per_min is None:
            self.risk_free_rate_per_min = self.get_risk_free_rate_per_min()

    def get_risk_free_rate_per_min(self):
        start = datetime.strptime(self.start_end_str[0], "%d%m%Y")
        end = datetime.strptime(self.start_end_str[1], "%d%m%Y")

        # Fetch 3-Month Treasury Bill rates from FRED
        risk_free_data = web.DataReader("DTB3", "fred", start, end)
        risk_free_data = risk_free_data / 100
        avg_rate = risk_free_data.mean().item()
        rf_per_minute = avg_rate / (252 * 390)  # 252 days * 390 mins/day

        return rf_per_minute

    def get_data(self) -> list[pd.DataFrame]:
        """
        Gets a list of dataframes of the currencies to be simulated with the specific entry strategy.
        If the data has not been preprocessed, it will be done so within this function else it will
        be loaded.
        """
        f_configs = CONFIGS.folder_paths  # folder configs
        entry_strategy = str(self.entry_strategy)

        all_preprocessed = {}
        for cur in self.currencies:
            raw_file_name = f"{cur}_{self.granularity}_{self.start_end_str[0]}_{self.start_end_str[1]}{FILE_EXT}"
            split_text = os.path.splitext(raw_file_name)
            strat_specific_path = split_text[0] + f"_{entry_strategy}" + split_text[1]

            preprocessed_file_path = os.path.join(
                f_configs.data_root,
                f_configs.preprocessed_folder,
                self.data_provider,
                cur,
                strat_specific_path,
            )

            if not os.path.exists(preprocessed_file_path):
                preprocessed_data = self._generate_preprocessed(
                    cur, f_configs, raw_file_name, preprocessed_file_path
                )
            else:
                preprocessed_data = load_csv(preprocessed_file_path)

            all_preprocessed[cur] = preprocessed_data

        return all_preprocessed

    def _generate_preprocessed(
        self,
        currency: str,
        folder_configs: dict[str, Any],
        raw_file_name: str,
        preprocessed_file_path: str,
    ):
        raw_folder_path = os.path.join(
            folder_configs.data_root,
            folder_configs.raw_folder,
            self.data_provider,
            currency,
        )

        raw_file_path = os.path.join(raw_folder_path, raw_file_name)
        # If raw file path dont exist, then we will need to generate
        # We will need to generate from a dataset that contains the range
        # if not throw error
        if not os.path.exists(raw_file_path):
            valid_raw_file_path = self._check_valid_date_range(
                raw_folder_path, raw_file_name
            )
            print(
                "Retrieving data that covers the 'start_str' and 'end_str' range specified."
            )
            data = extract_data_by_date(
                start_str=self.start_end_str[0],
                end_str=self.start_end_str[1],
                data_path=valid_raw_file_path,
                save_dir=raw_folder_path,
            )
        else:
            # if raw file exists alr just load
            data = load_csv(raw_file_path)

        print(f"Preprocessing data with {str(self.entry_strategy)}...")
        # print(data)
        preprocessed_data = self.entry_strategy.generate_entry_signal(data)

        os.makedirs(os.path.dirname(preprocessed_file_path), exist_ok=True)
        preprocessed_data.to_csv(preprocessed_file_path)
        print(
            f"Generated {str(self.entry_strategy)} preprocessed file saved to: {preprocessed_file_path}!"
        )

        return preprocessed_data

    def _check_valid_date_range(self, raw_folder_path: str, raw_file_name: str):
        base_str, start_str, end_str, _ = extract_base_string(raw_file_name)

        # print(base_str, start_str, end_str)
        valid_file = []
        for file in os.listdir(raw_folder_path):
            # print(file)
            if base_str in file:
                _, check_start_str, check_end_str, _ = extract_base_string(file)

                start_date = datetime.strptime(start_str, "%d%m%Y")
                end_date = datetime.strptime(end_str, "%d%m%Y")
                check_start_date, check_end_date = (
                    datetime.strptime(check_start_str, "%d%m%Y"),
                    datetime.strptime(check_end_str, "%d%m%Y"),
                )
                # print(start_date, end_date)
                # print(check_start_date, check_end_date)
                if start_date >= check_start_date and end_date <= check_end_date:
                    valid_file.append(file)
        if not valid_file:
            raise ValueError("The range of dates provided are not valid. Please check!")

        valid_file_path = os.path.join(raw_folder_path, valid_file[0])
        return valid_file_path  # return the first file name that covers the range of the inputted start and end date
