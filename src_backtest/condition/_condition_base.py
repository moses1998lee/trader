from abc import ABC, abstractmethod

import pandas as pd

from src_backtest.indicators._registry import _IndicatorRegistry
from src_backtest.utils.configs import configs

CONFIGS = configs()


class Condition(ABC):
    def __init__(self):
        indicators = self._required_indicators()
        self.indicators = {
            name.lower(): _IndicatorRegistry[name]() for name in indicators
        }
        self.configurations = self._initialize_configurations()
        self._set_attributes()
        print(
            f"{self.__class__.__name__} using indicators: {list(self.indicators.keys())}"
        )

    @abstractmethod
    def _evaluate(self, data: pd.DataFrame) -> pd.Series:
        """
        Evaluate the condition on the given data that already has indicator
        data evaluated on the dataset.
        """
        raise NotImplementedError

    @abstractmethod
    def _required_indicators(self) -> list[str]:
        """
        Return a list of string of the indicators required. This is to instantiate the respective
        indicators within the class within using them as inputs.
        """
        raise NotImplementedError(
            "._required_indicators() need to be defined so that indicators "
            "for the condition can be instantiated."
        )

    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        """
        Generate buy and sell signals based on the indicators and condition used.
        """
        data = self.apply_indicators(data)

        print(f"Processing with Condition {self.__class__.__name__}")
        signal_series = self._evaluate(data)
        keep_mask = self._keep_mask(data, signal_series)
        data[self._entry_col()] = signal_series.where(keep_mask, 0)
        data = data.dropna(axis=0)
        print("Signal Data Generated!")

        return data

    def _initialize_configurations(self):
        config_class_map = {
            s.replace("_", "").lower(): s for s in CONFIGS.conditions.keys()
        }

        class_key = self.__class__.__name__.lower()
        if class_key not in config_class_map:
            raise KeyError(f"{class_key} could not be found in .yaml config files.")

        return CONFIGS.conditions[config_class_map[class_key]]

    def _set_attributes(self):
        for attr, val in self.configurations.items():
            setattr(self, attr, val)
            print(f"{self.__class__.__name__} Condition '{attr}' set to '{val}'!")

    def apply_indicators(self, data: pd.DataFrame):
        print("Applying indicators...")
        if "time" not in data.columns:
            raise KeyError("'time' column is not present. Dataset incorrect!")

        dfs = [
            indicator(data.copy()).set_index("time")
            for indicator in self.indicators.values()
        ]

        merged_df = dfs[0]
        for df in dfs[1:]:
            new_cols = [
                c for c in df.columns if c not in merged_df.columns or c == "time"
            ]

            merged_df = pd.merge(merged_df, df[new_cols], on="time", how="outer")

        return merged_df

    def _col(self, data: pd.DataFrame, strings: list[str] | str):
        """A list of strings, will return the column name that has this list of strings."""

        matches = [col for col in data.columns if all(s in col for s in strings)]

        if not matches:
            raise KeyError(f"None of the columns contains all strings in {strings}!")
        if len(matches) > 1:
            raise ValueError("Should not have more than 1 match.")

        return matches[0]

    def _entry_col(self):
        return "entry"

    def _keep_mask(self, data: pd.DataFrame, new_signals: pd.Series):
        if self._entry_col() not in data.columns:
            data[self._entry_col()] = 0

        existing = data[self._entry_col()].fillna(0).astype(int)
        keep_mask = existing.eq(0) | existing.eq(new_signals)
        return keep_mask
