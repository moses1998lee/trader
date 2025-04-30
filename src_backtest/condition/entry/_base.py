from abc import ABC, abstractmethod

import pandas as pd

from src_backtest.indicators._registry import _IndicatorRegistry
from src_backtest.utils.configs import configs

CONFIGS = configs()


class Condition(ABC):
    def __init__(self):
        indicators = self._required_indicators()
        self.indicators = {name: _IndicatorRegistry[name] for name in indicators}
        self.configurations = self._initialize_configurations()
        print(f"{self.__name__} using indicators: {list(self.indicators.keys())}")

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

    @abstractmethod
    def evaluate(self, data: pd.DataFrame) -> pd.Series:
        """
        Evaluate the condition on the given data.
        """
        raise NotImplementedError

    def _initialize_configurations(self):
        if self.__name__ not in CONFIGS.conditions:
            raise KeyError(f"{self.__name__} could not be found in .yaml config files.")

        return CONFIGS.conditions[self.__name__]

    def indicator_col_map(self, columns):
        """Return a map of the name of indicator to the column name in the dataframe."""
        return {name: col for name in self.indicators for col in columns if name in col}

    def _col(data: pd.DataFrame, *args):
        """A list of strings, will return the column name that has this list of strings."""

        matches = []
        for col in data.columns:
            for string in args:
                if string not in col:
                    break

            matches.append(col)

        if len(matches) == 0:
            raise KeyError(f"None of the columns contains all strings in {args}!")

        if len(matches) >= 1:
            raise ValueError("Should not have more than 1 match.")

        return matches[0]

    def _entry_col(self):
        return "entry"

    def _keep_mask(self, data: pd.DataFrame, new_signals: pd.Series):
        existing = data[self._entry_col()].fillna(0).astype(int)
        keep_mask = existing.eq(0) | existing.eq(new_signals)
        return keep_mask
