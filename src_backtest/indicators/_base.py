from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Optional

import pandas as pd

from src_backtest.utils.configs import configs

all_configs = configs()


@dataclass
class Indicator(ABC):
    cols: Optional[list[str]] = None
    suffix: Optional[str] = None

    def __post_init__(self):
        """
        Initialize the indicator configurations from conf/indicators.yaml.
        """
        indicator_configs = self._initialize_configs()
        self._set_indicator_attributes(indicator_configs)

    @abstractmethod
    def __call__(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Compute indicator values and return a DataFrame with the results.

        The new columns with indicator values should be named:
        <original_col_name>_<indicator_name>_<suffix_related_to_indicator>
        """
        raise NotImplementedError("Subclasses should implement this method.")

    def _initialize_configs(self) -> dict[str, Any]:
        """
        Initialize the indicator configurations from conf/indicators.yaml.
        """
        return all_configs["indicators"][self.__class__.__name__.lower()]

    def _set_indicator_attributes(self, indicator_configs: dict[str, Any]) -> None:
        """
        Set the attributes of the indicator class.
        """
        for attr, value in indicator_configs.items():
            setattr(self, attr, value)
            print(f"{self.__class__.__name__} attribute {attr} set to {value}!")
