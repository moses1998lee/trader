"""
File includes classes that generate ENTRY signals: buy or sell signals (1,-1 respectively).
These would then be used by the 'Simulator' class to execute trade orders.

Each strategy may require a specific set of preprocessing steps, hence
a .transform_data() is necessary.
"""

from abc import ABC, abstractmethod
from typing import Any

import pandas as pd

from .errors import MissingNewCol
from .utils import configs

CONFIGS = configs()


class BaseEntry(ABC):
    def __init__(
        self,
        strategy_configs: dict[str, Any],
        strategy_new_cols: dict[str, Any],
        attributes: dict[str, Any] | None = None,
    ):
        """
        A dict of attribute_name:attribute_type needs to be defined and fed into BaseEntry baseclass
        initialization to ensure that the required attributes are correctly set.
        """
        if attributes is None:
            raise AttributeError(
                "Strategy incorrectly coded. Please ensure self.attributes "
                "has been defined as a dictionary containing the attribute name "
                "as keys and its respective expected types as the values."
            )
        self.class_name = None
        self._to_set_attributes = attributes
        self.strategy_configs = strategy_configs
        self.strategy_new_cols = strategy_new_cols
        self.configs = self.initialize_configs(strategy_configs)

    def __repr__(self):
        if self.class_name is None:
            raise Warning(
                "'self.class_name' not defined within strategy "
                "class, hence return default name."
            )
            return "DefaultBaseEntry"
        strategy_configs_str = "".join(
            [f"_{k}_{v}" for k, v in self.strategy_configs.items()]
        )
        return f"{self.class_name}{strategy_configs_str}"

    def initialize_configs(self, strategy_configs: dict[str, Any]):
        """
        Checks that the configurations contain the right attributes to be set and set them.

        Returns AttributeError if the attribute to be set in the configuration dict if the attribute
        is not present in the configuration dict or if it is missing some attributes to be set.
        """
        to_set_attributes = set(self._to_set_attributes.keys())
        provided_attributes = set(strategy_configs.keys())

        missing = to_set_attributes - provided_attributes
        if missing:
            raise AttributeError(
                f"Missing required attribute(s) in config: {', '.join(missing)}.\n"
                f"Expected all of: {', '.join(to_set_attributes)}"
            )

        extra = provided_attributes - to_set_attributes
        if extra:
            raise AttributeError(
                f"Received extra attribute(s) that cannot be set: {', '.join(extra)}.\n"
                f"Expected only: {', '.join(to_set_attributes)}"
            )

        for key, value in strategy_configs.items():
            expected_type = self._to_set_attributes[key]
            if not isinstance(value, expected_type):
                raise ValueError(
                    f"Expected '{expected_type} but got {type(value)} "
                    f"for value of '{key}'"
                )
            setattr(self, key, value)
        print(f"Configurations set: \n{strategy_configs}")

    def entry_price(self, data: pd.DataFrame) -> tuple[float, float]:
        """
        Returns both the 'buy' and 'sell' price by retrieving the
        'bid' and 'ask' price respectively.
        """
        buy_price = data[CONFIGS.generated.ask]
        sell_price = data[CONFIGS.generated.bid]

        return buy_price, sell_price

    def check_new_col_names(self, key: str):
        col_name = self.strategy_new_cols.get(key, None)
        if col_name is None:
            raise MissingNewCol(
                f"Missing column name '{key}', please ensure you contain "
                f"a name mapping with '{key}': 'col_name' within the 'strategy_new_cols' "
                "input."
            )

    @abstractmethod
    def generate_entry_signal(self, data: pd.DataFrame):
        """
        Function that generates entry signals, either a buy
        or a sell (1, -1 respectively) within a 'entry' column in the
        pd.DataFrame.

        If the strategy involves a buy and sell, it should the 'entry'
        column should include a list that contain [1, -1] to signify
        buy and sell.
        """
        raise NotImplementedError


class SpreadReversionEntry(BaseEntry):
    def __init__(
        self,
        spread_reversion_configs: dict[str, Any],
        strategy_new_cols: dict[str, Any],
    ):
        """
        A dict of attribute_name:attribute_type needs to be defined and fed into BaseEntry baseclass
        initialization to ensure that the required attributes are correctly set.
        """
        # These ATTRIBUTES will be converted to class attributes
        ATTRIBUTES = {
            "spread_reversion_window": int,
            "spread_std_threshold_scalar": float,
        }

        super().__init__(
            spread_reversion_configs, strategy_new_cols, ATTRIBUTES
        )  # Includes initialization of all configurations set as None above

        self.class_name = "spread"

    def generate_entry_signal(self, data: pd.DataFrame):
        data = self._compute_spread(data)
        spread_ub, spread_lb = self._compute_spread_ub_lb(data)

        volatile = data["spread"] > spread_ub
        not_volatile = data["spread"] < spread_lb  # Might use in the future

        data.loc[volatile, CONFIGS.data_names.generated.entry] = 1  # Buy if volatile
        data.loc[not_volatile, CONFIGS.data_names.generated.entry] = (
            0  # if not volatile, dont do anything
        )

        return data

    def _compute_spread(self, data: pd.DataFrame):
        """
        Function checks for the presence of the 'spread' column name required and then
        computes the spread and stores it to this col name.
        """
        self.check_new_col_names("spread")
        spread_name = self.strategy_new_cols["spread"]

        data[spread_name] = abs(
            data[CONFIGS.data_names.standard.columns.bid]
            - data[CONFIGS.data_names.standard.columns.ask]
        )
        return data

    def _compute_spread_ub_lb(self, data: pd.DataFrame):
        spread_ema = (
            data["spread"].ewm(span=self.spread_reversion_window, adjust=False).mean()
        )
        spread_std = data["spread"].rolling(window=self.spread_reversion_window).std()

        spread_ub = spread_ema + (self.spread_std_threshold_scalar * spread_std)
        spread_lb = spread_ema - (self.spread_std_threshold_scalar * spread_std)

        return spread_ub, spread_lb
