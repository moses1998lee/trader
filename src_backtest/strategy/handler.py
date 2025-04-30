"""
A strategy handles the Indicators class and Condition class to create a strategy.

It handles the interaction of these classes with the data.

Strategy applies the indicators on the data to generate new columns
that are then used by the Condition class to generate signals.
"""

import pandas as pd

from src_backtest.condition.entry._base import Condition


class StrategyHandler:
    def __init__(self, data: pd.DataFrame, conditions: list[Condition]):
        self.data = data
        self.conditions = conditions

    def apply_conditions(self):
        """
        Applies all the conditions within self.conditions. Only when the data passes
        all conditions, will a 'buy' or 'sell' signal be given within the dataframe.
        """
        # After .evaluate(), self.data should contain a 'entry' column indicating a
        # 'buy'(1) or 'sell'(-1) action or no action (0)
        for condition in self.conditions:
            condition.evaluate(self.data)
