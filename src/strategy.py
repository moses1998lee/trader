from abc import ABC, abstractmethod

import pandas as pd

BID_COL = "bid"
ASK_COL = "ask"
SIGNAL_COL = "signal"


class BaseStrategy(ABC):
    """All data assumes time, bid and ask for columns."""

    def __init__(self):
        pass

    @abstractmethod
    def transform_data(self, data):
        pass


class MeanReversion(BaseStrategy):
    def __init__(self, window: int, std_threshold: float):
        super().__init__()
        self.window = window
        self.std_threshold = std_threshold

    def __repr__(self):
        return "Strategy=MeanReversion"

    def transform_data(self, data: pd.DataFrame):
        """
        Requires:
        1. Moving Average - calculated on mid price
        2. Standard deviation
        3. Upper Bound
        4. Lower Bound
        """
        data["mid"] = (data[BID_COL] + data[ASK_COL]) / 2
        data["ma"] = data["mid"].rolling(window=self.window).mean()
        data["std"] = data["mid"].rolling(window=self.window).std()
        data["ub"] = data["ma"] + (data["std"] * self.std_threshold)
        data["lb"] = data["ma"] - (data["std"] * self.std_threshold)

        data[SIGNAL_COL] = 0
        data.loc[data["mid"] > data["ub"], SIGNAL_COL] = -1
        data.loc[data["mid"] < data["lb"], SIGNAL_COL] = 1

        return data
