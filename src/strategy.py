from abc import ABC, abstractmethod

import pandas as pd

BID_COL = "bid"
ASK_COL = "ask"
SIGNAL_COL = "signal"
VOLUME_COL = "volume"


class BaseStrategy(ABC):
    """All data assumes time, bid and ask for columns."""

    def __init__(self):
        pass

    @abstractmethod
    def transform_data(self, data):
        pass

    @abstractmethod
    def stoploss(self, data):
        pass


class MeanReversion(BaseStrategy):
    def __init__(
        self,
        mid_reversion_window: int,
        vol_reversion_window: int,
        mid_reversion_std: float,
        vol_reversion_std: float,
        vol_spike_window: int,
        stoploss_percent_std: float,
    ):
        super().__init__()
        self.mid_reversion_window = mid_reversion_window
        self.vol_reversion_window = vol_reversion_window
        self.mid_reversion_std = mid_reversion_std
        self.vol_reversion_std = vol_reversion_std
        self.vol_spike_window = vol_spike_window
        self.stoploss_percent_std = stoploss_percent_std

    def __repr__(self):
        return "Strategy=MeanReversion"

    def transform_data(self, data: pd.DataFrame):
        self.generate_lb_ub_data(data)
        self.generate_vol_data(data)

        self.generate_buy_signal(data)
        self.generate_sell_signal(data)
        data.dropna(inplace=True)
        return data

    def generate_lb_ub_data(self, data: pd.DataFrame):
        """
        Requires:
        1. Moving Average - calculated on mid price
        2. Standard deviation
        3. Upper Bound
        4. Lower Bound
        """
        data["mid"] = (data[BID_COL] + data[ASK_COL]) / 2
        data["ma"] = data["mid"].rolling(window=self.mid_reversion_window).mean()
        data["std"] = data["mid"].rolling(window=self.mid_reversion_window).std()
        data["ub"] = data["ma"] + (data["std"] * self.mid_reversion_std)
        data["lb"] = data["ma"] - (data["std"] * self.mid_reversion_std)

    def generate_buy_signal(self, data: pd.DataFrame):
        lb_ub_condition = data["mid"] < data["lb"]
        vol_condition = data["vol_condition_met"] == 1
        data.loc[lb_ub_condition & vol_condition, SIGNAL_COL] = 1

    def generate_sell_signal(self, data: pd.DataFrame):
        lb_ub_condition = data["mid"] > data["ub"]
        vol_condition = data["vol_condition_met"] == 1
        data.loc[lb_ub_condition & vol_condition, SIGNAL_COL] = -1

    def generate_vol_data(self, data: pd.DataFrame):
        data["vol_ma"] = (
            data[VOLUME_COL].rolling(window=self.vol_reversion_window).mean()
        )
        data["vol_std"] = (
            data[VOLUME_COL].rolling(window=self.vol_reversion_window).std()
        )
        data["vol_bound"] = data["vol_ma"] + (data["vol_std"] * self.vol_reversion_std)
        spike = data[VOLUME_COL] > data["vol_bound"]  # boolean Series

        data["vol_condition_met"] = (
            spike.rolling(window=self.vol_spike_window, min_periods=1)
            .max()  # 1 if any True, 0 otherwise
            .astype(int)
        )

    def stoploss(self, data):
        """
        SL for MeanReversion:
            std below or std above current price for buy and sell orders respectively.
        """
        # print(f"SL STD, STD: {self.stoploss_percent_std}, {data['std']}")
        if data[SIGNAL_COL] == 0:
            raise ValueError(
                "No stoploss to compute when there is no buy or sell signal!"
            )
        if data[SIGNAL_COL] == 1:
            stoploss = data[ASK_COL] - (self.stoploss_percent_std * data["std"])

        if data[SIGNAL_COL] == -1:
            stoploss = data[BID_COL] + (self.stoploss_percent_std * data["std"])

        return stoploss
