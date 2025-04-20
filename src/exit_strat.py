"""
File includes classes that generate EXIT signals (1 if you should exit the position should there exist any).
These would then be used by the 'Simulator' class to execute trade orders.
"""

from abc import ABC, abstractmethod

import pandas as pd

from .utils import configs

CONFIGS = configs()


class BaseExit(ABC):
    def __init__(self, risk_to_reward_str: str):
        self.mapper = {
            1: CONFIGS.data_names.standard.columns.bid,  # buy order: look for bid price
            -1: CONFIGS.data_names.standard.columns.ask,  # sell order: look for ask price
        }
        self.risk_to_reward = self.risk_to_reward_converter(risk_to_reward_str)

    @abstractmethod
    def stop_loss(self):
        raise NotImplementedError

    @abstractmethod
    def take_profit(self):
        raise NotImplementedError

    def risk_to_reward_converter(self, risk_to_reward_str: str):
        risk_str, reward_str = risk_to_reward_str.split(",")
        risk_to_reward = (float(risk_str), float(reward_str))

        return risk_to_reward


class MinSLRatioTP(BaseExit):
    """
    This class sets the stop loss to be the minimum of a window of the entry signal.
    The take profit is set to be a predefined ratio from the entry price.
    """

    def __init__(self, min_sl_window_lookback: int, risk_to_reward_str: str):
        super().__init__(
            risk_to_reward_str
        )  # Initializes self.risk_to_reward & self.mapper
        self.min_sl_window_lookback = min_sl_window_lookback

    def stop_loss(
        self,
        entry_price: float,
        current_idx: int,
        data: pd.DataFrame,
        signal: int | list[int],
    ):
        """Given a window lookback, the minimum / maximum given whether it is a buy(bid price) or sell(ask price) order
        respectively."""
        if self.min_sl_window_lookback > current_idx:
            # print(
            #     f"not large enough window: {current_idx} vs {self.min_sl_window_lookback}"
            # )
            return None  # Invalid stop_loss. Not enough data to determine

        # print(f"SL SIGNAL: {self._valid_signal(signal)}")
        if self._valid_signal(signal):
            # print("VALID SIGNAL!!!")
            window_data = data.iloc[
                current_idx - self.min_sl_window_lookback : current_idx - 1
            ]

            position_window_data = self._get_position_data(window_data, signal)
            stop_loss = position_window_data.min()
            if (
                abs(entry_price - stop_loss) / entry_price <= 0.001
            ):  # if less than 1%, not counted
                return None

            return stop_loss

    def take_profit(
        self,
        entry_price: float,
        stop_loss: float,
        signal: int | list[int],
    ):
        if stop_loss is None:
            return None

        diff = abs(entry_price - stop_loss)
        tp_amt = diff * (self.risk_to_reward[1] / self.risk_to_reward[0])

        def tp_price(position_type: int, entry_price: float):
            if position_type == 1:
                return entry_price + tp_amt
            if position_type == -1:
                return entry_price - tp_amt

        if self._valid_signal(signal):
            if isinstance(signal, list):
                return [tp_price(position, entry_price) for position in signal]

            return tp_price(signal, entry_price)

    def _valid_signal(self, signal: int | list[int]):
        if isinstance(signal, list):
            if len(signal) >= 0:
                return True

        if signal == 1 or signal == -1:
            return True

        return False

    def _get_position_data(self, data: pd.DataFrame, signal: int | list[int]):
        if isinstance(signal, list):
            return {position: data[self.mapper[position]] for position in signal}

        return data[self.mapper[signal]]
