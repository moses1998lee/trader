from typing import Callable

import pandas as pd

from src_backtest.condition.entry._base import Condition


# TODO: register all conditions as we will use strings to also inject conditions
# into StrategyHandler as it is cleaner.
class ALMAHMA(Condition):
    def _required_indicators(self) -> list[str]:
        """hma and alma used for this condition."""
        return ["hma", "alma"]

    def _evaluate(self, data: pd.DataFrame):
        bid_hma_col, bid_alma_col = (
            self._col(data, ["bid", "hma"]),
            self._col(data, ["bid", "alma"]),
        )
        ask_hma_col, ask_alma_col = (
            self._col(data, ["ask", "hma"]),
            self._col(data, ["ask", "alma"]),
        )

        # normalise by previous range which is the pct change from the previous 2 candles
        hma_bid_change_percentage = data[bid_hma_col].pct_change() / abs(
            data[bid_hma_col].diff().shift(1)
        )
        hma_ask_change_percentage = data[ask_hma_col].pct_change() / abs(
            data[ask_hma_col].diff().shift(1)
        )
        alma_bid_change_percentage = data[bid_alma_col].pct_change() / abs(
            data[bid_alma_col].diff().shift(1)
        )
        alma_ask_change_percentage = data[ask_alma_col].pct_change() / abs(
            data[ask_alma_col].diff().shift(1)
        )

        # Checking for the spike over a specific window
        alma_bid_change = alma_bid_change_percentage.rolling(
            self.alma_spike_window
        ).min()
        alma_ask_change = alma_ask_change_percentage.rolling(
            self.alma_spike_window
        ).min()

        alma_bid_change = alma_bid_change_percentage.rolling(
            self.alma_spike_window
        ).min()
        alma_ask_change = alma_ask_change_percentage.rolling(
            self.alma_spike_window
        ).max()

        hma_bid_change = hma_bid_change_percentage.rolling(self.hma_spike_window).min()
        hma_ask_change = hma_ask_change_percentage.rolling(self.hma_spike_window).max()

        alma_bid_signal = alma_bid_change >= self.alma_spike_threshold
        hma_bid_signal = hma_bid_change >= self.hma_spike_threshold
        print(f"hma alma bid: {hma_bid_signal.sum(), alma_bid_signal.sum()}")
        sell_signal = alma_bid_signal & hma_bid_signal

        alma_ask_signal = alma_ask_change >= self.alma_spike_threshold
        hma_ask_signal = hma_ask_change >= self.hma_spike_threshold
        print(f"hma alma ask: {hma_ask_signal.sum(), alma_ask_signal.sum()}")
        buy_signal = alma_ask_signal & hma_ask_signal

        print(f"sell: {sell_signal.sum()}")
        print(f"buy: {buy_signal.sum()}")
        print(f"total: {len(hma_bid_change)}")

        signal_series = pd.Series(0, index=data.index, dtype=int)
        signal_series.loc[buy_signal] = 1
        signal_series.loc[sell_signal] = -1

        return signal_series

    def np_combine(
        self,
        series_a: pd.Series,
        series_b: pd.Series,
        new_series_name: str,
        combine_with_np_func: Callable,
    ):
        # assume series1, series2 share the same index
        combined_series = pd.Series(
            combine_with_np_func(series_a.values, series_b.values),
            index=series_a.index,
            name=new_series_name,
        )
        return combined_series

    def _change_percentage(self, series: pd.Series, window: int):
        """
        The measure of what counts as a spike within the given window in self.configurations.

        """
        window_change_percentage = (series.shift(-window) - series) / series.shift(
            -window
        )

        return window_change_percentage
