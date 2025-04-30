import pandas as pd

from src_backtest.condition.entry._base import Condition


# TODO: register all conditions as we will use strings to also inject conditions
# into StrategyHandler as it is cleaner.
class ALMAHMA(Condition):
    def _required_indicators(self) -> list[str]:
        """hma and alma used for this condition."""
        return ["hma", "alma"]

    def evaluate(self, data: pd.DataFrame):
        bid_hma_col, bid_alma_col = (
            self._col(data, ["bid", "hma"]),
            self._col(data, ["bid", "alma"]),
        )
        ask_hma_col, ask_alma_col = (
            self._col(data, ["ask", "hma"]),
            self._col(data, ["ask", "alma"]),
        )

        for _, indicator in self.indicators.items():
            data = indicator(data)

        hma_bid_change_percentage = self._change_percentage(
            data[bid_hma_col], self.hma_spike_window
        )
        hma_ask_change_percentage = self._change_percentage(
            data[ask_hma_col], self.hma_spike_window
        )
        alma_bid_change_percentage = self._change_percentage(
            data[bid_alma_col], self.alma_spike_window
        ).shift(-self.hma_look_forward_alma_window)
        alma_ask_change_percentage = self._change_percentage(
            data[ask_alma_col], self.alma_spike_window
        ).shift(-self.hma_look_forward_alma_window)

        # get masks that look forward `look_forward` bars
        ask_up, _ = self._look_forward_spike_masks(
            series=hma_ask_change_percentage.combine(
                alma_ask_change_percentage, min
            ),  # both must spike → take the min of the two
            window=self.hma_look_forward_alma_window,
            threshold=self.hma_spike_percentage,
        )
        bid_down, _ = self._look_forward_spike_masks(
            series=hma_bid_change_percentage.combine(
                alma_bid_change_percentage, max
            ),  # both must spike down → take the max (most negative)
            window=self.hma_look_forward_alma_window,
            threshold=self.hma_spike_percentage,
        )

        new_signals = pd.Series(0, index=data.index, dtype=int)
        new_signals.loc[ask_up] = 1
        new_signals.loc[bid_down] = -1

        keep_mask = self._keep_mask(data, new_signals)
        data[self._entry_col()] = new_signals.where(keep_mask, 0)

        return data

    def _look_forward_spike_masks(
        self, series: pd.Series, window: int, threshold: float
    ) -> tuple[pd.Series, pd.Series]:
        """
        Returns two masks indicating where:
        - up_spike:  series increases by > threshold at ANY point in the next `window` bars
        - down_spike: series decreases by > threshold (i.e. < -threshold) at ANY point in the next `window` bars

        :param series:    Percentage-change series (aligned to original data)
        :param window:    Number of bars to look forward
        :param threshold: Absolute percentage threshold for a spike
        :return:          (up_spike_mask, down_spike_mask)
        """
        # roll over the next `window` values, anchored at current row
        # shift by -(window-1) so that the “window” spans current bar → next window-1 bars
        rolled_max = series.rolling(window, min_periods=1).max().shift(-(window - 1))
        rolled_min = series.rolling(window, min_periods=1).min().shift(-(window - 1))

        up_spike = rolled_max > threshold
        down_spike = rolled_min < -threshold

        return up_spike, down_spike

    def _change_percentage(self, series: pd.Series, window: int):
        """
        The measure of what counts as a spike within the given window in self.configurations.

        """
        window_change_percentage = (series.shift(-window) - series) / series.shift(
            -window
        )

        return window_change_percentage
