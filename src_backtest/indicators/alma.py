from dataclasses import dataclass

import numpy as np
import pandas as pd

from src_backtest.indicators._base import Indicator
from src_backtest.indicators._registry import _IndicatorRegistry
from src_backtest.utils.decorators import register_indicator


@register_indicator(_IndicatorRegistry)
@dataclass
class ALMA(Indicator):
    """
    Arnaud Legoux Moving Average (ALMA) indicator.

    ALMA applies a Gaussian-weighted rolling window to smooth a time series
    with reduced lag compared to traditional moving averages.
    """

    def __call__(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Compute ALMA and append result columns to a copy of the input DataFrame.

        The weight for lag i is:
            w_i = exp(-((i - m)^2) / (2 * sigma^2)) / sum_j exp(-((j - m)^2) / (2 * sigma^2))
        where
            m = (period - 1) * offset,
            i, j ∈ [0, period-1].

        Args:
            df (pd.DataFrame): Input data containing numeric columns.

        Returns:
            pd.DataFrame: A new DataFrame with one ALMA column per selected input column.
        """
        if self.cols is None:
            cols = df.select_dtypes(include="number").columns
        else:
            cols = self.cols

        m = (self.period - 1) * self.offset
        idx = np.arange(self.period)
        raw_w = np.exp(-((idx - m) ** 2) / (2 * self.sigma**2))
        w = raw_w / raw_w.sum()

        out = df.copy()
        for col in cols:
            name = self.suffix or f"{col}_alma_{self.period}"
            vals = df[col].rolling(self.period).apply(lambda x: np.dot(x, w), raw=True)
            out[name] = vals

        return out
