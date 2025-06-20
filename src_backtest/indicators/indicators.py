from dataclasses import dataclass

import numpy as np
import pandas as pd

from src_backtest.indicators._base import Indicator
from src_backtest.indicators._registry import register_indicator


@register_indicator
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
            m = (window - 1) * offset,
            i, j ∈ [0, window-1].

        Args:
            df (pd.DataFrame): Input data containing numeric columns.

        Returns:
            pd.DataFrame: A new DataFrame with one ALMA column per selected input column.
        """
        if self.cols is None:
            cols = df.select_dtypes(include="number").columns
        else:
            cols = self.cols

        m = (self.window - 1) * self.offset
        idx = np.arange(self.window)
        raw_w = np.exp(-((idx - m) ** 2) / (2 * self.sigma**2))
        w = raw_w / raw_w.sum()

        out = df.copy()
        for col in cols:
            name = self.suffix or f"{col}_alma_{self.window}"
            vals = df[col].rolling(self.window).apply(lambda x: np.dot(x, w), raw=True)
            out[name] = vals

        return out


@register_indicator
@dataclass
class HMA(Indicator):
    """
    Hull Moving Average (HMA) indicator.

    HMA reduces lag by combining two weighted moving averages:
      1) A_t = WMA(P, n/2)
      2) B_t = WMA(P, n)
      3) D_t = 2·A_t − B_t
      4) HMA_t = WMA(D, √n)

    where
      WMA_t(X, m) = ∑_{i=0}^{m−1} (m−i)·X_{t−i}  /  [m(m+1)/2]
    """

    def __call__(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Compute HMA and append result columns to a copy of the input DataFrame.

        Args:
            df (pd.DataFrame): Input data containing numeric columns.

        Returns:
            pd.DataFrame: A new DataFrame with one HMA column per selected input column.
                          Columns are named "{col}_hma_{window}".
        """

        def _wma(x: np.ndarray) -> float:
            w = np.arange(1, len(x) + 1)
            return np.dot(x, w) / w.sum()

        half = self.window // 2
        root = int(np.sqrt(self.window))

        out = df.copy()
        # select numeric columns (or use self.cols if provided)
        cols = (
            df.select_dtypes(include="number").columns
            if self.cols is None
            else self.cols
        )

        for col in cols:
            series = df[col]
            # 1) WMA over n/2
            wma_half = series.rolling(half).apply(_wma, raw=True)
            # 2) WMA over n
            wma_full = series.rolling(self.window).apply(_wma, raw=True)
            # 3) D = 2·WMA(n/2) − WMA(n)
            diff = 2 * wma_half - wma_full
            # 4) HMA = WMA(D, √n)
            hma_vals = diff.rolling(root).apply(_wma, raw=True)

            name = self.suffix or f"{col}_hma_{self.window}"
            out[name] = hma_vals

        return out
