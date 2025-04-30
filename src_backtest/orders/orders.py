from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Callable, Optional

import pandas as pd

from src_backtest.utils.decorators import update_status


@dataclass(repr=False)
class Order(ABC):
    """
    Order class meant to hold order.
    Main functions:
        - allow for replacing old take profits with new ones
        - allow for trailing stop loss
        - allow for trailing tp

        - allow for extraction of relevant order information
            - entry price
            - position size
            - stop loss
            - take profits
            - current pnl
            - trade value
    """

    symbol: str
    position_size: float
    entry_price: float
    entry_time: pd.DatetimeIndex
    stop_loss: float
    leverage: float | int
    take_profits: dict[int, float] = field(default_factory=dict)

    status: str = "open"  # "closed", "stopped", "target_hit"
    close_price: Optional[float] = None
    close_time: Optional[pd.DatetimeIndex] = None

    def __repr__(self) -> str:
        cls_name = self.__class__.__name__
        return f"{cls_name}"

    @abstractmethod
    def pnl(self) -> float:
        """Return profit based on order type."""
        raise NotImplementedError

    @abstractmethod
    def _hit_stoploss(self, current_price: float) -> bool:
        """Criteria for hitting stoploss."""
        raise NotImplementedError

    @abstractmethod
    def _hit_take_profit(current_price: float) -> bool:
        """Criteria for hitting take profit."""
        raise NotImplementedError

    def hit_stoploss(self, current_price: float) -> bool:
        """Check if stop loss has been hit and update status."""
        if self._hit_stoploss(current_price):
            self.status = "hit_stoploss"
            return True
        return False

    def hit_take_profit(self, current_price: float) -> bool:
        """Check if take profit has been hit and update status."""
        if self._hit_take_profit(current_price):
            self.status = "hit_target"
            return True
        return False

    def position_type(self) -> int:
        """Return the order type based on obj name"""
        if "sell" in self.__class__.__name__.lower():
            return -1
        if "buy" in self.__class__.__name__.lower():
            return 1
        raise ValueError("Order type not recognized.")

    def notional(self) -> float:
        """Return the face value of the order."""
        return abs(self.position_size * self.entry_price)

    @update_status("closed")
    def close_order(self, close_price: float, close_time: pd.DatetimeIndex) -> None:
        """Close the order."""
        self.close_price = close_price
        self.close_time = close_time

    def adjust_sl_tp(
        self, new_stoploss: float, new_take_profits: dict[int, float]
    ) -> None:
        """
        Adjusts the stop loss and take profit of the order.
        """
        self.stop_loss = new_stoploss
        self.take_profits = new_take_profits

    def _update_stoploss_status(self, hit_sl_func: Callable):
        """
        Update the stop loss status of the order.
        """

        self.status = hit_sl_func(self)
        return self.status


@dataclass
class SellOrder(Order):
    def pnl(self) -> float:
        """Return sell order profit."""

        return (self.entry_price - self.close_price) * self.position_size

    def _hit_stoploss(self, current_price: float) -> bool:
        """Check if stop loss has been hit."""
        return current_price >= self.stop_loss

    def _hit_take_profit(self, current_price: float) -> bool:
        """Check if take profit has been hit."""
        return current_price <= self.take_profits[min(self.take_profits.keys())]


@dataclass
class BuyOrder(Order):
    def pnl(self) -> float:
        """Return sell order profit."""

        return (self.close_price - self.entry_price) * self.position_size

    def _hit_stoploss(self, current_price: float) -> bool:
        """Check if stop loss has been hit."""
        return current_price <= self.stop_loss

    def _hit_take_profit(self, current_price: float) -> bool:
        """Check if take profit has been hit."""
        return current_price >= self.take_profits[min(self.take_profits.keys())]
