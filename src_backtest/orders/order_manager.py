from dataclasses import dataclass
from typing import Optional

import pandas as pd

from src_backtest.orders.order_adjustment import (
    StoplossAdjustment,
    TakeProfitAdjustment,
)
from src_backtest.orders.orders import Order


class OrderManager:
    """
    Manages order creation, modification, and cancellation.
    """

    # TODO: add printer into function to track prints
    def __init__(self):
        self.order_adjustment: OrderAdjustmentHandler

        self.open_orders: dict[int, Order] = {}
        self.closed_orders: dict[int, Order] = {}
        self.order_id_counter = 0

    def update_orders(self, current_price: float, current_time: pd.DatetimeIndex):
        closed = []
        for order_id, order in self.open_orders.items():
            if order.hit_stoploss(current_price):
                order.close_order(current_price, current_time)
                closed.append(order_id)

            # adjust sl & tp if hit target
            if order.hit_take_profit(current_price):
                self.order_adjustment.adjust()  # if no strategies, .adjust() doesn't change sl or tp

                if self.order_adjustment.no_strategies():
                    order.close_order(current_price, current_time)
                    closed.append(order_id)

        for order_id in closed:
            order_id, order = self.open_orders.pop(order_id)
            self.closed_orders[order_id] = order

    def create_order(self, order: Order) -> None:
        """
        Create a new order and add it to the open orders.
        """
        self.open_orders[self.order_id_counter] = order
        self.order_id_counter += 1

    def all_open_orders(self):
        return list(self.open_orders.values())


@dataclass
class OrderAdjustmentHandler:
    """
    Takes in Order and adjusts the order according to specified strategy defined in function
    """

    order: Order
    stoploss_adjustment_strategy: Optional[StoplossAdjustment] = None
    take_profit_adjustment_strategy: Optional[TakeProfitAdjustment] = None

    def adjust(
        self,
    ) -> None:
        """
        sl_strategy and tp_strategy will always act on Order object
        to determine the new stop loss and take profit values.
        """
        # Instantiate with original first
        new_stoploss = self.order.stop_loss
        new_take_profits = self.order.take_profits

        if self.stoploss_adjustment_strategy is not None:
            new_stoploss = self.stoploss_adjustment_strategy.new_stoploss()

        if self.take_profit_adjustment_strategy is not None:
            new_take_profits = self.take_profit_adjustment_strategy.new_take_profit()

        self.order.adjust_sl_tp(
            new_stoploss=new_stoploss, new_take_profits=new_take_profits
        )

    def no_strategies(self) -> bool:
        """
        Check if there no strategies are set for adjustment.
        This would indicate that we would just 'close' the order
        if we hit the stoploss or take profit.
        """
        return (
            self.stoploss_adjustment_strategy is None
            and self.take_profit_adjustment_strategy is None
        )
