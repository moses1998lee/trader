"""
Contains the logic for adjusting stoploss and take profit orders.

New stoploss and take profit strategies are defined in the respective classes
StoplossAdjustment and TakeProfitAdjustment as functions.

The strategy is then called upon when instantiating the class witht he respective string name
that is mapped to the respective function.

This way, we can easily iterate through all strategies if we decide to test multiple
order adjustment strategies at once.
"""

from typing import Callable

from src_backtest.orders.orders import Order


class StoplossAdjustment:
    """
    More stoploss adjustment strategies can be added as a function with the expected
    input and output to be the same as the function call .new_stoploss().

    When StoplossADjustment is instantiated, the stoploss_strategy is mapped to the function
    to the function that is called when .new_stoploss() is called.
    """

    def __init__(self, order: Order, stoploss_strategy: str):
        self.order = order

        # Map strategy names to the concrete methods
        strategies: dict[str, Callable[[float], float]] = {
            "trailing": self._trailing_stop,
            # add more strategies here…
        }

        try:
            self._strategy_fn = strategies[stoploss_strategy]
        except KeyError:
            raise ValueError(f"Unknown stoploss strategy: {stoploss_strategy!r}")

    def new_stoploss(self, current_price: float) -> float:
        """Delegate to the chosen strategy function."""
        return self._strategy_fn(current_price)

    def _trailing_stop(self, current_price: float) -> float:
        """Move stop loss to 50% above stoploss price and 1:1"""
        _ = current_price  # unused for this strategy
        entry_to_sl_dist = self.order.entry_price - self.order.stop_loss

        # position_type will account for whether the dist should be added or subtracted
        new_stoploss = self.order.stop_loss + (
            self.order.position_type() * entry_to_sl_dist / 2
        )

        return new_stoploss


class TakeProfitAdjustment:
    def __init__(self, order: Order, take_profit_strategy: str):
        """
        More take profit adjustment strategies can be added as a function with the expected
        input and output to be the same as the function call .new_take_profit().

        When TakeProfitAdjustment is instantiated, the take_profit_strategy is mapped to the function
        """
        self.order = order

        # Map strategy names to the concrete methods
        strategies: dict[str, Callable[[float], float]] = {
            "basic_shift": self._basic_shift,
            # add more strategies here…
        }

        try:
            self._strategy_fn = strategies[take_profit_strategy]
        except KeyError:
            raise ValueError(f"Unknown take profit strategy: {take_profit_strategy!r}")

    def new_take_profit(self, current_price: float) -> float:
        """Delegate to the chosen strategy function."""
        return self._strategy_fn(current_price)

    def _basic_shift(self, current_price: float) -> float:
        """
        Shift all take profits by 100% of entry price to stoploss price in 'positive' direction.
        if 'sell', then shift tp lower.
        if 'buy' then shift tp higher.
        """
        _ = current_price
        entry_to_sl_dist = self.order.entry_price - self.order.stop_loss
        new_last_tp = {
            max(self.order.take_profits.keys()) + 1: self.order.take_profits[-1]
            + entry_to_sl_dist
        }

        new_take_profits = {
            # shift all the other tp_units up by one
            **{
                tp_unit + 1: tp
                for tp_unit, tp in self.order.take_profits.items()
                if tp_unit != 1
            },
            **{new_last_tp},  # add one more tp at the end
        }

        return new_take_profits
