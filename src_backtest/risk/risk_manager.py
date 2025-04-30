from src_backtest.account import Account
from src_backtest.orders.order_manager import Order, OrderManager


class RiskManager:
    """Manages position size and overall ensuring the account doesn't blow up."""

    def __init__(self, account: Account, leverage: float | int):
        self.account = account

        # TODO: to take into consideration multiple assets running and check trade values across all positions

    def still_within_margin(self, order_manager: OrderManager, new_order: Order):
        all_open_orders = order_manager.all_open_orders()

        order_margins = [order.notional() / order.leverage for order in all_open_orders]
        consumed_margin = sum(order_margins)

        new_consumed_margin = (
            consumed_margin + new_order.notional() / new_order.leverage
        )

        return new_consumed_margin <= self.account.capital
