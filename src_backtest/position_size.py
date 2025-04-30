import math


class PositionSize:
    def __init__(self, allowed_leverage: float, max_risk: float):
        self.allowed_leverage = allowed_leverage
        self.max_risk = max_risk

    def position_size(
        self, entry_price: float, stop_loss: float, current_capital: float
    ):
        """
        Calculates position size based on max risk of current capital.
        Return None if not a viable trade.
        """

        if any(price is None for price in (entry_price, stop_loss)):
            return None

        # print(entry_price, stop_loss)
        diff = abs(entry_price - stop_loss)
        max_risk_value = self.max_risk * current_capital

        # print(entry_price, stop_loss)
        # print(f"max risk, diff: {max_risk_value, diff}")
        position_size = max_risk_value / diff

        # print(f"position_size: {position_size}")
        position_size = math.floor(position_size)

        trade_value = position_size * entry_price

        if self._viable_position_size(trade_value, current_capital):
            return position_size

    def _viable_position_size(self, trade_value: float, current_capital: float):
        if trade_value > self.allowed_leverage * current_capital:
            return False
        return True
