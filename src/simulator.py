import math
from dataclasses import dataclass
from datetime import datetime

import pandas as pd

from .strategy import BaseStrategy

BID_COL = "bid"
ASK_COL = "ask"
SIGNAL_COL = "signal"
LEVERAGE = 50  # 50x leverage


@dataclass
class Account:
    capital: float

    def add_deduct_money(self, amount: float):
        """Adds or deduct money depending on if the amount is positive of negative"""
        self.capital += amount

    def bust(self):
        return self.capital <= 0


class Simulator:
    def __init__(
        self,
        account: Account,
        max_risk: float,
        risk_reward: tuple[float, float],
        df: pd.DataFrame,
        strategy: BaseStrategy,
    ):
        self.account = account
        # currently using initial capital to calculate risk reward amount
        self.initial_capital = account.capital
        self.leverage = LEVERAGE
        self.max_risk = max_risk
        self.risk_reward = risk_reward
        self.strategy = strategy
        self.data = strategy.transform_data(df)

        self.has_position = False
        self.stoploss = None
        self.take_profit = None

        self.position_value = None
        self.current_position_type = None
        self.current_position_price = None
        self.current_position_size = None

        self.n_positions = 0

    def simulate(self, start_str: str | None = None, end_str: str | None = None):
        start = None if start_str is None else datetime.strptime(start_str, "%d%m%Y")
        end = None if end_str is None else datetime.strptime(end_str, "%d%m%Y")

        self.data = self.data.loc[start:end]

        for time, data in self.data.iterrows():
            if self.account.bust():
                break
            if not self.has_position and self.should_purchase(data):
                self.open_position(time, data)
                # print("1")
                continue

            if self.has_position:
                # print("2")
                if self.should_close(data):
                    self.close_position(time, data)

        percentage_gained = round(
            (
                (self.account.capital - self.initial_capital)
                * 100
                / self.initial_capital
            ),
            2,
        )
        gained_str = (
            f"+{percentage_gained} %"
            if percentage_gained > 0
            else f"{percentage_gained} %"
        )
        return_str = f"Capital: {self.initial_capital} → {self.account.capital:.2f}, {gained_str}\nPositions Taken: {self.n_positions}"

        print("__________________________________________")
        print("\nSimulation Ended!")
        # print(return_str)
        return return_str

    def should_purchase(self, data: pd.Series):
        """If has no position then return if signal column returns -1 (sell) or 1 (buy)."""
        if not self.has_position:
            return abs(data[SIGNAL_COL].item())  # True of 1 or -1 else False (0)
        return False

    def hit_stoploss(self, closing_price: float):
        if self.current_position_type == 1:
            return closing_price <= self.stoploss
        if self.current_position_type == -1:
            return closing_price >= self.stoploss

    def hit_take_profit(self, closing_price: float):
        if self.current_position_type == 1:
            return closing_price >= self.take_profit
        if self.current_position_type == -1:
            return closing_price <= self.take_profit

    def should_close(self, data: pd.Series):
        """
        Checks if the current stoploss or take profit is hit -> meaning we
        will need to close the order.
        """
        if self.has_position:
            closing_price = self.closing_price(data)
            # print(f"UP, LT, PT: {unrealised_profit, loss_threshold, profit_threshold}")
            # if self.current_position_type + data[SIGNAL_COL] == 0:
            # print(
            #     f"{self._position_str_map(data[SIGNAL_COL])} signal: ${self.current_position_price}, ${closing_price}"
            # )
            # return True

            if self.hit_stoploss(closing_price):
                # print(
                #     f"Stoploss signal: ${self.current_position_price}, ${closing_price}"
                # )
                return True
            if self.hit_take_profit(closing_price):
                # print(
                #     f"Takeprofit signal: ${self.current_position_price}, ${closing_price}"
                # )
                return True

        return False

    def can_open(self, position_value: float):
        if position_value / self.leverage > self.account.capital:
            return False
        return True

    def compute_position_size(self, stoploss: float, current_price: float):
        """
        diff = |current_price - stoploss|
        diff will be the amount loss if we bought 1 unit of the currency.
        position size = self.max_risk in dollar amt / diff for 1 unit of currency.
        """
        diff = abs(current_price - stoploss)
        max_risk_in_dollars = (
            self.risk_reward[0] * self.max_risk * self.initial_capital
        )  # We assume max risk does not scale with changing size of account
        # We assume stays constant with initial capital

        # print(current_price, stoploss)
        # print(diff)
        position_size = math.floor(max_risk_in_dollars / diff)

        return position_size

    def tp(self, data: pd.Series, current_price: float):
        """Computes take profit based on given current price based on signal to buy or sell."""
        diff = abs(current_price - self.stoploss)

        if data[SIGNAL_COL] == 0:
            raise ValueError("No take profit if no buy or sell signal!")

        if data[SIGNAL_COL] == 1:
            return current_price + (self.risk_reward[1] * diff)

        if data[SIGNAL_COL] == -1:
            return current_price - (self.risk_reward[1] * diff)

    def open_position(self, time: datetime, data: pd.Series):
        current_price = self.opening_price(data)
        self.stoploss = self.strategy.stoploss(data)
        self.take_profit = self.tp(data, current_price)

        position_size = self.compute_position_size(self.stoploss, current_price)
        position_value = current_price * position_size
        # print(f"Position Value: {position_value}")

        if self.can_open(position_value):
            # print(
            #     f"Stoploss, Takeprofit: ${float(self.stoploss)}, ${float(self.take_profit)}"
            # )
            self.has_position = True
            self.current_position_type = data[SIGNAL_COL]
            self.current_position_price = current_price
            self.current_position_size = position_size
            self.position_value = position_value
            print(
                f"{self._position_str_map(self.current_position_type)} Position Opened "
                f"(${self.account.capital:.2f}): {time} @ {self.current_position_price}"
            )

    def opening_price(self, data: pd.Series):
        """Checks opening price based on given signal.
        If buy then checks 'ASK' price, if sell then checks
        'BID' price."""
        if data[SIGNAL_COL] == 0:
            raise ValueError(
                "Not supposed to have opening price if no signal to make a purchase."
            )
        if data[SIGNAL_COL] == 1:
            return data[ASK_COL]
        if data[SIGNAL_COL] == -1:
            return data[BID_COL]

    def unrealised_profit(self, closing_price: float):
        if self.current_position_type == 1:
            return (
                closing_price - self.current_position_price
            ) * self.current_position_size
        if self.current_position_type == -1:
            return (
                self.current_position_price - closing_price
            ) * self.current_position_size

    def close_position(self, time: datetime, data: pd.Series):
        closing_price = self.closing_price(data)
        unrealised_profit = self.unrealised_profit(closing_price)

        profit_str = (
            f"+ ${unrealised_profit:.2f}"
            if unrealised_profit >= 0
            else f"- ${abs(unrealised_profit):.2f}"
        )
        print(
            f"{self._position_str_map(self.current_position_type)} ({self.current_position_size}) Position Closed "
            f"(${self.account.capital:.2f} {profit_str}): {time} @ {closing_price}"
        )
        self.has_position = False
        self.current_position_type = None
        self.current_position_price = None

        self.account.add_deduct_money(unrealised_profit)
        self.n_positions += 1

    def closing_price(self, data: pd.Series):
        """Checks if current position is a buy or sell and retrieve the relevant
        closing price.

        for e.g. if we are holding a buy position (self.current_position_type = 1),
        then we will retrieve the 'BID' price since we will sell our order if we close it
        and vice versa for a sell position."""
        if not self.has_position:
            raise ValueError(
                "Please double check implementation. There is no position open"
                "currently, so there should be no closing price."
            )

        if self.current_position_type == 1:
            return data[BID_COL].item()

        if self.current_position_type == -1:
            return data[ASK_COL].item()

    def _position_str_map(self, signal: int):
        if signal == 0:
            return "NULL????"
        if signal == 1:
            return "BUY"
        if signal == -1:
            return "SELL"


# %%
