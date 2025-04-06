from dataclasses import dataclass
from datetime import datetime

import pandas as pd

from .strategy import BaseStrategy

BID_COL = "bid"
ASK_COL = "ask"
SIGNAL_COL = "signal"
LEVERAGE = 50  # 50x leverage
ONE_LOT = 100


@dataclass
class Account:
    capital: float

    def add_deduct_money(self, amount: float):
        """Adds or deduct money depending on if the amount is positive of negative"""
        self.capital += amount


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
        self.position_value = None
        self.current_position_type = None
        self.current_position_price = None

        self.n_positions = 0

    def simulate(self, start_str: str | None = None, end_str: str | None = None):
        start = None if start_str is None else datetime.strptime(start_str, "%d%m%Y")
        end = None if end_str is None else datetime.strptime(end_str, "%d%m%Y")

        self.data = self.data.loc[start:end]

        for time, data in self.data.iterrows():
            if not self.has_position and self.should_purchase(data):
                self.open_position(time, data)
                continue

            if self.has_position:
                if self.should_close(data):
                    # print(f"SHOULD CLOSE: {self.should_close(data)}")
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
        print("\nSimulation Ended!")
        print(f"Capital: {self.initial_capital} → {self.account.capital}, {gained_str}")

    def should_purchase(self, data: pd.Series):
        """If has no position then return if signal column returns -1 (sell) or 1 (buy)."""
        if not self.has_position:
            return abs(data[SIGNAL_COL].item())  # True of 1 or -1 else False (0)
        return False

    def should_close(self, data: pd.Series):
        """Checks if position is present, then checks if
        position type currently is opp from the signal given.

        e.g. if currently holding buy but signal says sell, then we expect
        price to fall, hence we should close the order anyway.

        Also checks if the current stoploss or take profit is hit -> meaning we
        will need to close the order."""
        profit_threshold = self.risk_reward[1] * self.max_risk * self.initial_capital
        loss_threshold = self.risk_reward[0] * self.max_risk * self.initial_capital

        if self.has_position:
            closing_price = self.closing_price(data)
            unrealised_profit = (closing_price - self.current_position_price) * ONE_LOT

            # print(f"UP, LT, PT: {unrealised_profit, loss_threshold, profit_threshold}")
            # if self.current_position_type + data[SIGNAL_COL] == 0:
            #     return True

            if unrealised_profit >= profit_threshold:  # If profit till takeprofit
                return True
            if unrealised_profit <= -loss_threshold:  # If lose till stoploss
                return True

        return False

    def can_open(self, position_value: float):
        if position_value / self.leverage > self.account.capital:
            return False
        return True

    def open_position(self, time: datetime, data: pd.Series):
        current_price = float(self.opening_price(data))
        position_value = current_price * ONE_LOT

        if self.can_open(position_value):
            self.has_position = True
            self.current_position_type = data[SIGNAL_COL]
            self.current_position_price = current_price
            self.position_value = position_value
            print(
                f"Position Opened (${self.account.capital:.2f}): {time} @ {self.current_position_price}"
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

    def close_position(self, time: datetime, data: pd.Series):
        closing_price = self.closing_price(data)
        unrealised_profit = (closing_price - self.current_position_price) * ONE_LOT

        self.has_position = False
        self.current_position_type = None
        self.current_position_price = None
        print(
            f"Position Closed (${self.account.capital:.2f}+{unrealised_profit:.2f}): {time} @ {closing_price}"
        )
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
