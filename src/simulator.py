from datetime import datetime
from typing import Any, Optional

import pandas as pd

from .account import Account
from .configurator import Configurator
from .position_size import PositionSize
from .printer import Printer
from .tracker import Trade, TradeTracker


class Simulator:
    def __init__(self, configurator: Configurator, verbose: int = 0):
        self.configurator = configurator
        self.all_data = configurator.get_data()
        self.verbose = verbose
        self.printer = Printer(configurator, verbose)

        self.initial_capital = configurator.initial_capital
        self.position_size = PositionSize(
            allowed_leverage=configurator.allowed_leverage,
            max_risk=configurator.max_risk,
        )

        self.entry_strategy = configurator.entry_strategy
        self.exit_strategy = configurator.exit_strategy

        self.trade_tracker = TradeTracker()

    def simulate(self):
        for cur, data in self.all_data.items():
            self.printer.print_begin(cur)
            account = Account(capital=self.initial_capital)
            self.trade_tracker.reset()

            lowest_capital = self.initial_capital

            for pos, row in enumerate(data.itertuples(index=True), start=0):
                if account.bust():
                    break
                time = row.Index
                entry_signal = 0 if pd.isna(row.entry) else int(row.entry)

                # Only if valid entry then check for entry, sl, tp
                if self._entry_signal(entry_signal):
                    # print("ENTRY SIGNAL!!")
                    # print("valid entry signal")
                    entry_price, stop_loss, take_profit = self.get_entry_sl_tp(
                        data, pos, entry_signal
                    )

                    position_size = self.position_size.position_size(
                        entry_price, stop_loss, account.capital
                    )

                    # print(
                    #     f"entry, sl, tp, position_size: {entry_price, stop_loss, take_profit, position_size}"
                    # )
                    # If valid trade: valid entry, sl, tp and position_size
                    # print(entry_price, stop_loss, take_profit, position_size)
                    if self.valid_trade(
                        entry_price, stop_loss, take_profit, position_size
                    ):
                        # print("valid trade!")
                        # If valid then enter trade
                        self.enter_trade(
                            account=account,
                            entry_time=time,
                            entry_price=entry_price,
                            stop_loss=stop_loss,
                            take_profit=take_profit,
                            direction=entry_signal,
                            position_size=position_size,
                        )

                # If there are open trades, check if they need to be closed
                if self.trade_tracker.open_trades:
                    to_close_trades = {}
                    for trade_id, trade in self.trade_tracker.open_trades.items():
                        exit_price = self._exit_price(row, trade.direction)

                        if self._should_exit(exit_price, trade):
                            self.close_trades(
                                account=account,
                                to_close_trades=to_close_trades,
                                trade_id=trade_id,
                                exit_price=exit_price,
                                exit_time=time,
                                trade=trade,
                            )
                            lowest_capital = min(lowest_capital, account.capital)
                self.trade_tracker.update()  # modify dictionaries storing trade data only after

            start_str, end_str = self.configurator.start_end_str
            pnl_percent = (
                (account.capital - self.initial_capital) / self.initial_capital
            ) * 100
            max_drawdown = (
                (self.initial_capital - lowest_capital) / self.initial_capital
            ) * 100

            self.printer.store_end_simulation(
                start=start_str,
                end=end_str,
                currency=cur,
                initial_capital=self.initial_capital,
                final_capital=account.capital,
                pnl_percent=pnl_percent,
                max_drawdown=max_drawdown,
                trade_tracker=self.trade_tracker,
            )
        self.printer.print_end_simulation_summary()

    def enter_trade(
        self,
        account: Account,
        entry_time: datetime,
        entry_price: float,
        stop_loss: float,
        take_profit: float,
        direction: Optional[int],
        position_size: int,
    ):
        trade = Trade(
            entry_time, entry_price, stop_loss, take_profit, direction, position_size
        )
        assert stop_loss < entry_price, (
            f"Stop loss not less than entry_price! Clearly incorrect! Sl, Entry: {stop_loss, entry_price}"
        )
        trade_open_str = self.trade_tracker.open_trade(trade)
        self.printer.print_trade_open(trade_open_str, account.capital)

    def close_trades(
        self,
        account: Account,
        to_close_trades: dict[int, Any],
        trade_id: int,
        exit_price: float,
        exit_time: datetime,
        trade: Trade,
    ):
        status, pnl = self._status_and_pnl(exit_price, trade)

        to_close_trades[trade_id] = {
            "exit_time": exit_time,
            "exit_price": exit_price,
            "pnl": pnl,
            "status": status,
        }
        for trade_id, close_trade_kwargs in to_close_trades.items():
            trade_close_str = self.trade_tracker.close_trade(
                trade_id, **close_trade_kwargs
            )
            account.add_deduct_money(pnl)
            self.printer.print_trade_close(
                trade_close_str=trade_close_str, current_capital=account.capital
            )

    def _status_and_pnl(self, exit_price: float, trade: Trade):
        if trade.direction == 1:
            if exit_price <= trade.stop_loss:
                status = "stopped"
            elif exit_price >= trade.take_profit:
                status = "target_hit"
            else:
                status = "closed"
            pnl = (exit_price - trade.entry_price) * trade.position_size

        if trade.direction == -1:
            if exit_price >= trade.stop_loss:
                status = "stopped"
            elif exit_price <= trade.take_profit:
                status = "target_hit"
            else:
                status = "closed"
            pnl = (trade.entry_price - exit_price) * trade.position_size

        return status, pnl

    def _calculate_pnl(self, exit_price: float, trade: Trade):
        return abs(exit_price - trade.entry_price) * trade.position_size

    def _should_exit(self, exit_price: float, trade: Trade):
        if trade.direction == 1:
            if exit_price <= trade.stop_loss or exit_price >= trade.take_profit:
                return True
        if trade.direction == -1:
            if exit_price >= trade.stop_loss or exit_price <= trade.take_profit:
                return True

        return False

    def _entry_signal(self, entry_signal: int):
        if entry_signal in [1, -1]:
            return True
        return False

    def valid_trade(
        self,
        entry_price: float,
        stop_loss: float,
        take_profit: float,
        position_size: int,
    ):
        # print(f"Valid Trade?: {entry_price, stop_loss, take_profit, position_size}")
        if any(
            price is None for price in (entry_price, stop_loss, take_profit)
        ):  # no sl or tp
            return False

        if position_size is None:  # cannot take position
            return False

        if self.trade_tracker.open_trades:
            return False

        return True

    def get_entry_sl_tp(self, data: pd.DataFrame, current_idx: int, entry_signal: int):
        # print(data.info())
        row = data.iloc[current_idx]

        entry_price = self._entry_price(row, entry_signal)
        # print(f"ENTRY PRICE: {entry_price}")

        if entry_price is None:
            raise ValueError(
                f"Entry price should not be None. Likely incorrect signal: {entry_signal}"
            )

        stop_loss = self.exit_strategy.stop_loss(
            entry_price, current_idx, data, entry_signal
        )
        take_profit = self.exit_strategy.take_profit(
            entry_price, stop_loss, entry_signal
        )

        return entry_price, stop_loss, take_profit

    def _entry_price(self, row: pd.Series, entry_signal: int):
        if entry_signal == 1:
            return row.ask
        if entry_signal == -1:
            return row.bid
        return None

    def _exit_price(self, row: pd.Series, current_position_type: int):
        """
        If it is a long position, then return bid price.
        If it is short position, return ask price
        """
        if current_position_type == 1:
            return row.bid

        if current_position_type == -1:
            return row.ask

        raise ValueError(
            f"Invalid position type: {current_position_type} or row: {row}"
        )


# %%
