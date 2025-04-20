from dataclasses import dataclass
from datetime import datetime
from typing import Optional


@dataclass
class Trade:
    """When class is instantiated, a given set of attributes must be configured.
    Defaults to 'open' status because when class is instantiated, it assumes a trade
    is open."""

    entry_time: datetime
    entry_price: float
    stop_loss: float
    take_profit: float
    direction: int  # direction of trade -> 1: buy, -1: sell
    position_size: float  # position size of trade
    status: Optional[str] = "open"  # 'open', 'closed', 'stopped', 'target_hit'

    id: Optional[int] = None  # set by TradeTracker
    exit_time: Optional[datetime] = None
    exit_price: Optional[float] = None
    pnl: Optional[float] = None

    def close(self, exit_time: datetime, exit_price: float, status: str, pnl: float):
        self.exit_time = exit_time
        self.exit_price = exit_price
        self.status = status
        self.pnl = pnl


class TradeTracker:
    """
    Tracks and manages all trades in a backtest or live simulation.
    """

    open_trades: dict[int, Trade] = {}
    to_close: dict[int, Trade] = {}
    to_open: dict[int, Trade] = {}
    trade_history: dict[int, Trade] = {}
    direction_mapper = {1: "long", -1: "short"}

    n_wins = 0
    n_losses = 0
    id = 0

    def reset(self):
        """reset all trade trackers."""
        self.open_trades = {}
        self.to_close = {}
        self.to_open = {}
        self.trade_history = {}

        self.n_wins = 0
        self.n_losses = 0

    def update(self):
        for (
            trade_id,
            trade,
        ) in self.to_open.items():
            self.open_trades[trade_id] = trade

        for trade_id, trade in self.to_close.items():
            _ = self.open_trades.pop(trade_id)
            self.trade_history[trade_id] = trade

    def open_trade(self, trade: Trade) -> None:
        """
        Adds a new trade to the tracker.
        """
        trade.id = self.id
        self.to_open[self.id] = trade
        self.id += 1

        trade_open_str = (
            f"{trade.entry_time}, "
            f"id: {trade.id:<3} "
            f"{self.direction_mapper[trade.direction]:<10} ---- "
            f"E: {trade.entry_price:<8.5f} "
            f"SL: {trade.stop_loss:<8.5f} "
            f"TP: {trade.take_profit:<8.5f} "
            f"PositionSize: {trade.position_size:<8.2f}"
        )

        # trade_open_str = f"{trade.entry_time}, id: {trade.id:<5} ({self.direction_mapper[trade.direction]:<5}) E: {trade.entry_price:<8} SL: {trade.stop_loss:<8}: tp {trade.take_profit:<8}: position_size {trade.position_size:<8}"

        return trade_open_str

    def close_trade(
        self,
        trade_id: int,
        exit_time: datetime,
        exit_price: float,
        status: str,
        pnl: float,
    ):
        trade = self.open_trades[trade_id]
        trade.close(exit_time, exit_price, status, pnl)

        if trade.pnl > 0:
            self.n_wins += 1
        if trade.pnl <= 0:
            self.n_losses += 1

        self.to_close[trade_id] = trade

        trade_close_str = f"{trade.exit_time}, id: {trade.id:<3} {trade.status:<10} ---- PNL: $ {trade.pnl:.2f}"
        # trade_close_str = f"{exit_time}: position {status.upper()}: ${pnl}"
        return trade_close_str

    def total_pnl(self) -> float:
        """
        Computes the total profit or loss across all closed trades.
        """
        return sum(t.pnl for t in list(self.trades.values()) if t.pnl is not None)
