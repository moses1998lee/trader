# %%
import pandas as pd

from src_backtest.account import Account
from src_backtest.condition.entry.alma_hma import ALMAHMA
from src_backtest.orders.order_manager import Order, OrderManager
from src_backtest.risk.risk_manager import RiskManager

# order_adjustment = OrderAdjustmentHandler(stoploss_adjustment_strategy=StoplossAdjustment, take_profit_adjustment_strategy=TakeProfitAdjustment)
account = Account(10000)
risk_manager = RiskManager(account, 50)
condition = ALMAHMA()
order_manager = OrderManager()


data = pd.read_csv("data/raw/oanda/eur_usd/eur_usd_h1_01012024_31122024.csv")
processed = condition.evaluate(data)
# TODO: save data

for pos, row in processed.iterrows():
    if row["entry"] == 1:
        entry_price = row["ask"]
    if row["entry"] == -1:
        entry_price = row["bid"]

    Order(
        symbol="eurusd",
    )
