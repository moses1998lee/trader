# %%
import pandas as pd

from src.simulator import Account, Simulator
from src.strategy import MeanReversion

pd.set_option("display.max_rows", None)  # None means no limit
pd.set_option("display.max_columns", None)  # Display all columns

pd.set_option("display.width", None)  # Use None to automatically detect the width
pd.set_option("display.max_colwidth", None)  # Use None for unlimited column width


MAX_RISK = 0.01  # 1% of account
STD_THRESHOLD = 1  # 110% of std
RISK_REWARD = (1, 1)  # 1 MAX_RISK is to 3 REWARD
MEAN_REVERSION_WINDOW = 4
STOPLOSS_PERCENT_STD = 0.15
CURRENCIES = ["EUR_JPY", "EUR_USD", "USD_CAD", "USD_JPY"]
# CURRENCIES = ["EUR_USD"]

# start,end = None, None
start, end = "01012024", "31122024"


if __name__ == "__main__":
    return_strs = []

    for cur in CURRENCIES:
        df = pd.read_csv(
            f"data/raw/{cur}/{cur}_H1_01012010_31032025.csv",
            index_col=0,
            parse_dates=True,
        )
        acc = Account(10000)
        strat = MeanReversion(
            window=MEAN_REVERSION_WINDOW,
            std_threshold=STD_THRESHOLD,
            stoploss_percent_std=STOPLOSS_PERCENT_STD,
        )
        sim = Simulator(
            account=acc,
            max_risk=MAX_RISK,
            risk_reward=RISK_REWARD,
            strategy=strat,
            df=df,
        )

        print(f"Running simulation for {cur}...")
        # e.g. start,end string: 01012024 (ddmmyyyy)
        str = sim.simulate(
            start, end
        )  # start, end indicates start and end date the simulation is runned on the dataset
        return_strs.append(str)

    print(f"start: {start}, end: {end}")
    for cur, s in zip(CURRENCIES, return_strs):
        print(f"{cur}\n{s}")

# %%
