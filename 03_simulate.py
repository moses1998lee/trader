# %%
import pandas as pd

from src.simulator import Account, Simulator
from src.strategy import MeanReversion

pd.set_option("display.max_rows", None)  # None means no limit
pd.set_option("display.max_columns", None)  # Display all columns

pd.set_option("display.width", None)  # Use None to automatically detect the width
pd.set_option("display.max_colwidth", None)  # Use None for unlimited column width


MAX_RISK = 0.01  # 1% of account
STD_THRESHOLD = 1.1  # 110% of std
RISK_REWARD = (1, 1)  # 1 MAX_RISK is to 3 REWARD
MEAN_REVERSION_WINDOW = 8
start, end = "01012024", "31122024"

df = pd.read_csv(
    "data/raw/EUR_JPY/EUR_JPY_H1_01012024_01032025.csv", index_col=0, parse_dates=True
)
acc = Account(10000)
strat = MeanReversion(window=MEAN_REVERSION_WINDOW, std_threshold=STD_THRESHOLD)
sim = Simulator(
    account=acc, max_risk=MAX_RISK, risk_reward=RISK_REWARD, strategy=strat, df=df
)


if __name__ == "__main__":
    # e.g. start,end string: 01012024 (ddmmyyyy)
    sim.simulate(
        start, end
    )  # start, end indicates start and end date the simulation is runned on the dataset
