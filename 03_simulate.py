# %%
import pandas as pd

from src.simulator import Account, Simulator
from src.strategy import MeanReversion

pd.set_option("display.max_rows", None)  # None means no limit
pd.set_option("display.max_columns", None)  # Display all columns

pd.set_option("display.width", None)  # Use None to automatically detect the width
pd.set_option("display.max_colwidth", None)  # Use None for unlimited column width


MAX_RISK = 0.01  # 1% of account
RISK_REWARD = (1, 1)  # 1 MAX_RISK is to 3 REWARD

MID_REVERSION_WINDOW = (
    30  # window to calculate 'mid' price moving average and 'lb' and 'ub'
)
MID_STD_THRESHOLD = (
    1  # the percent of std from MA that counts as lower and upper bounds
)
# window to calculate the 'vol_bound' and used in tandem with checking spikes within VOL_SPIKE_WINDOW
VOL_REVERSION_WINDOW = 120
VOL_STD_THRESHOLD = (
    1  # threshold that is used from the 'vol_ma' that is used to determine a vol spike
)
VOL_SPIKE_WINDOW = 15  # Checks any 15 datapoints before if there is a spike

STOPLOSS_PERCENT_STD = (
    12  # how many times of std away from current price that is the stoploss
)
STARTING_CAPITAL = 1000

# ---------
# DATASETS
# ---------
CURRENCIES = ["EUR_JPY", "EUR_USD", "USD_CAD", "USD_JPY"]
# CURRENCIES = ["EUR_USD"]
GRANULARITY = "M1"
DATASET_DATE_STR = "01012010_31122024"

# start,end = None, None
start, end = "01012024", "31122024"


if __name__ == "__main__":
    return_strs = []

    for cur in CURRENCIES:
        df = pd.read_csv(
            f"data/raw/{cur}/{cur}_{GRANULARITY}_{DATASET_DATE_STR}.csv",
            index_col=0,
            parse_dates=True,
        )
        acc = Account(STARTING_CAPITAL)
        strat = MeanReversion(
            mid_reversion_window=MID_REVERSION_WINDOW,
            vol_reversion_window=VOL_REVERSION_WINDOW,
            mid_reversion_std=MID_STD_THRESHOLD,
            vol_reversion_std=VOL_STD_THRESHOLD,
            vol_spike_window=VOL_SPIKE_WINDOW,
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
        print(f"{cur}\n{str}")

        return_strs.append(str)

    print("\n\n############# SUMMARY #############")
    print(f"start: {start}, end: {end}")
    for cur, s in zip(CURRENCIES, return_strs):
        print(f"{cur}\n{s}")

# %%
