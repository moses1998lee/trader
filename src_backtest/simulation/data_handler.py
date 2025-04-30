"""
combines multiple assets (their dataframes) and processes data
based on the respective strategy selected.
"""

from dataclasses import dataclass

@dataclass
class Data:

class DataHandler:
    """
    Align indexes by time, implement respective entry strategy on each symbol.
        - means: different symbols may have different strategies

    Combines all asset datasets into one master dataframe with columns indicating
    '<symbol>_bid' and '<symbol>_ask' prices, as well as'<symbol>_entry_signal'.

    '<symbol>_entry_signal' should contain 4 different signals that are strings:
        - 'sell_seeking_<id>': looking to sell
        - 'buy_seeking_<id>': looking to buy

        - 'sell_confirmed_<id>': confirmed sell
        - 'buy_confirmed_<id>': confirmed buy
        they should also contain a 'signal_id' so that the 'confirmation' signal 
        is tagged to a specific 'seeking' signal since multiple 'seeking' signals can be 
        generated before any confirmation.
    """



