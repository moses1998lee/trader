import os
from datetime import datetime

import pandas as pd
import requests

from src.utils import configs

CONFIGS = configs()
API = CONFIGS.oanda.api_key
BASE_URL = CONFIGS.oanda.oanda_base_url
HEADERS = {"Authorization": f"Bearer {API}", "Content-type": "application/json"}


def make_verbose_api_request(url, headers=None, params=None):
    """
    Make an API request and handle errors with verbose output.

    Args:
        url (str): The URL to make the API request to.
        params (dict, optional): A dictionary of parameters to send in the query string.

    Returns:
        dict: A dictionary containing the response data or a verbose error message.
    """
    try:
        response = requests.get(url, headers=headers, params=params)
        response.raise_for_status()  # Check for HTTP errors
    except requests.exceptions.HTTPError:
        # Attempt to parse error message from response body
        error_message = response.json().get("message", "No error message provided")
        raise ValueError(
            {
                "error": f"HTTP error occurred: {response.status_code} - {error_message}",
                "status_code": response.status_code,
            }
        )
    except requests.exceptions.RequestException as err:
        raise ValueError(
            {
                "error": f"Network or request error occurred: {err}",
                "status_code": None,
            }
        )
    else:
        return response.json()  # Return the successful response


def get_candles(
    instrument: str, start_date: datetime, granularity: str, price_type: str = "M"
):
    """Requests from oanda's api candles. Returns mid price point as a list of dicts."""
    url = f"v3/instruments/{instrument}/candles"
    endpoint = os.path.join(BASE_URL, url)

    params = {
        "from": start_date.isoformat() + "Z",
        "count": 5000,
        "granularity": granularity,
        "price": price_type,
    }

    data = make_verbose_api_request(endpoint, HEADERS, params)
    candles = data.get("candles", [])
    # print(candles)
    if not candles:
        print("WARNING: No candle data present.")

    if price_type == "BA":
        historical_data = [
            {
                "time": candle["time"],
                "bid": candle["bid"]["c"],
                "ask": candle["ask"]["c"],
                "volume": candle["volume"],
            }
            for candle in candles
            if candle.get("complete", False)
        ]
    if price_type == "M":
        historical_data = [
            {
                "time": candle["time"],
                "open": candle["mid"]["o"],
                "high": candle["mid"]["h"],
                "low": candle["mid"]["l"],
                "close": candle["mid"]["c"],
                "volume": candle["volume"],
            }
            for candle in candles
            if candle.get("complete", False)
        ]

    return historical_data


def to_df(data: list[dict]):
    """Convert list of items (json) into dataframe."""
    columns = list(data[0].keys())
    df_dict = {col: [] for col in columns}
    for item in data:
        for col, val in item.items():
            df_dict[col].append(val)

    return pd.DataFrame(df_dict)


def str_to_dt(str: str):
    iso = str.replace("Z", "+00:00")
    dt = datetime.fromisoformat(iso).replace(tzinfo=None)
    return dt


def fetch_historical(
    instrument: str, granularity: str, price_type: str, start_str: str, end_str: str
):
    """Fetches historical data from oanda by requesting multiple times based on start and end date str.
    Returns a df for easy data manipulation"""
    dt_start = datetime.strptime(start_str, "%d%m%Y")
    dt_end = datetime.strptime(end_str, "%d%m%Y")
    instrument = instrument.upper()
    granularity = granularity.upper()

    current = dt_start
    all_data = []
    prev_candles = []
    while current < dt_end:
        candles = get_candles(
            instrument,
            start_date=current,
            granularity=granularity,
            price_type=price_type,
        )
        if prev_candles == candles:
            break
        prev_candles = candles

        current = str_to_dt(candles[-1]["time"])
        all_data.extend(candles)

    if not all_data:
        raise ValueError("NO DATA")

    df = to_df(all_data)
    df["time"] = df["time"].apply(str_to_dt)
    correct_range_df = df.loc[df["time"] <= dt_end]

    # Convert to datetime
    correct_range_df.loc[:, "time"] = pd.to_datetime(correct_range_df["time"])
    correct_range_df.set_index("time", inplace=True)
    return correct_range_df
