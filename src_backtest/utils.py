"""
Includes any sort of miscellaneous utils not worthy of a new
.py file.
"""

import glob
import os
import re
from datetime import datetime

import omegaconf
import pandas as pd
from omegaconf import OmegaConf

RELATIVE_CONF_DIR = "conf"  # from project root


def configs() -> omegaconf.DictConfig:
    """
    Loads all .yaml file configurations in the /conf folder.
    """
    yaml_files = glob.glob(os.path.join(RELATIVE_CONF_DIR, "*.yaml"))
    configs = [OmegaConf.load(file) for file in yaml_files]
    merged_config = OmegaConf.merge(*configs)

    return merged_config


def standardize_data(data: pd.DataFrame):
    """Standardizes and checks that the columns contain the standard
    required columns as present in the config.yaml."""
    configurations = configs()
    data.index.name = configurations.data_names.standard.index.time
    data.columns = [col.lower() for col in data.columns]

    if not isinstance(data.index, pd.DatetimeIndex):
        raise AttributeError(
            f"Data should be a pd.DateTimeIndex but got {type(data.index)} instead"
        )

    cols_required = set(configurations.data_names.standard.columns.values())
    missing_cols = cols_required - set(data.columns)
    if missing_cols:
        raise AttributeError(f"Missing standard columns {missing_cols} in data!")

    return data


def extract_data_by_date(
    start_str: str | None, end_str: str | None, data_path: str, save_dir: str
):
    """
    Function extracts data by the given start_str and end_str date.
    If both are None, then the full dataset is returned.

    Function assumes data_path to be a str to a .csv file where this file
    should have the standard 'bid', 'ask', 'volume' columns and the index 'time'
    as a pandas datatime object. As a safe measure, we rename the .index and lower()
    the column names with standardize_data().
    """

    data = load_csv(data_path)
    data = standardize_data(data)

    if start_str is None and end_str is None:
        return data

    start_date, end_date = (
        datetime.strptime(start_str, "%d%m%Y"),
        datetime.strptime(end_str, "%d%m%Y"),
    )

    filtered = data.loc[start_date:end_date]
    # print(filtered)
    file_base_string, _, _, ext = extract_base_string(data_path)
    new_file_path = os.path.join(
        save_dir, file_base_string + f"_{start_str}_{end_str}" + ext
    )
    filtered.to_csv(new_file_path)
    print(f"{start_str} to {end_str} data saved to {new_file_path}")

    return filtered


def extract_base_string(string: str):
    """Function extracts the data_path base string up till the date.
    This is done so that we can replace the data path name with the correct date
    corresponding to the date range within the pd.DataFrame.

    e.g. 'eur_usd_tick_01012020_31122024' would extract 'eur_usd_tick_'
    """
    base_path, ext = os.path.splitext(string)

    pattern = (
        r"^(?P<base>.+?)_"  # group 1: stuff before the underscore
        r"(?P<dates>[0-9]{8}(?:_[0-9]{8})*)$"  # group 2: one or more 8‑digit blocks
    )

    base_string_match = re.match(pattern, os.path.basename(base_path))

    if base_string_match:
        base_string, dates = base_string_match["base"], base_string_match["dates"]

    else:
        raise ValueError("No matches found!")

    if base_string == "" or dates == "":
        raise ValueError("Error in extracting base string of data_path, please check!")

    start_date_str, end_date_str = dates.split("_")
    return base_string, start_date_str, end_date_str, ext


def load_csv(data_path: str):
    """Loads csv, hiding away the additional loading configs."""
    data = pd.read_csv(data_path, index_col=0, parse_dates=True)

    return data
