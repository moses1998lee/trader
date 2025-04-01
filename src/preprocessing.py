import pandas as pd


def calculate_smas(df: pd.DataFrame, periods: list[int], close_name: str):
    """For all the periods within the periods list, compute the simple moving
    average and add new columns containing those computed data."""
    for period in periods:
        df[f"sma_{period}"] = df[close_name].rolling(window=period).mean()

    return df


def calculate_emas(df: pd.DataFrame, spans: list[int], close_name: str):
    """For all the periods within the span list, compute the exponential moving
    average and add new columns containing those computed data."""

    for span in spans:
        df[f"ema_{span}"] = df[close_name].ewm(span=span, adjust=False).mean()

    return df


def calculate_rsis(df: pd.DataFrame, rsi_windows: list[int], close_name: str):
    """For all the windows within the window list, compute the relative strength
    index and add new columns containing those computed data."""
    delta = df[close_name].diff()
    delta_gain = delta.where(delta > 0, 0)
    delta_loss = -delta.where(delta < 0, 0)

    for window in rsi_windows:
        gain = delta_gain.rolling(window=window).mean()
        loss = delta_loss.rolling(window=window).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        df[f"rsi_{window}"] = rsi
    return df


def compute_lag_features(df: pd.DataFrame, first_col: str, second_col: str, lag: int):
    first = df[first_col]
    second = df[second_col].shift(-lag)

    feature = (first - second) / second

    return feature


# def process_for_esn(df, **esn_kwargs):
#     esn_init = esn_kwargs["esn_init"]
#     esn_fit = esn_kwargs["esn_fit"]
#     esn = ESN(n_inputs=input_dim, n_outputs=output_dim, **esn_init)

#     lagged_df = create_lagged_features(df)
#     train_input, train_target = np.array(lagged_df), np.array(train_target)
#     input_dim, output_dim = train_input.shape[1], train_target.shape[1]

#     esn.fit()

#     pass


# # Function to create lagged data
# def create_lagged_features(df, n_lags=5):
#     lagged_df = pd.DataFrame(index=df.index)
#     for col in df.columns:
#         for i in range(n_lags):
#             lagged_df[f"{col}_lag_{i + 1}"] = df[col].shift(i + 1)
#     return lagged_df.dropna()  # dropping rows with NaN values due to shifting


# def add_esn_feature():
#     pass
