import os

import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yfinance as yf
from gymnasium import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import DummyVecEnv

# Import the ESN module from the local pyESN folder
from pyESN.pyESN import ESN


def fetch_data(symbol: str, start_date: str, end_date: str) -> pd.DataFrame:
    """
    Fetch historical market data for a given symbol using Yahoo Finance.

    Args:
        symbol (str): The ticker symbol (e.g., 'EURUSD=X' for EUR/USD).
        start_date (str): The start date in 'YYYY-MM-DD' format.
        end_date (str): The end date in 'YYYY-MM-DD' format.

    Returns:
        pd.DataFrame: A DataFrame with columns Open, High, Low, Close, Volume.
    """
    data = yf.download(symbol, start=start_date, end=end_date)
    data = data[["Open", "High", "Low", "Close", "Volume"]]
    return data


def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Preprocess the raw market data by engineering features.

    Features:
      - Returns: Percentage change of the Close price.
      - Volatility: Rolling standard deviation of returns (window=10).
      - Momentum: Difference between short-term and long-term EMAs.

    Args:
        df (pd.DataFrame): Raw market data.

    Returns:
        pd.DataFrame: DataFrame with new columns 'Returns', 'Volatility', 'Momentum'.
    """
    df = df.copy()
    df["Returns"] = df["Close"].pct_change()
    df["Volatility"] = df["Returns"].rolling(window=10).std()
    ema_short = df["Close"].ewm(span=10, adjust=False).mean()
    ema_long = df["Close"].ewm(span=26, adjust=False).mean()
    df["Momentum"] = ema_short - ema_long
    df.dropna(inplace=True)
    return df


def compute_esn_feature(returns: np.ndarray) -> np.ndarray:
    """
    Compute an additional feature using an Echo State Network (ESN) on the returns.

    Since the current pyESN package does not have a `run` method, we train the ESN on a subset of the data (using the returns as both
    input and target) and then predict the transformed signal on the full series. This produces a new feature that captures the
    non-linear temporal patterns in the returns.

    Args:
        returns (np.ndarray): Array of return values with shape (n_samples, 1).

    Returns:
        np.ndarray: ESN prediction for each time step, to be used as a transformed feature.
    """
    esn = ESN(
        n_inputs=1, n_outputs=1, n_reservoir=50, spectral_radius=1.25, random_state=42
    )
    train_length = int(len(returns) * 0.8)
    train_input = returns[:train_length]
    train_target = returns[:train_length]
    esn.fit(train_input, train_target)
    esn_feature = esn.predict(returns)
    return np.squeeze(esn_feature)


def add_esn_feature(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add an ESN-generated feature column to the DataFrame.

    Args:
        df (pd.DataFrame): Preprocessed DataFrame with a 'Returns' column.

    Returns:
        pd.DataFrame: DataFrame with an added 'ESN_Feature' column.
    """
    returns_array = df["Returns"].values.reshape(-1, 1)
    df["ESN_Feature"] = compute_esn_feature(returns_array)
    return df


class TradingEnv(gym.Env):
    """
    A custom Gymnasium environment for trading.

    The environment simulates a simple trading scenario with:
      - Observations: A vector of engineered features for the current time step
                      along with account info (balance, position).
      - Actions: Discrete actions {0: Hold, 1: Buy (go long), 2: Sell (go short)}.
      - Reward: Change in total account value, net of a fixed transaction cost.
    """

    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        df: pd.DataFrame,
        initial_balance: float = 10000.0,
        transaction_cost_pct: float = 0.001,
    ):
        """
        Initialize the trading environment.

        Args:
            df (pd.DataFrame): DataFrame with features. Must contain 'Close', 'Returns',
                               'Volatility', 'Momentum', and 'ESN_Feature' columns.
            initial_balance (float): Starting cash balance.
            transaction_cost_pct (float): Transaction cost per trade (e.g., 0.001 = 0.1%).
        """
        super(TradingEnv, self).__init__()
        self.df = df.reset_index(drop=True)
        self.initial_balance = initial_balance
        self.transaction_cost_pct = transaction_cost_pct
        self.current_step = 0
        self.balance = initial_balance
        self.position = 0  # +1 for long, -1 for short, 0 for flat
        self.position_price = 0.0

        self.action_space = spaces.Discrete(3)
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(6,), dtype=np.float32
        )

    def _next_observation(self) -> np.ndarray:
        """
        Get the next observation from the dataset.

        Returns:
            np.ndarray: Concatenated vector of market features and account information.
        """
        row = self.df.iloc[self.current_step]
        # Use .item() to extract scalar values from single-element Series
        features = np.array(
            [
                row["Returns"].item(),
                row["Volatility"].item(),
                row["Momentum"].item(),
                row["ESN_Feature"].item(),
            ]
        )
        account_info = np.array(
            [float(self.balance), float(self.position)], dtype=np.float32
        )
        obs = np.concatenate((features, account_info), axis=0)
        return obs.astype(np.float32)

    def _take_action(self, action: int):
        """
        Execute the given action.

        Args:
            action (int): The action to take (0=Hold, 1=Buy, 2=Sell).
        """
        current_price = self.df.iloc[self.current_step]["Close"]
        cost = 0.0

        if action == 1:  # Buy
            if self.position == -1:
                cost = abs(self.position) * current_price * self.transaction_cost_pct
                self.balance += self.position * current_price
                self.position = 0
            if self.position == 0:
                self.position = 1
                self.position_price = current_price
                cost += self.balance * self.transaction_cost_pct
                self.balance = 0

        elif action == 2:  # Sell
            if self.position == 1:
                cost = abs(self.position) * current_price * self.transaction_cost_pct
                self.balance += self.position * current_price
                self.position = 0
            if self.position == 0:
                self.position = -1
                self.position_price = current_price
                cost += self.balance * self.transaction_cost_pct
                self.balance = 0

        self.balance -= cost

    def step(self, action: int):
        """
        Perform one step in the environment.

        Args:
            action (int): The action to execute.

        Returns:
            tuple: (observation, reward, terminated, truncated, info)
        """
        prev_total_asset = self._get_total_asset()
        print(f"action: {action}")
        self._take_action(action)
        self.current_step += 1
        current_total_asset = self._get_total_asset()
        reward = current_total_asset - prev_total_asset
        print(f"reward: {reward}")
        terminated = self.current_step >= len(self.df) - 1
        truncated = False
        obs = self._next_observation()
        info = {"total_asset": current_total_asset}
        return obs, reward, terminated, truncated, info

    def reset(self, seed=None, options=None):
        """
        Reset the environment to initial state.

        Returns:
            tuple: (observation, info)
        """
        super().reset(seed=seed)
        self.current_step = 0
        self.balance = self.initial_balance
        self.position = 0
        self.position_price = 0.0
        obs = self._next_observation()
        return obs, {}

    def _get_total_asset(self) -> float:
        """
        Calculate the total asset value (balance plus open position value).

        Returns:
            float: Total asset value.
        """
        current_price = self.df.iloc[self.current_step]["Close"]
        print(f"df: {self.df}")
        print(f"current_price: {current_price}")
        position_value = self.position * current_price
        print(f"PV: {position_value}")
        return self.balance + position_value

    def render(self, mode="human"):
        """
        Render the current state of the environment.
        """
        total_asset = self._get_total_asset()
        print(
            f"Step: {self.current_step} | Balance: {self.balance:.2f} | Position: {self.position} | Total Asset: {total_asset:.2f}"
        )


class TrainingProgressCallback(BaseCallback):
    """
    A custom callback for logging training progress.
    Prints training progress every 1000 timesteps.
    """

    def __init__(self, verbose=1):
        super(TrainingProgressCallback, self).__init__(verbose)

    def _on_step(self) -> bool:
        if self.n_calls % 1000 == 0:
            print(f"Training progress: {self.num_timesteps} timesteps completed.")
        return True


def train_rl_model(
    env: gym.Env, total_timesteps: int = 50000, checkpoint_path: str = "./checkpoints/"
) -> PPO:
    """
    Train a PPO-based RL agent on the given environment.
    This function will only save the model if a KeyboardInterrupt occurs during training.

    Args:
        env (gym.Env): The trading environment.
        total_timesteps (int): Total training timesteps.
        checkpoint_path (str): Directory to store the model if training is interrupted.

    Returns:
        PPO: The trained PPO model.
    """
    if not os.path.exists(checkpoint_path):
        os.makedirs(checkpoint_path)
    vec_env = DummyVecEnv([lambda: env])
    model = PPO("MlpPolicy", vec_env, verbose=1)
    checkpoint_file = os.path.join(checkpoint_path, "latest_model")

    try:
        model.learn(total_timesteps=total_timesteps)
    except KeyboardInterrupt:
        checkpoint_file = os.path.join(checkpoint_path, "interrupted_model")
        print("KeyboardInterrupt detected. Saving the current model state.")

        model.save(checkpoint_file)
        print(f"Model saved to {checkpoint_file}")
        return model

    model.save(checkpoint_file)
    print(f"Model saved to {checkpoint_file}")
    return model


def run_inference_with_plot(model: PPO, env: gym.Env):
    """
    Run the trained RL model on the environment (simulate live trading) and plot the total portfolio value over time.

    Args:
        model (PPO): The trained RL agent.
        env (gym.Env): The trading environment.
    """
    obs, _ = env.reset()
    done = False
    portfolio_values = [env._get_total_asset()]

    while not done:
        action, _ = model.predict(obs)
        obs, reward, done, truncated, info = env.step(action)
        portfolio_values.append(env._get_total_asset())

    plt.figure(figsize=(12, 6))
    plt.plot(portfolio_values, label="Portfolio Value", color="blue")
    plt.xlabel("Time Steps")
    plt.ylabel("Portfolio Value")
    plt.title("Portfolio Value Over Time")
    plt.legend()
    plt.grid(True)
    plt.savefig("portfolio_value.png")
    print("Plot saved as portfolio_value.png")


def main():
    """
    Main function to execute the trading pipeline with a training/testing split.

    Steps:
      1. Retrieve historical data.
      2. Split data chronologically to avoid leakage.
      3. Preprocess and engineer features on both train and test data.
      4. Compute ESN-based feature and append to DataFrames.
      5. Create separate trading environments for training and testing.
      6. Train the RL model on the training environment (with checkpointing) or load checkpoint if interrupted.
      7. Run inference on the test environment and plot results.
    """
    # Fetch data covering a longer period
    symbol = "EURUSD=X"
    full_start_date = "2015-01-01"
    full_end_date = "2020-01-01"
    data = fetch_data(symbol, full_start_date, full_end_date)

    # Split data chronologically
    train_data = data.loc["2015-01-01":"2018-12-31"].copy()
    test_data = data.loc["2019-01-01":full_end_date].copy()

    # Preprocess and engineer features for training data
    train_data_preprocessed = preprocess_data(train_data)
    train_data_with_esn = add_esn_feature(train_data_preprocessed)

    # Preprocess and engineer features for test data
    test_data_preprocessed = preprocess_data(test_data)
    test_data_with_esn = add_esn_feature(test_data_preprocessed)

    # Optionally, visualize engineered features on training data
    train_data_with_esn[["Returns", "Volatility", "Momentum", "ESN_Feature"]].plot(
        subplots=True, figsize=(10, 8)
    )
    plt.show()

    # Create training and testing environments
    train_env = TradingEnv(train_data_with_esn)
    test_env = TradingEnv(test_data_with_esn)

    checkpoint_path = "./checkpoints/"
    try:
        model = train_rl_model(
            train_env, total_timesteps=500000, checkpoint_path=checkpoint_path
        )
    except KeyboardInterrupt:
        print("Training interrupted. Loading latest checkpoint for inference.")
        try:
            for name in ["latest_model", "interrupted_model"]:
                checkpoint_file = os.path.join(checkpoint_path, name)
                vec_env = DummyVecEnv([lambda: train_env])
                model = PPO.load(checkpoint_file, env=vec_env)
        except:
            raise KeyError(
                "The saved filename is not valid! Must be either 'latest_model' or 'interrupted_model'."
            )

    # Run inference on the test environment to evaluate on unseen data
    run_inference_with_plot(model, test_env)


if __name__ == "__main__":
    main()

# # %%
# import matplotlib.pyplot as plt
# import numpy as np


# class EchoStateNetwork:
#     def __init__(self, reservoir_size, spectral_radius=0.9):
#         # Initialize network parameters
#         self.reservoir_size = reservoir_size

#         # Reservoir weights
#         self.W_res = np.random.rand(reservoir_size, reservoir_size) - 0.5
#         self.W_res *= spectral_radius / np.max(np.abs(np.linalg.eigvals(self.W_res)))

#         # Input weights
#         self.W_in = np.random.rand(reservoir_size, 1) - 0.5

#         # Output weights (to be trained)
#         self.W_out = None

#     def train(self, input_data, target_data):
#         # Run reservoir with input data
#         reservoir_states = self.run_reservoir(input_data)

#         # Train the output weights using pseudo-inverse
#         self.W_out = np.dot(np.linalg.pinv(reservoir_states), target_data)

#     def predict(self, input_data):
#         # Run reservoir with input data
#         reservoir_states = self.run_reservoir(input_data)

#         # Make predictions using the trained output weights
#         predictions = np.dot(reservoir_states, self.W_out)

#         return predictions

#     def run_reservoir(self, input_data):
#         # Initialize reservoir states
#         reservoir_states = np.zeros((len(input_data), self.reservoir_size))

#         # Run the reservoir
#         for t in range(1, len(input_data)):
#             reservoir_states[t, :] = np.tanh(
#                 np.dot(self.W_res, reservoir_states[t - 1, :])
#                 + np.dot(self.W_in, input_data[t])
#             )

#         return reservoir_states


# # Generate synthetic data (input: random noise, target: sine wave)
# time = np.arange(0, 20, 0.1)
# noise = 0.1 * np.random.rand(len(time))
# sine_wave_target = np.sin(time)

# # Create an Echo State Network
# reservoir_size = 50

# esn = EchoStateNetwork(reservoir_size)
# # Prepare training data
# training_input = noise[:, None]
# training_target = sine_wave_target[:, None]

# # Train the ESN
# esn.train(training_input, training_target)

# # Generate test data (similar to training data for simplicity)
# test_input = noise[:, None]
# # Make predictions
# predictions = esn.predict(test_input)

# # Plot the results
# plt.figure(figsize=(10, 6))
# plt.plot(time, sine_wave_target, label="True Sine Wave", linestyle="--", marker="o")
# plt.plot(time, predictions, label="ESN Prediction", linestyle="--", marker="o")
# plt.xlabel("Time")
# plt.ylabel("Amplitude")
# plt.legend()
# plt.title("Echo State Network Learning to Generate Sine Wave")
# plt.show()


# # %%
# import numpy as np

# # Define the matrices
# W_res1 = np.array([[0.8, 0.3], [0.2, 0.9]])
# W_res2 = np.array([[2, 0.3], [0.2, 2]])

# # Calculate eigenvalues
# eigenvalues1 = np.linalg.eigvals(W_res1)
# eigenvalues2 = np.linalg.eigvals(W_res2)

# print("Eigenvalues of W_res1:", eigenvalues1)
# print("Eigenvalues of W_res2:", eigenvalues2)

# # %%
# # Define an initial state
# x_initial = np.array([1, 0])

# # Evolution under W_res1
# states_res1 = [x_initial]
# for _ in range(5):
#     states_res1.append(W_res1.dot(states_res1[-1]))

# # Evolution under W_res2
# states_res2 = [x_initial]
# for _ in range(5):
#     states_res2.append(W_res2.dot(states_res2[-1]))

# print("States under W_res1:", states_res1)
# print("States under W_res2:", states_res2)

# # %%
