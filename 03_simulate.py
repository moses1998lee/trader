# %%

from src.configurator import Configurator
from src.entry_strat import SpreadReversionEntry
from src.exit_strat import MinSLRatioTP
from src.simulator import Simulator
from src.utils import configs

CONFIGS = configs()

sim_configs = CONFIGS.simulation_configs
trade_configs = sim_configs.trading_configs
entry_strat_configs = CONFIGS.entry_strategies
exit_strat_configs = CONFIGS.exit_strategies

entry_strategy = SpreadReversionEntry(
    spread_reversion_configs=CONFIGS.entry_strategies.spread_reversion.configs,
    strategy_new_cols=CONFIGS.entry_strategies.spread_reversion.new_cols,
)
exit_strategy = MinSLRatioTP(
    min_sl_window_lookback=exit_strat_configs.min_sl_ratio_tp.min_sl_window_lookback,
    risk_to_reward_str=trade_configs.risk_to_reward_str,
)

# Account class and PositionSize class instantiated by Configurator
configurator = Configurator(
    data_provider=sim_configs.provider,
    granularity=sim_configs.granularity,
    currencies=sim_configs.currencies,
    start_end_str=(sim_configs.start_str, sim_configs.end_str),
    initial_capital=trade_configs.capital,
    allowed_leverage=trade_configs.allowed_leverage,
    max_risk=trade_configs.max_risk,
    entry_strategy=entry_strategy,
    exit_strategy=exit_strategy,
)

if __name__ == "__main__":
    simulator = Simulator(configurator, verbose=sim_configs.verbose)
    simulator.simulate()
