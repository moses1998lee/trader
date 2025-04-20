from .configurator import Configurator


class Printer:
    def __init__(self, configurator: Configurator, verbose: bool = int):
        self.verbose = verbose
        self.extra_verbose = self._configure_extra_verbose()
        self.configurator = configurator
        self.end_simulation = {}

    def _configure_extra_verbose(self):
        if self.verbose == 2:
            return 1
        else:
            0

    def print_begin(self, currency: str):
        if self.verbose:
            print(
                f"Simulation Begin for: {currency}\n"
                f"{self.configurator.data_provider}, {self.configurator.granularity}"
                "\n------------------------------------------------------\n"
            )

    def print_trade_close(self, trade_close_str: str, current_capital: float):
        if self.extra_verbose:
            print(f"{trade_close_str}: Capital Remaining: {current_capital}")

    def print_trade_open(self, trade_open_str: str, current_capital: float):
        if self.extra_verbose:
            print(f"{trade_open_str}: Capital Remaining: {current_capital}")

    def store_end_simulation(
        self,
        start: str,
        end: str,
        currency: str,
        initial_capital: float,
        final_capital: float,
        pnl_percent: float,
        max_drawdown: float,
    ):
        end_simulation_str = f"{currency}  =  {start}-{end}: ${initial_capital} → ${final_capital}, pnl: {pnl_percent:.2f}%  max_drawdown %: {max_drawdown:.1f}%"
        self.end_simulation[currency] = end_simulation_str

    def print_end_simulation_summary(self):
        if self.verbose:
            print("--------------------------------------------------")
            for _, summary in self.end_simulation.items():
                print(summary)


# %%
