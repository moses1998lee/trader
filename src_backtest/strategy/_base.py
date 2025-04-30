class BaseEntry(ABC):
    def __init__(self, indicators: list[str]):
        """
        Base class for entry strategies.

        indicators (list[str]): List of indicators to be used in the strategy.
            This is extracted from _IndicatorRegistry in src_backtest.indicators.indicators
        """
        self.class_name = None
        self.strategy_configs = None

    def __repr__(self):
        return f"{self.class_name}"

    @abstractmethod
    def process_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Process the data with respects to the required information for the
        specific entry strategy using the relevant indicators.
        """
        raise NotImplementedError

    def indicators(self) -> dict[str, Callable]:
        """
        Return the indicators used in the strategy.
        """
        return _IndicatorRegistry
