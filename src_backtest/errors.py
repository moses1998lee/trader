class MissingNewCol(Exception):
    """
    MissingColumnName is a user-defined exception class that is raised when the new column name required
    cannot be found within the strategy specific input argument 'strategy_new_col'

    message : str
        Human-readable explanation of the error.
    error_code : int, optional
        A specific code representing the error type, which can be used for easier debugging or mapping
        error responses.
    """

    def __init__(self, message: str, error_code: int = None) -> None:
        super().__init__(message)
        self.message = message
        self.error_code = error_code

    def __str__(self) -> str:
        if self.error_code is not None:
            return f"[Error {self.error_code}] {self.message}"
        return self.message
