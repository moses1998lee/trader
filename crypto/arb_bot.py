# %%
from src.api import request

PARAMS = {"symbol": "BTCUSDT", "type": "FULL"}
print(request("/ticker/bookTicker", **PARAMS))
# %%

if __name__ == "__main__":
    pass
