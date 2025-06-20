import requests
from src.utils.generic import load_configs

binance = "https://api.binance.com/api/v3"
configs = load_configs()


def request(endpoint_type: str, **params):
    """Generic request used to reach endpoints"""
    url = binance + get_endpoint(endpoint_type)
    r = requests.get(url, params=params, timeout=10)
    r.raise_for_status()

    payload = r.json()

    return float(payload["price"])


def get_endpoint(endpoint_type: str):
    """Retrieves actual endpoint url based on 'english'"""
    match endpoint_type:
        case "depth":
            return "depth"
        case "historical":
            return "/historicalTrades"
        case "24hr_statistics":
            return "/ticker/24hr"
        case "statistics":
            return "/ticker"
        case "orderbook":
            return "bookTicker"
