from typing import Callable

_IndicatorRegistry: dict[str, Callable] = {}


def _register_indicator(indicator_registry: dict[str, Callable]):
    """
    Returns a decorator that will put class into a given registry.
    """

    def decorator(cls):
        key = cls.__name__.lower()
        if key in indicator_registry:
            raise KeyError(f"'{key}' already exists in registry!")

        indicator_registry[key] = cls
        return cls

    return decorator


register_indicator = _register_indicator(_IndicatorRegistry)
