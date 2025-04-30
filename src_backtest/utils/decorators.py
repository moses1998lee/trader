from functools import wraps
from typing import Callable


def update_status(status_name: str):
    """
    Decorator to set the status of a method if it return True.
    """

    def decorator(fn):
        @wraps(fn)
        def wrapper(self, *args, **kwargs):
            hit = fn(self, *args, **kwargs)
            if hit:
                self.status = status_name
            return hit

        return wrapper

    return decorator


def register(registry: dict[str, Callable]):
    """
    Returns a decorator that will put fn into the given registry.
    """

    def decorator(fn):
        registry[fn.__name__] = fn
        return fn

    return decorator


def register_indicator(indicator_registry: dict[str, Callable]):
    """
    Returns a decorator that will put class into a given registry.
    """

    def decorator(cls):
        indicator_registry[cls.__name__.lower()] = cls
        return cls

    return decorator
