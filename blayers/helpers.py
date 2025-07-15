import inspect
from functools import wraps
from typing import Callable


def accept_all(func: Callable) -> Callable:  # type: ignore
    sig = inspect.signature(func)
    valid_params = set(sig.parameters)

    @wraps(func)
    def wrapper(*args, **kwargs):  # type: ignore
        filtered_kwargs = {k: v for k, v in kwargs.items() if k in valid_params}
        return func(*args, **filtered_kwargs)

    return wrapper


def validate(func: Callable) -> Callable:  # type: ignore
    @wraps(func)
    def wrapper(*args, **kwargs):  # type: ignore
        sig = inspect.signature(func)
        bound = sig.bind(*args, **kwargs)
        y = bound.arguments["y"]  # Explicitly assume 'y' exists
        if len(y.shape) != 2:
            raise ValueError(
                "Argument 'y' must be a jax.Array with exactly 2 dimensions."
            )
        return func(*args, **kwargs)

    return wrapper
