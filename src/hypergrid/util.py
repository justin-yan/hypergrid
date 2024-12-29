from typing import Callable, Type


def instantiate_lambda(cls: Type) -> Callable:
    return lambda ge: cls(**ge._asdict())
