from typing import Callable


def instantiate_lambda(cls: type) -> Callable:
    return lambda ge: cls(**ge._asdict())
