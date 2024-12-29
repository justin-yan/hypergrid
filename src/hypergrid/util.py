from typing import Callable, Type, TypeVar

T = TypeVar("T")


def instantiate_lambda(cls: Type) -> Callable:
    return lambda ge: cls(**ge._asdict())
