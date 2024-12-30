import random
from typing import Any, Iterator, Protocol, TypeVar, runtime_checkable

from hypergrid.gen.iterable import HIterable

T = TypeVar("T", covariant=True)


@runtime_checkable
class Distribution(HIterable, Protocol[T]):
    def sample(self) -> T: ...

    def __iter__(self) -> Iterator[T]:
        while True:
            yield self.sample()

    def __call__(self, *args: Any, **kwargs: Any) -> T:
        return self.sample()


class Uniform(Distribution):
    def __init__(self, low: float, high: float) -> None:
        self.low = low
        self.high = high

    def sample(self) -> float:
        return random.uniform(self.low, self.high)
