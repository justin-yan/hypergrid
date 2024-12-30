from itertools import islice
from typing import Iterator, Protocol, Self, TypeVar, runtime_checkable

from hypergrid.dimension import Dimension

T = TypeVar("T")


@runtime_checkable
class HIterable(Protocol[T]):
    name: str = "anonymous"

    def __iter__(self) -> Iterator[T]: ...

    def take(self, n: int) -> Dimension[T]:
        return Dimension(**{self.name: [i for i in islice(self, n)]})

    def with_name(self, name: str) -> Self:
        self.name = name
        return self


class ExponentialStep(HIterable):
    def __init__(self, start: float, step: float) -> None:
        self.start = start
        self.step = step

    def __iter__(self) -> Iterator[float]:
        cursor = self.start
        while True:
            yield cursor
            cursor *= self.step
