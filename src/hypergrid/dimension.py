import random
from collections.abc import Callable, Collection
from typing import TYPE_CHECKING, Generic, Iterator, Protocol, Self, TypeAlias, TypeVar, runtime_checkable

if TYPE_CHECKING:
    from hypergrid.grid import HGrid

T = TypeVar("T")
T_co = TypeVar("T_co", covariant=True)
RawDimension: TypeAlias = tuple[str, Collection]


@runtime_checkable
class Dimension(Protocol[T_co]):
    name: str

    def __repr__(self) -> str: ...

    def __str__(self) -> str:
        return self.__repr__()

    def __iter__(self) -> Iterator[T_co]: ...

    def sample(self) -> T_co: ...

    def to_grid(self) -> "HGrid":
        from hypergrid.grid import HGrid

        return HGrid(self)


class FixedDimension(Dimension, Generic[T]):
    _sampling_strategy: Callable[..., T] = random.choice

    def __init__(self, **kwargs: Collection[T]):
        assert len(kwargs) == 1, "FixedDimension is 1-d, use Grids for multiple dimensions"
        for name, values in kwargs.items():
            assert isinstance(values, Collection), "FixedDimension assumes finite length"
            self.name = name
            self.values = values

    def __repr__(self) -> str:
        return f"FixedDimension({repr(self.values)})"

    def __iter__(self) -> Iterator[T]:
        yield from self.values

    def sample(self) -> T:
        return self._sampling_strategy(self.values)

    def with_sampling_strategy(self, strategy: Callable[..., T]) -> Self:
        self._sampling_strategy = strategy
        return self


@runtime_checkable
class DistributionDimension(Dimension, Protocol[T]):
    _itermax: int = 500

    def __iter__(self) -> Iterator[T]:
        for _ in range(self._itermax):
            yield self.sample()

    def sample_n(self, n: int) -> FixedDimension[T]:
        return FixedDimension(**{self.name: [self.sample() for _ in range(n)]})

    def with_itermax(self, n: int) -> Self:
        self._itermax = n
        return self
