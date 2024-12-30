import random
from collections.abc import Callable, Collection
from typing import TYPE_CHECKING, Generic, Iterator, Self, TypeAlias, TypeVar

if TYPE_CHECKING:
    from hypergrid.grid import HyperGrid

T = TypeVar("T")
T_co = TypeVar("T_co", covariant=True)
RawDimension: TypeAlias = tuple[str, Collection]


class Dimension(Generic[T]):
    name: str
    _sampling_strategy: Callable[..., T] = random.choice

    def __init__(self, **kwargs: Collection[T]):
        assert len(kwargs) == 1, "Dimension is 1-d, use Grids for multiple dimensions"
        for name, values in kwargs.items():
            assert isinstance(values, Collection), "Dimension assumes finite length"
            self.name = name
            self.values = values

    def __repr__(self) -> str:
        return f"Dimension({repr(self.values)})"

    def __str__(self) -> str:
        return self.__repr__()

    def __len__(self) -> int:
        return len(self.values)

    def __iter__(self) -> Iterator[T]:
        yield from self.values

    def sample(self) -> T:
        return self._sampling_strategy(self.values)

    def with_sampling_strategy(self, strategy: Callable[..., T]) -> Self:
        self._sampling_strategy = strategy
        return self

    def with_name(self, name: str) -> Self:
        self.name = name
        return self

    def to_grid(self) -> "HyperGrid":
        from hypergrid.grid import HyperGrid

        return HyperGrid(self)
