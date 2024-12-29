from collections.abc import Collection, Iterable
from typing import TYPE_CHECKING, Generic, Iterator, Protocol, TypeAlias, TypeVar, runtime_checkable

if TYPE_CHECKING:
    from hypergrid.grid import HGrid

T = TypeVar("T")
T_co = TypeVar("T_co", covariant=True)
RawDimension: TypeAlias = tuple[str, Iterable]


@runtime_checkable
class Dimension(Protocol[T_co]):
    name: str

    def __repr__(self) -> str: ...

    def __str__(self) -> str:
        return self.__repr__()

    def __iter__(self) -> Iterator[T_co]: ...

    def to_grid(self) -> "HGrid":
        from hypergrid.grid import HGrid

        return HGrid(self)

    @staticmethod
    def make(**kwargs: Iterable) -> "Dimension":
        assert len(kwargs) == 1, "Dimensions must be 1-D"
        retdim: Dimension
        for name, value in kwargs.items():
            match value:
                case Collection():
                    retdim = FDimension(**{name: value})
                case _:
                    retdim = IDimension(**{name: value})
        return retdim


class FDimension(Dimension, Generic[T]):
    def __init__(self, **kwargs: Collection[T]):
        assert len(kwargs) == 1, "F(ixed)Dimension is 1-d, use Grids for multiple dimensions"
        for name, values in kwargs.items():
            assert isinstance(values, Collection), "FDimension assumes finite length, use IDimension for infinite iterable"
            self.name = name
            self.values = values

    def __repr__(self) -> str:
        return f"FDimension({repr(self.values)})"

    def __iter__(self) -> Iterator[T]:
        yield from self.values


class IDimension(Dimension, Generic[T]):
    def __init__(self, **kwargs: Iterable[T]):
        assert len(kwargs) == 1, "I(terable)Dimension is 1-d, use Grids for multiple dimensions"
        for name, values in kwargs.items():
            self.name = name
            self.values = values

    def __repr__(self) -> str:
        return f"IDimension({repr(self.values)})"

    def __iter__(self) -> Iterator[T]:
        yield from self.values
