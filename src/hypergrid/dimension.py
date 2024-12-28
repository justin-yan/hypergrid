from typing import TYPE_CHECKING, Generic, Iterable, Iterator, Protocol, TypeVar, runtime_checkable

if TYPE_CHECKING:
    from hypergrid.grid import Grid

T = TypeVar("T")
T_co = TypeVar("T_co", covariant=True)


@runtime_checkable
class IDimension(Protocol[T_co]):
    name: str

    def __repr__(self) -> str: ...

    def __str__(self) -> str:
        return self.__repr__()

    def __iter__(self) -> Iterator[T_co]: ...

    def to_grid(self) -> "Grid":
        from hypergrid.grid import Grid

        return Grid(self)


class Dimension(IDimension, Generic[T]):
    def __init__(self, **kwargs: Iterable[T]):
        assert len(kwargs) == 1, "Dimensions are 1-d, use Grids for multiple dimensions"
        for name, values in kwargs.items():
            self.name = name
            self.values = values

    def __repr__(self) -> str:
        return f"Dimension({repr(self.values)})"

    def __iter__(self) -> Iterator[T]:
        yield from self.values
