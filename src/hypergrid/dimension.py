from typing import Generic, Iterator, Protocol, TypeVar

T = TypeVar("T")
T_co = TypeVar("T_co", covariant=True)


class IDimension(Protocol[T_co]):
    name: str

    def __repr__(self) -> str: ...

    def __str__(self) -> str:
        return self.__repr__()

    def __iter__(self) -> Iterator[T_co]: ...


# TODO: Create a type alias for raw python type instantiation options (range/slice iterators, list, etc.)
# TODO: support generator (infinite) dimensions


class Dimension(IDimension, Generic[T]):
    def __init__(self, **kwargs: list[T]):
        assert len(kwargs) == 1, "Dimensions are 1-d, use Grids for multiple dimensions"
        for name, values in kwargs.items():
            self.name = name
            self.values = values

    def __repr__(self) -> str:
        return f"Dimension({repr(self.values)})"

    def __iter__(self) -> Iterator[T]:
        yield from self.values


if __name__ == "__main__":
    print(str(Dimension(test=[1, 2, 3])))
