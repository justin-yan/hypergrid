from __future__ import annotations

import itertools
from collections import namedtuple
from typing import TYPE_CHECKING, Any, Callable, Iterable, Iterator, Protocol, Type, runtime_checkable

from hypergrid.util import instantiate_lambda

if TYPE_CHECKING:
    from sklearn.model_selection import ParameterGrid

from hypergrid.dimension import Dimension, RawDimension


@runtime_checkable
class Grid(Protocol):
    grid_element: Type[tuple]

    @property
    def dimension_names(self) -> list[str]:
        return list(self.grid_element._fields)  # type: ignore[attr-defined]

    def __repr__(self) -> str: ...

    def __str__(self) -> str:
        return self.__repr__()

    def __iter__(self) -> Iterator: ...

    def __add__(self, other: Grid | Dimension | RawDimension) -> SumGrid:
        match other:
            case Grid():
                return SumGrid(self, other)
            case Dimension():
                return SumGrid(self, HGrid(other))
            case (str(s), it) if isinstance(it, Iterable):  # RawDimension
                return SumGrid(self, HGrid(Dimension.make(**{s: it})))
            case _:
                raise ValueError("Invalid argument for grid operation")

    def __or__(self, other: Grid | Dimension | RawDimension) -> SumGrid:
        return self.__add__(other)

    def __mul__(self, other: Grid | Dimension | RawDimension) -> ProductGrid:
        match other:
            case Grid():
                return ProductGrid(self, other)
            case Dimension():
                return ProductGrid(self, HGrid(other))
            case (str(s), it) if isinstance(it, Iterable):  # RawDimension
                return ProductGrid(self, HGrid(Dimension.make(**{s: it})))
            case _:
                raise ValueError("Invalid argument for grid operation")

    def __and__(self, other: Grid | Dimension | RawDimension) -> ZipGrid:
        match other:
            case Grid():
                return ZipGrid(self, other)
            case Dimension():
                return ZipGrid(self, HGrid(other))
            case (str(s), it) if isinstance(it, Iterable):  # RawDimension
                return ZipGrid(self, HGrid(Dimension.make(**{s: it})))
            case _:
                raise ValueError("Invalid argument for grid operation")

    def filter(self, predicate: Callable[[Any], bool]) -> FilterGrid:
        return FilterGrid(self, predicate)

    def select(self, *dim_names: str) -> SelectGrid:
        return SelectGrid(self, *dim_names)

    def map(self, **kwargs: Callable[[Any], Any]) -> MapGrid:
        return MapGrid(self, **kwargs)

    def map_to(self, **kwargs: Callable[[Any], Any]) -> MapToGrid:
        return MapToGrid(self, **kwargs)

    def instantiate(self, **kwargs: Type) -> MapToGrid:
        return self.map_to(**{name: instantiate_lambda(cls) for name, cls in kwargs.items()})

    def to_sklearn(self) -> ParameterGrid:  # type: ignore[no-any-unimported]
        from hypergrid.ext.sklearn import _grid_to_sklearn

        return _grid_to_sklearn(self)


class HGrid(Grid):
    dimensions: list[Dimension]

    def __init__(self, *args: Dimension, **kwargs: Iterable) -> None:
        dims = list(args)
        for dim, values in kwargs.items():
            dims.append(Dimension.make(**{dim: values}))
        assert len(dims) > 0, "Must provide at least one meaningful dimension"
        assert len(dims) == len(set(dims)), "Dimension names must be unique"
        self.dimensions = dims
        self.grid_element = namedtuple("GridElement", [dim.name for dim in self.dimensions])  # type: ignore[misc]

    def __repr__(self) -> str:
        dim_str = ", ".join([repr(dim) for dim in self.dimensions])
        return f"HGrid({dim_str})"

    def __iter__(self) -> Iterator:
        for element_tuple in itertools.product(*[dim.__iter__() for dim in self.dimensions]):
            yield self.grid_element(*element_tuple)


class SumGrid(Grid):
    def __init__(self, grid1: Grid, grid2: Grid) -> None:
        assert set(grid1.dimension_names) == set(grid2.dimension_names)
        self.grid1 = grid1
        self.grid2 = grid2
        self.grid_element = grid1.grid_element

    def __repr__(self) -> str:
        return f"SumGrid({repr(self.grid1)}, {repr(self.grid2)})"

    def __iter__(self) -> Iterator:
        for grid_element in itertools.chain(self.grid1, self.grid2):
            yield grid_element


class ProductGrid(Grid):
    def __init__(self, grid1: Grid, grid2: Grid) -> None:
        assert set(grid1.dimension_names).isdisjoint(set(grid2.dimension_names)), "Dimensions must be exactly matching"
        self.grid1 = grid1
        self.grid2 = grid2
        self.grid_element = namedtuple("GridElement", grid1.dimension_names + grid2.dimension_names)  # type: ignore[misc]

    def __repr__(self) -> str:
        return f"ProductGrid({repr(self.grid1)}, {repr(self.grid2)})"

    def __iter__(self) -> Iterator:
        for grid_element1, grid_element2 in itertools.product(self.grid1, self.grid2):
            yield self.grid_element(*(grid_element1 + grid_element2))


class ZipGrid(Grid):
    """
    Mimic python "zip" of two iterables.
    """

    def __init__(self, grid1: Grid, grid2: Grid) -> None:
        assert set(grid1.dimension_names).isdisjoint(set(grid2.dimension_names)), "Dimensions must be exactly matching"
        self.grid1 = grid1
        self.grid2 = grid2
        self.grid_element = namedtuple("GridElement", grid1.dimension_names + grid2.dimension_names)  # type: ignore[misc]

    def __repr__(self) -> str:
        return f"ZipGrid({repr(self.grid1)}, {repr(self.grid2)})"

    def __iter__(self) -> Iterator:
        for grid_element1, grid_element2 in zip(self.grid1, self.grid2):
            yield self.grid_element(*(grid_element1 + grid_element2))


class FilterGrid(Grid):
    def __init__(self, grid: Grid, predicate: Callable[[Any], bool]) -> None:
        self.grid = grid
        self.predicate = predicate
        self.grid_element = grid.grid_element

    def __repr__(self) -> str:
        return f"FilterGrid({repr(self.grid)}, {self.predicate.__name__})"

    def __iter__(self) -> Iterator:
        for grid_element in self.grid:
            if self.predicate(grid_element):
                yield grid_element


class SelectGrid(Grid):
    def __init__(self, grid: Grid, *select_dims: str) -> None:
        assert len(set(select_dims)) == len(select_dims), "Selected columns must all be unique"
        assert set(select_dims) <= set(grid.dimension_names), "Selected dimensions must be subset of grid dimensions"
        self.grid = grid
        self.select_dims = select_dims
        self.grid_element = namedtuple("GridElement", [name for name in grid.dimension_names if name in self.select_dims])  # type: ignore[misc]

    def __repr__(self) -> str:
        return f"SelectGrid({repr(self.grid)}, {repr(self.select_dims)})"

    def __iter__(self) -> Iterator:
        for grid_element in self.grid:
            element_list = []
            for dim_name in self.dimension_names:
                try:
                    selected_value = getattr(grid_element, dim_name)
                except AttributeError:
                    selected_value = None
                element_list.append(selected_value)
            yield self.grid_element(*element_list)


class MapGrid(Grid):
    def __init__(self, grid: Grid, **kwargs: Callable[[Any], Any]) -> None:
        assert len(set(kwargs.keys())) == len(kwargs.keys()), "New columns must all have unique names"
        self.grid = grid
        self.dimension_mapping = kwargs
        self.grid_element = namedtuple("GridElement", list(kwargs.keys()))  # type: ignore[misc]

    def __repr__(self) -> str:
        mappings_str = ", ".join([f"{dim_name}={func.__name__}" for dim_name, func in self.dimension_mapping.items()])
        return f"MapGrid({repr(self.grid)}, {mappings_str})"

    def __iter__(self) -> Iterator:
        for grid_element in self.grid:
            new_values = {dim_name: func(grid_element) for dim_name, func in self.dimension_mapping.items()}
            yield self.grid_element(**new_values)


class MapToGrid(Grid):
    def __init__(self, grid: Grid, **kwargs: Callable[[Any], Any]) -> None:
        assert len(set(kwargs.keys())) == len(kwargs.keys()), "New columns must all have unique names"
        assert set(grid.dimension_names).isdisjoint(set(kwargs.keys())), "New columns must not have name collisions with old columns"
        self.grid = grid
        self.dimension_mapping = kwargs
        self.grid_element = namedtuple("GridElement", grid.dimension_names + list(kwargs.keys()))  # type: ignore[misc]

    def __repr__(self) -> str:
        mappings_str = ", ".join([f"{dim_name}={func.__name__}" for dim_name, func in self.dimension_mapping.items()])
        return f"MapToGrid({repr(self.grid)}, {mappings_str})"

    def __iter__(self) -> Iterator:
        for grid_element in self.grid:
            new_values = {dim_name: func(grid_element) for dim_name, func in self.dimension_mapping.items()}
            yield self.grid_element(**(grid_element._asdict() | new_values))
