from __future__ import annotations

import itertools
from collections import namedtuple
from typing import Any, Callable, Iterator, List, Protocol, Type, runtime_checkable

from hypergrid.dimension import Dimension, IDimension


@runtime_checkable
class IGrid(Protocol):
    grid_element: Type[tuple]

    @property
    def dimension_names(self) -> list[str]:
        return list(self.grid_element._fields)  # type: ignore[attr-defined]

    def __repr__(self) -> str: ...

    def __str__(self) -> str:
        return self.__repr__()

    def __iter__(self) -> Iterator: ...

    def __add__(self, other: IGrid | IDimension) -> SumGrid:
        match other:
            case IGrid():
                return SumGrid(self, other)
            case IDimension():
                return SumGrid(self, Grid(other))

    def __or__(self, other: IGrid | IDimension) -> SumGrid:
        return self.__add__(other)

    def __mul__(self, other: IGrid | IDimension) -> ProductGrid:
        match other:
            case IGrid():
                return ProductGrid(self, other)
            case IDimension():
                return ProductGrid(self, Grid(other))

    def __and__(self, other: IGrid | IDimension) -> ZipGrid:
        match other:
            case IGrid():
                return ZipGrid(self, other)
            case IDimension():
                return ZipGrid(self, Grid(other))

    def filter(self, predicate: Callable[[Any], bool]) -> FilterGrid:
        return FilterGrid(self, predicate)

    def select(self, *dim_names: str) -> SelectGrid:
        return SelectGrid(self, *dim_names)

    def map(self, **kwargs: Callable[[Any], Any]) -> MapGrid:
        return MapGrid(self, **kwargs)

    def map_to(self, **kwargs: Callable[[Any], Any]) -> MapToGrid:
        return MapToGrid(self, **kwargs)


class Grid(IGrid):
    dimensions: list[IDimension]

    def __init__(self, *args: IDimension, **kwargs: List[Any]) -> None:
        dims = list(args)
        for dim, values in kwargs.items():
            dims.append(Dimension(**{dim: values}))
        assert len(dims) > 0, "Must provide at least one meaningful dimension"
        assert len(dims) == len(set(dims)), "Dimension names must be unique"
        self.dimensions = dims
        self.grid_element = namedtuple("GridElement", [dim.name for dim in self.dimensions])  # type: ignore[misc]

    def __repr__(self) -> str:
        dim_str = ", ".join([repr(dim) for dim in self.dimensions])
        return f"Grid({dim_str})"

    def __iter__(self) -> Iterator:
        for element_tuple in itertools.product(*[dim.__iter__() for dim in self.dimensions]):
            yield self.grid_element(*element_tuple)


class SumGrid(IGrid):
    def __init__(self, grid1: IGrid, grid2: IGrid) -> None:
        assert set(grid1.dimension_names) == set(grid2.dimension_names)
        self.grid1 = grid1
        self.grid2 = grid2
        self.grid_element = grid1.grid_element

    def __repr__(self) -> str:
        return f"SumGrid({repr(self.grid1)}, {repr(self.grid2)})"

    def __iter__(self) -> Iterator:
        for grid_element in itertools.chain(self.grid1, self.grid2):
            yield grid_element


class ProductGrid(IGrid):
    def __init__(self, grid1: IGrid, grid2: IGrid) -> None:
        assert set(grid1.dimension_names).isdisjoint(set(grid2.dimension_names)), "Dimensions must be exactly matching"
        self.grid1 = grid1
        self.grid2 = grid2
        self.grid_element = namedtuple("GridElement", grid1.dimension_names + grid2.dimension_names)  # type: ignore[misc]

    def __repr__(self) -> str:
        return f"ProductGrid({repr(self.grid1)}, {repr(self.grid2)})"

    def __iter__(self) -> Iterator:
        for grid_element1, grid_element2 in itertools.product(self.grid1, self.grid2):
            yield self.grid_element(*(grid_element1 + grid_element2))


class ZipGrid(IGrid):
    """
    Mimic python "zip" of two iterables.
    """

    def __init__(self, grid1: IGrid, grid2: IGrid) -> None:
        assert set(grid1.dimension_names).isdisjoint(set(grid2.dimension_names)), "Dimensions must be exactly matching"
        self.grid1 = grid1
        self.grid2 = grid2
        self.grid_element = namedtuple("GridElement", grid1.dimension_names + grid2.dimension_names)  # type: ignore[misc]

    def __repr__(self) -> str:
        return f"ZipGrid({repr(self.grid1)}, {repr(self.grid2)})"

    def __iter__(self) -> Iterator:
        for grid_element1, grid_element2 in zip(self.grid1, self.grid2):
            yield self.grid_element(*(grid_element1 + grid_element2))


class FilterGrid(IGrid):
    def __init__(self, grid: IGrid, predicate: Callable[[Any], bool]) -> None:
        self.grid = grid
        self.predicate = predicate
        self.grid_element = grid.grid_element

    def __repr__(self) -> str:
        return f"FilterGrid({repr(self.grid)}, {self.predicate.__name__})"

    def __iter__(self) -> Iterator:
        for grid_element in self.grid:
            if self.predicate(grid_element):
                yield grid_element


class SelectGrid(IGrid):
    def __init__(self, grid: IGrid, *select_dims: str) -> None:
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


class MapGrid(IGrid):
    def __init__(self, grid: IGrid, **kwargs: Callable[[Any], Any]) -> None:
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


class MapToGrid(IGrid):
    def __init__(self, grid: IGrid, **kwargs: Callable[[Any], Any]) -> None:
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


if __name__ == "__main__":
    print([i for i in Grid(test=[1, 2, 3], test2=[5, 6, 7])])
