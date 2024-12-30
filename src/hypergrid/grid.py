from __future__ import annotations

import itertools
import random
from collections import namedtuple
from collections.abc import Collection
from functools import cached_property
from math import prod
from typing import TYPE_CHECKING, Any, Callable, Iterator, Optional, Protocol, Type, runtime_checkable

from hypergrid.gen.iterable import HIterable
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

    def __len__(self) -> int: ...

    def __iter__(self) -> Iterator: ...

    def sample(self) -> tuple: ...

    def __add__(self, other: Grid | Dimension | RawDimension) -> SumGrid:
        match other:
            case Grid():
                return SumGrid(self, other)
            case Dimension():
                return SumGrid(self, HyperGrid(other))
            case (str(s), coll) if isinstance(coll, Collection):  # RawDimension
                return SumGrid(self, HyperGrid(Dimension(**{s: coll})))
            case _:
                raise ValueError("Invalid argument for grid operation")

    def __or__(self, other: Grid | Dimension | RawDimension) -> SumGrid:
        return self.__add__(other)

    def __mul__(self, other: Grid | Dimension | RawDimension) -> ProductGrid:
        match other:
            case Grid():
                return ProductGrid(self, other)
            case Dimension():
                return ProductGrid(self, HyperGrid(other))
            case (str(s), coll) if isinstance(coll, Collection):  # RawDimension
                return ProductGrid(self, HyperGrid(Dimension(**{s: coll})))
            case _:
                raise ValueError("Invalid argument for grid operation")

    def __and__(self, other: Grid | Dimension | HIterable | RawDimension) -> ZipGrid:
        match other:
            case Grid():
                return ZipGrid(self, other)
            case Dimension():
                return ZipGrid(self, HyperGrid(other))
            case (str(s), coll) if isinstance(coll, Collection):  # RawDimension
                return ZipGrid(self, HyperGrid(Dimension(**{s: coll})))
            case HIterable():
                return ZipGrid(self, HyperGrid(other.take(len(self))))
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


class HyperGrid(Grid):
    dimensions: list[Dimension]

    def __init__(self, *args: Dimension, **kwargs: Collection) -> None:
        dims = list(args)
        for dim, values in kwargs.items():
            dims.append(Dimension(**{dim: values}))
        assert len(dims) > 0, "Must provide at least one meaningful dimension"
        assert len(dims) == len(set(dims)), "Dimension names must be unique"
        self.dimensions = dims
        self.grid_element = namedtuple("GridElement", [dim.name for dim in self.dimensions])  # type: ignore[misc]

    def __repr__(self) -> str:
        dim_str = ", ".join([repr(dim) for dim in self.dimensions])
        return f"HyperGrid({dim_str})"

    def __len__(self) -> int:
        return prod([len(dim) for dim in self.dimensions])

    def __iter__(self) -> Iterator:
        for element_tuple in itertools.product(*[dim.__iter__() for dim in self.dimensions]):
            yield self.grid_element(*element_tuple)

    def sample(self) -> tuple:
        return self.grid_element(*tuple([dim.sample() for dim in self.dimensions]))


class SumGrid(Grid):
    def __init__(self, grid1: Grid, grid2: Grid) -> None:
        assert set(grid1.dimension_names) == set(grid2.dimension_names)
        self.grid1 = grid1
        self.grid2 = grid2
        self.grid_element = grid1.grid_element

    def __repr__(self) -> str:
        return f"SumGrid({repr(self.grid1)}, {repr(self.grid2)})"

    def __len__(self) -> int:
        return len(self.grid1) + len(self.grid2)

    def __iter__(self) -> Iterator:
        for grid_element in itertools.chain(self.grid1, self.grid2):
            yield grid_element

    def sample(self) -> tuple:
        return random.choice([ge for ge in self])


class ProductGrid(Grid):
    def __init__(self, grid1: Grid, grid2: Grid) -> None:
        assert set(grid1.dimension_names).isdisjoint(set(grid2.dimension_names)), "Dimensions must be exactly matching"
        self.grid1 = grid1
        self.grid2 = grid2
        self.grid_element = namedtuple("GridElement", grid1.dimension_names + grid2.dimension_names)  # type: ignore[misc]

    def __repr__(self) -> str:
        return f"ProductGrid({repr(self.grid1)}, {repr(self.grid2)})"

    def __len__(self) -> int:
        return len(self.grid1) * len(self.grid2)

    def __iter__(self) -> Iterator:
        for grid_element1, grid_element2 in itertools.product(self.grid1, self.grid2):
            yield self.grid_element(*(grid_element1 + grid_element2))

    def sample(self) -> tuple:
        ge1 = self.grid1.sample()
        ge2 = self.grid2.sample()
        return self.grid_element(*(ge1 + ge2))


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

    def __len__(self) -> int:
        return min(len(self.grid1), len(self.grid2))

    def __iter__(self) -> Iterator:
        for grid_element1, grid_element2 in zip(self.grid1, self.grid2):
            yield self.grid_element(*(grid_element1 + grid_element2))

    def sample(self) -> tuple:
        return random.choice([ge for ge in self])


class FilterGrid(Grid):
    _iter_cache: Optional[list] = None

    def __init__(self, grid: Grid, predicate: Callable[[Any], bool]) -> None:
        self.grid = grid
        self.predicate = predicate
        self.grid_element = grid.grid_element

    def __repr__(self) -> str:
        return f"FilterGrid({repr(self.grid)}, {self.predicate.__name__})"

    def __len__(self) -> int:
        return self._len

    @cached_property
    def _len(self) -> int:
        return len([x for x in self])

    def __iter__(self) -> Iterator:
        for grid_element in self.grid:
            if self.predicate(grid_element):
                yield grid_element

    def sample(self) -> tuple:
        if self._iter_cache is not None:
            sublist = self._iter_cache
        else:
            sublist = [ge for ge in self]
            self._iter_cache = sublist
        return random.choice(sublist)


class SelectGrid(Grid):
    def __init__(self, grid: Grid, *select_dims: str) -> None:
        assert len(set(select_dims)) == len(select_dims), "Selected columns must all be unique"
        assert set(select_dims) <= set(grid.dimension_names), "Selected dimensions must be subset of grid dimensions"
        self.grid = grid
        self.select_dims = select_dims
        self.grid_element = namedtuple("GridElement", [name for name in grid.dimension_names if name in self.select_dims])  # type: ignore[misc]

    def __repr__(self) -> str:
        return f"SelectGrid({repr(self.grid)}, {repr(self.select_dims)})"

    def __len__(self) -> int:
        return len(self.grid)

    def __iter__(self) -> Iterator:
        for grid_element in self.grid:
            yield self.grid_element(*self._process_single(grid_element))

    def sample(self) -> tuple:
        return self.grid_element(*self._process_single(self.grid.sample()))

    def _process_single(self, ge: tuple) -> list:
        element_list = []
        for dim_name in self.dimension_names:
            try:
                selected_value = getattr(ge, dim_name)
            except AttributeError:
                selected_value = None
            element_list.append(selected_value)
        return element_list


class MapGrid(Grid):
    def __init__(self, grid: Grid, **kwargs: Callable[[Any], Any]) -> None:
        assert len(set(kwargs.keys())) == len(kwargs.keys()), "New columns must all have unique names"
        self.grid = grid
        self.dimension_mapping = kwargs
        self.grid_element = namedtuple("GridElement", list(kwargs.keys()))  # type: ignore[misc]

    def __repr__(self) -> str:
        mappings_str = ", ".join([f"{dim_name}={func.__name__}" for dim_name, func in self.dimension_mapping.items()])
        return f"MapGrid({repr(self.grid)}, {mappings_str})"

    def __len__(self) -> int:
        return len(self.grid)

    def __iter__(self) -> Iterator:
        for grid_element in self.grid:
            yield self.grid_element(**self._process_single(grid_element))

    def sample(self) -> tuple:
        return self.grid_element(**self._process_single(self.grid.sample()))

    def _process_single(self, ge: tuple) -> dict:
        return {dim_name: func(ge) for dim_name, func in self.dimension_mapping.items()}


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

    def __len__(self) -> int:
        return len(self.grid)

    def __iter__(self) -> Iterator:
        for grid_element in self.grid:
            yield self.grid_element(**self._process_single(grid_element))

    def sample(self) -> tuple:
        return self.grid_element(**self._process_single(self.grid.sample()))

    def _process_single(self, ge: tuple) -> dict:
        new_values = {dim_name: func(ge) for dim_name, func in self.dimension_mapping.items()}
        return ge._asdict() | new_values  # type: ignore
