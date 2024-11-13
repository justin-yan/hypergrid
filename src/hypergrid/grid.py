from __future__ import annotations

import itertools
from collections import OrderedDict, namedtuple
from typing import Any, Callable, List, Protocol


class IGrid(Protocol):
    def __repr__(self) -> str: ...

    def __str__(self) -> str: ...

    def __iter__(self) -> Any: ...

    def __add__(self, other: IGrid) -> SumGrid:
        return SumGrid(self, other)

    def __mul__(self, other: IGrid) -> ProductGrid:
        return ProductGrid(self, other)

    def __and__(self, other: IGrid) -> CoGrid:
        return CoGrid(self, other)

    def apply(self, **kwargs: Callable[[Any], Any]) -> IGrid:
        # TODO: what should we do if no lambda is provided?
        # TODO: what should the expected signature of the lambda be?  Should we unpack the namedtuple or receive a namedtuple instead?
        applied_grids = []

        for dim_name, transform in kwargs.items():
            transformed_elements = [transform(grid_element) for grid_element in self]
            new_grid = Grid(**{dim_name: transformed_elements})
            applied_grids.append(new_grid)

        final_co_grid: IGrid = applied_grids[0]
        for grid in applied_grids[1:]:
            final_co_grid = CoGrid(final_co_grid, grid)

        return final_co_grid

    def select(self, *dim_names: str) -> IGrid:
        # TODO: what do we do if no dimnames are provided?
        selected_elements: dict = {dim_name: [] for dim_name in dim_names}

        for grid_element in self:
            for dim_name in dim_names:
                # TODO: How do we want to handle the case when a value isn't present?
                selected_value = getattr(grid_element, dim_name)
                selected_elements[dim_name].append(selected_value)

        selected_grids = [Grid(**{dim_name: values}) for dim_name, values in selected_elements.items()]

        final_co_grid: IGrid = selected_grids[0]
        for grid in selected_grids[1:]:
            final_co_grid = CoGrid(final_co_grid, grid)

        return final_co_grid

    def map(self, **kwargs: Callable[[Any], Any]) -> MapGrid:
        return MapGrid(self, **kwargs)

    def filter(self, predicate: Callable[[Any], bool]) -> FilterGrid:
        return FilterGrid(self, predicate)


class SumGrid(IGrid):
    def __init__(self, grid1: IGrid, grid2: IGrid) -> None:
        self.grid1 = grid1
        self.grid2 = grid2

    def __repr__(self) -> str:
        return f"SumGrid({repr(self.grid1)}, {repr(self.grid2)})"

    def __str__(self) -> str:
        return f"SumGrid({str(self.grid1)}, {str(self.grid2)})"

    def __iter__(self) -> Any:
        for grid_element in itertools.chain(self.grid1, self.grid2):
            yield grid_element


class ProductGrid(IGrid):
    # TODO: Make product grids fail if subdimensions collide?  How does this work with things like maps, etc.?
    def __init__(self, grid1: IGrid, grid2: IGrid) -> None:
        self.grid1 = grid1
        self.grid2 = grid2
        self.namedtuple_cache: dict = {}

    def __repr__(self) -> str:
        return f"ProductGrid({repr(self.grid1)}, {repr(self.grid2)})"

    def __str__(self) -> str:
        return f"ProductGrid({str(self.grid1)}, {str(self.grid2)})"

    def __iter__(self) -> Any:
        # TODO: is there a way to do this without accessing private attributes?
        for grid_element1, grid_element2 in itertools.product(self.grid1, self.grid2):
            field_names1 = grid_element1._fields
            field_names2 = grid_element2._fields
            concatenated_field_names = list(field_names1) + list(field_names2)
            field_names_key = tuple(concatenated_field_names)

            if field_names_key not in self.namedtuple_cache:
                self.namedtuple_cache[field_names_key] = namedtuple("GridElement", concatenated_field_names)

            concatenated_namedtuple_class = self.namedtuple_cache[field_names_key]
            concatenated_element = concatenated_namedtuple_class(*(grid_element1 + grid_element2))
            yield concatenated_element


class Grid(IGrid):
    dimensions: OrderedDict[str, List[Any]]

    def __init__(self, **kwargs: List[Any]) -> None:
        self.dimensions = OrderedDict()
        for dim, values in kwargs.items():
            self.dimensions[dim] = values

    def __repr__(self) -> str:
        dim_str = ", ".join([f"{dim}={values}" for dim, values in self.dimensions.items()])
        return f"Grid({dim_str})"

    def __str__(self) -> str:
        return self.__repr__()

    def __iter__(self) -> Any:
        fieldnames: tuple = tuple(self.dimensions.keys())
        namedtuple_class = namedtuple("GridElement", fieldnames)  # type: ignore
        for element_tuple in itertools.product(*self.dimensions.values()):
            yield namedtuple_class(*element_tuple)


class GridShapeMismatchError(Exception):
    pass


class CoGrid(IGrid):
    def __init__(self, grid1: IGrid, grid2: IGrid) -> None:
        if len(list(grid1)) == len(list(grid2)):
            self.grid1 = grid1
            self.grid2 = grid2
            self.namedtuple_cache: dict = {}
        else:
            raise GridShapeMismatchError("The shapes of the input grids must be the same")

    def __repr__(self) -> str:
        return f"CoGrid({repr(self.grid1)}, {repr(self.grid2)})"

    def __str__(self) -> str:
        return f"CoGrid({str(self.grid1)}, {str(self.grid2)})"

    def __iter__(self) -> Any:
        for grid_element1, grid_element2 in zip(self.grid1, self.grid2):
            field_names1 = grid_element1._fields
            field_names2 = grid_element2._fields
            concatenated_field_names = list(field_names1) + list(field_names2)
            field_names_key = tuple(concatenated_field_names)

            if field_names_key not in self.namedtuple_cache:
                self.namedtuple_cache[field_names_key] = namedtuple("GridElement", concatenated_field_names)

            concatenated_namedtuple_class = self.namedtuple_cache[field_names_key]
            concatenated_element = concatenated_namedtuple_class(*(grid_element1 + grid_element2))
            yield concatenated_element


class MapGrid(IGrid):
    def __init__(self, grid: IGrid, **kwargs: Callable[[Any], Any]) -> None:
        self.grid = grid
        self.dimension_mapping = kwargs
        self.namedtuple_cache: dict = {}

    def __repr__(self) -> str:
        mappings_str = ", ".join([f"{dim_name}={func.__name__}" for dim_name, func in self.dimension_mapping.items()])
        return f"MapGrid({repr(self.grid)}, {mappings_str})"

    def __str__(self) -> str:
        return f"MapGrid({str(self.grid)}, {', '.join(self.dimension_mapping.keys())})"

    def __iter__(self) -> Any:
        for grid_element in self.grid:
            new_values = {dim_name: func(grid_element) for dim_name, func in self.dimension_mapping.items()}
            concatenated_values = tuple(grid_element) + tuple(new_values.values())
            field_names = grid_element._fields + tuple(new_values.keys())

            field_names_key = tuple(field_names)
            if field_names_key not in self.namedtuple_cache:
                self.namedtuple_cache[field_names_key] = namedtuple("GridElement", field_names)

            concatenated_namedtuple_class = self.namedtuple_cache[field_names_key]
            concatenated_element = concatenated_namedtuple_class(*concatenated_values)
            yield concatenated_element


class FilterGrid(IGrid):
    def __init__(self, grid: IGrid, predicate: Callable[[Any], bool]) -> None:
        self.grid = grid
        self.predicate = predicate

    def __repr__(self) -> str:
        return f"FilterGrid({repr(self.grid)}, {self.predicate.__name__})"

    def __str__(self) -> str:
        return f"FilterGrid({str(self.grid)}, {self.predicate.__name__})"

    def __iter__(self) -> Any:
        for grid_element in self.grid:
            if self.predicate(grid_element):
                yield grid_element
