try:
    from sklearn.model_selection import ParameterGrid
except ImportError:
    raise ImportError("If using sklearn conversion functionality, install hypergrid with `sklearn` extras via `pip install hypergrid[sklearn]`")

from hypergrid.grid import Grid, HyperGrid, ProductGrid


def _grid_to_sklearn(grid: Grid) -> ParameterGrid:  # type: ignore[no-any-unimported]
    """
    SKLearn's ParameterGrid accepts {str: sequence}

    Because these ParameterGrids don't directly compose, we use a recursive helper, and then convert the composable dicts
      into a ParameterGrid in this outer wrapper.
    """
    return ParameterGrid(_grid_to_sklearn_recursive_helper(grid))


def _grid_to_sklearn_recursive_helper(grid: Grid) -> dict:
    """
    SKLearn's param_grid dictionaries only support simple cartesian products, so only the Grid and ProductGrid elements
      are convertible to SKLearn parameter grids.
    """
    match grid:
        case HyperGrid():
            # TODO: will OOM on infinite iterators
            return {dim.name: [v for v in dim] for dim in grid.dimensions}
        case ProductGrid():
            d1 = _grid_to_sklearn_recursive_helper(grid.grid1)
            d2 = _grid_to_sklearn_recursive_helper(grid.grid2)
            return d1 | d2
        case _:
            raise ValueError("Converting Grid to SKLearn ParameterGrid is not compatible with")
