from hypergrid.dsl import HGrid


def test_basic_grid_sklearn_conversion():
    g = HGrid(test=[1, 2, 3])
    g2 = HGrid(test2=[4, 5, 6])
    pg = (g * g2).to_sklearn()
    assert set(pg.param_grid[0].keys()) == {"test", "test2"}
    assert len([i for i in pg]) == 9
