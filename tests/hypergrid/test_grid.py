from typing import cast

import pytest

from hypergrid.grid import Grid


def test_base_construction():
    g1 = Grid(example=[1, 2, 3], example2=["a", "b", "c"])
    print([i for i in g1])
    assert len(list(g1)) == 9
    assert list(g1)[0].example == 1
    assert list(g1)[0].example2 == "a"


def test_grid_sums():
    g1 = Grid(example=[1, 2, 3])
    g2 = Grid(example=[4, 5, 6])
    assert len(list(g1 + g2)) == 6
    assert len(list(g1 | g2)) == 6


def test_grid_products():
    base_g = Grid(example=[1, 2, 3], example2=["a", "b", "c"])
    g1 = Grid(example=[1, 2, 3])
    g2 = Grid(example2=["a", "b", "c"])
    assert len(list(g1 * g2)) == 9
    assert list(g1 * g2) == list(base_g)


def test_zipgrids():
    g1 = Grid(example=[1, 2, 3])
    g2 = Grid(example2=["a", "b", "c"])
    assert len(list(g1 & g2)) == 3
    assert list(g1 & g2)[0].example == 1
    assert list(g1 & g2)[0].example2 == "a"


def test_filtergrids():
    g1 = Grid(example=[1, 2, 3])
    fg = g1.filter(lambda e: cast(int, e.example) % 2 == 0)  # https://github.com/python/mypy/issues/5697#issuecomment-425738017
    assert len(list(fg)) == 1
    assert list(fg)[0].example == 2


def test_selectgrids():
    g1 = Grid(example=[1, 2, 3], example2=[2, 4, 6])
    selected = g1.select("example2")
    print([i for i in selected])
    assert len(list(selected)) == 9
    assert list(selected)[0].example2 == 2


def test_mapgrids():
    g1 = Grid(example=[1, 2, 3])
    mg = g1.map(example2=lambda e: e.example * 2)
    assert len(list(mg)) == 3
    assert list(mg)[0].example2 == 2
    with pytest.raises(AttributeError):
        list(mg)[0].example

    g1 = Grid(example=[1, 2, 3])
    mg = g1.map_to(example2=lambda e: e.example * 2)
    assert len(list(mg)) == 3
    assert list(mg)[0].example == 1
    assert list(mg)[0].example2 == 2
