import pytest

from hypergrid.dimension import Dimension


def test_base_dimension_construct():
    dim = Dimension(test=[1, 2, 3])
    assert len(dim.values) == 3
    assert dim.name == "test"

    dim = Dimension(test=range(4))
    assert len(dim.values) == 4

    with pytest.raises(AssertionError):
        Dimension(test1=[1, 2, 3], test2=[1, 2, 3])


def test_base_dimension_iter():
    dim = Dimension(test=[1, 2, 3])
    assert len([i for i in dim]) == 3
