import pytest

from hypergrid.dimension import FixedDimension


def test_fixed_dimension_construct():
    dim = FixedDimension(test=[1, 2, 3])
    assert len(dim.values) == 3
    assert dim.name == "test"

    dim = FixedDimension(test=range(4))
    assert len(dim.values) == 4

    with pytest.raises(AssertionError):
        FixedDimension(test1=[1, 2, 3], test2=[1, 2, 3])


def test_fixed_dimension_iter():
    dim = FixedDimension(test=[1, 2, 3])
    assert len([i for i in dim]) == 3
