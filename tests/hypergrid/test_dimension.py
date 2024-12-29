import itertools

import pytest

from hypergrid.dimension import Dimension, FDimension, IDimension


def test_fixed_dimension_construct():
    dim = FDimension(test=[1, 2, 3])
    assert len(dim.values) == 3
    assert dim.name == "test"

    dim = FDimension(test=range(4))
    assert len(dim.values) == 4

    with pytest.raises(AssertionError):
        FDimension(test1=[1, 2, 3], test2=[1, 2, 3])


def test_fixed_dimension_iter():
    dim = FDimension(test=[1, 2, 3])
    assert len([i for i in dim]) == 3


def test_dimension_factory():
    f = Dimension.make(test=[1, 2, 3])
    assert isinstance(f, FDimension)

    i = Dimension.make(test=itertools.count(start=0, step=1))
    assert isinstance(i, IDimension)
