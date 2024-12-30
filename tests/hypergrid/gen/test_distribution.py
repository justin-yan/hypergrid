from itertools import islice

from hypergrid.gen.distribution import Uniform
from hypergrid.grid import HyperGrid


def test_distribution_lambda():
    ud = Uniform(low=1, high=10)
    g = HyperGrid(test=range(100))
    g = g.map(result=ud)
    assert all([1 <= rn.result <= 10 for rn in g])


def test_uniform_distribution():
    ud = Uniform(low=1, high=10)

    assert all([1 <= rn <= 10 for rn in islice(ud, 100)])
