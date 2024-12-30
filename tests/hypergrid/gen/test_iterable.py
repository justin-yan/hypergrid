import math
from itertools import islice

from hypergrid.gen.iterable import ExponentialStep
from hypergrid.grid import HyperGrid


def test_iterable_zip():
    es = ExponentialStep(start=1, step=1.1)
    g = HyperGrid(test=range(100))
    g = g & es
    manifested_list = [ge for ge in g]
    assert len(g) == 100 == len(manifested_list)
    assert math.isclose(manifested_list[-1].anonymous, 1.1**99)
    es.with_name("es")
    g = HyperGrid(test=range(100))
    g = g & es
    assert all([1 <= ge.es for ge in g])


def test_exponential_sequence():
    es = ExponentialStep(start=1, step=1.1)

    assert all([math.isclose(p[0], p[1]) for p in zip(islice(es, 5), [1, 1.1, 1.21, 1.331, 1.4641])])
    assert es.take(5).name == "anonymous"
    assert es.with_name("test").take(5).name == "test"
