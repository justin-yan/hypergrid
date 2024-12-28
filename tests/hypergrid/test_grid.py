import operator
from functools import reduce
from math import prod
from typing import cast

import pytest
from hypothesis import given
from hypothesis import strategies as st
from hypothesis.strategies import DrawFn, composite

from hypergrid.grid import Grid


@composite
def het_typed_lists(draw: DrawFn):
    retlists = []
    for _ in range(draw(st.integers(min_value=1, max_value=5))):
        element_strategy = draw(
            st.sampled_from(
                [
                    st.integers(),
                    st.text(),
                    st.floats(allow_infinity=False, allow_nan=False),
                    st.booleans(),
                ]
            )
        )
        retlists.append(draw(st.lists(element_strategy)))
    return retlists


@composite
def hom_typed_lists(draw: DrawFn):
    element_strategy = draw(
        st.sampled_from(
            [
                st.integers(),
                st.text(),
                st.floats(allow_infinity=False, allow_nan=False),
                st.booleans(),
            ]
        )
    )
    num_lists = draw(st.integers(min_value=1, max_value=5))
    return [draw(st.lists(element_strategy)) for _ in range(num_lists)]


@given(het_typed_lists())
def test_base_construction(lists):
    g1 = Grid(**{f"example{i}": v for i, v in enumerate(lists)})
    manifested_list = list(g1)
    length = len(manifested_list)
    assert length == prod([len(hlist) for hlist in lists])
    if length > 0:
        manifested_list[0].example0 == lists[0][0]


@given(hom_typed_lists())
def test_grid_sums(lists):
    gs = [Grid(example=hlist) for hlist in lists]
    assert len([e for e in reduce(operator.add, gs[1:], gs[0])]) == sum([len(hlist) for hlist in lists])
    assert len([e for e in reduce(operator.or_, gs[1:], gs[0])]) == sum([len(hlist) for hlist in lists])


@given(het_typed_lists())
def test_grid_products(lists):
    gs = [Grid(**{f"example{i}": hlist}) for i, hlist in enumerate(lists)]
    manifested_list = [e for e in reduce(operator.mul, gs[1:], gs[0])]
    length = len(manifested_list)
    assert length == prod([len(hlist) for hlist in lists])
    if length > 0:
        rank = len(lists)
        for i in range(rank):
            assert getattr(manifested_list[0], f"example{i}") == lists[i][0]


@given(het_typed_lists())
def test_zipgrids(lists):
    gs = [Grid(**{f"example{i}": hlist}) for i, hlist in enumerate(lists)]
    manifested_list = [e for e in reduce(operator.and_, gs[1:], gs[0])]
    length = len(manifested_list)
    assert length == min([len(hlist) for hlist in lists])
    if length > 0:
        rank = len(lists)
        for i in range(rank):
            assert getattr(manifested_list[0], f"example{i}") == lists[i][0]


@given(st.lists(st.integers()))
def test_filtergrids(hlist):
    g1 = Grid(example=hlist)
    fg = g1.filter(lambda e: cast(int, e.example) % 10 == 0)
    assert all([i.example % 10 == 0 for i in fg])


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
