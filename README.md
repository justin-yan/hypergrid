# hypergrid

Hypergrid enables concise declaration and manipulation of parameter grid spaces, with an aim towards use cases such as hyperparameter tuning or defining large batch jobs.

Use the following features to lazily declare a parameter grid:

- Dimension and Grid direct instantiation
- `+` and `|` for "sum" or "union" types (concatenation)
- `*` for "product" types
- `&` for coiteration (zip)
- `select` to project dimensions by name

There are also a few transformations that can be lazily applied element-wise, which take a GridElement (a namedtuple of dimension<->value) as input.

- `filter` to apply boolean predicate
- `map` for lambda transformation
- `map_to` for map + concat

Once a parameter grid is declared, there are two ways to "materialize" your grid, which return GridElements.

- `__iter__`: a grid is directly iterable
- `sample`: allows you to sample from the grid according to a sampling strategy

## Usage Examples

```python
from hypergrid.dsl import *
from dataclasses import dataclass

# First, we need to create a Dimension, which is essentially a named, finite, 1-d collection
d = Dimension(custom_name=[1, 2, 3])        # any python Collection will work - set, dict, range(), etc.
assert d.name == "custom_name"              # the argument name is used as the dimension's name.
d.with_name("ints")                         # which you can reset
Uniform(low=1, high=5).take(5)              
ExponentialStep(start=1, step=1.1).take(6)  # You can also take a dimension from a Distribution or HIterable

# You can `len(d)` or `[i for i in d]`, but grids are more interesting
g = d.to_grid()
i2d = Dimension(ints=[4, 5, 6])
cd = Dimension(chars=["a", "b", "c", "d"])
union_ints = g + i2d     # result is length 6: Concatenate two grids that have the same underlying dimensions
product_g = g * cd       # result is length 12: tuples (1, "a") - take the cartesian product of the underlying grids
zip_g = g & cd           # result is length 3: tuples (1, "a") - zip two grids together, up to the shorter grid

ml = [i for i in zip_g]  # You can iterate through a grid
tl = zip_g.take(5)       # or you can just take up to a certain number of grid elements from it
print(tl[0].ints), print(tl[0].chars)  # The iterator elements are python NamedTuples taken from the dimension names.

# These gridelements can be referenced and used in the grid higher-order functions
zip_g.filter(lambda ge: ge.chars in ["a", "b"])    # result is length 2: keep the tuples (1, "a") and (2, "b")
zip_g.map(doubled=lambda ge: ge.ints * 2)          # result is length 3, with single attribute (drops `ints` and `chars`)
mt = zip_g.map_to(doubled=lambda ge: ge.ints * 2)  # result is length 3, appends `doubled` and keeps `ints` and `chars`
print(mt.select("doubled", "ints").take(1)[0])     # resulting gridelement no longer has `chars`

# There are some other utility methods on a grid:
zip_g.sample()                                     # Randomly samples a single grid element from a grid
zip_g.to_sklearn()                                 # The Grid.to_* methods convert HyperGrids to other grid formats

# The general idea is to allow for fairly extensive grid construction routines
@dataclass
class FakeModel:
    idx: int
    param1: float

g = HyperGrid(  # A grid with 4 x 10 combinations
    ExponentialStep(start=1.0, step=1.5).take(4).with_name("param1"),
    idx=range(10)
).instantiate(model=FakeModel).select("model") + \
    HyperGrid(  # A different grid with 15 combinations
        Uniform(low=-1, high=1).take(5).with_name("param1"),
        idx=[10, 20, 30]        
    ).instantiate(model=FakeModel).select("model")
assert len(g) == 55
g.sample()
```

