# hypergrid

Hypergrid enables concise declaration and manipulation of parameter grid spaces, with an aim towards use cases such as hyperparameter tuning or defining large batch jobs.

Use the following features to lazily declare a parameter grid:

- Dimension and Grid direct instantiation.
- `+` and `|` for "sum" or "union" types (concatenation)
- `*` for "product" types
- `&` for coiteration (zip)
- `filter` to apply boolean predicate
- `select` to project dimensions
- `map` for lambda transformation
- `map_to` for map + concat

Once a parameter grid is declared, there are two ways to "materialize" your grid:

- `__iter__`: a grid is directly iterable
- `sample`: allows you to sample from the grid according to a sampling strategy
