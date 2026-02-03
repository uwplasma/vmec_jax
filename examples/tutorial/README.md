# Tutorial (00–09)

These scripts are intended to be read/run in order. They provide a guided path from
input parsing and geometry reconstruction to simple fixed-boundary solves.

Run from the repo root:

```bash
python examples/tutorial/00_parse_and_boundary.py examples/data/input.circular_tokamak --out boundary_step0.npz --verbose
python examples/tutorial/04_geom_metrics.py examples/data/input.circular_tokamak --out geom_step2.npz --verbose
python examples/tutorial/08_solve_fixed_boundary.py examples/data/input.circular_tokamak --verbose
```

Outputs are typically `.npz` files suitable for quick plotting or regression comparisons.

