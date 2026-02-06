# Visualization

Plotting and export scripts.

- `boundary_figures.py`: boundary surface figures.
- `profiles_volume_figures.py`: profiles + volume sanity figures.
- `vtk_field_and_fieldlines.py`: VTK export for ParaView (fields and fieldlines).
- `n3are_showcase.py`: high-resolution n3are plots (cross-sections, |B|, iota, 3D surface). Uses the high-level driver API.

Plotting scripts default to **showing** figures; pass `--save` to write outputs.

These scripts may require optional dependencies (e.g. `matplotlib` or `pyvista`).
