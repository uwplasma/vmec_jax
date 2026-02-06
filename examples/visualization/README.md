# Visualization

Plotting and export scripts.

- `boundary_figures.py`: boundary surface figures.
- `profiles_volume_figures.py`: profiles + volume sanity figures.
- `vtk_field_and_fieldlines.py`: VTK export for ParaView (fields and fieldlines).
- `n3are_showcase.py`: high-resolution n3are plots (cross-sections, |B|, iota, 3D surface). Uses the high-level driver API.
- `n3are_vmec2000_vs_vmecjax.py`: side-by-side VMEC2000 vs vmec_jax figures for n3are (cross-sections, 3D, |B|, profiles).

Plotting scripts default to **showing** figures; pass `--save` to write outputs.

These scripts may require optional dependencies (e.g. `matplotlib` or `pyvista`).
