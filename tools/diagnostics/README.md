# Diagnostics (Developer-Only)

This folder contains parity breakdown scripts, investigation notebooks-as-scripts,
and research utilities used during the VMEC2000/VMEC++ parity push.

These scripts are intentionally not part of the stable, user-facing examples:

- they may rely on optional external installs (VMEC2000, VMEC++, simsopt),
- they may be slow or produce large reports/figures,
- their CLI/API may change without notice.

For user-facing entrypoints, start from:

- `examples/showcase_axisym_input_to_wout.py`
- `vmec_jax.api`

