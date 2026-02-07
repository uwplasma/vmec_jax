# Bundled data

This folder contains small VMEC input files (`input.*`) and corresponding
reference `wout_*.nc` files used in the test suite.

Some inputs are adapted from other VMEC implementations (e.g. VMEC++) but the
reference `wout_*_reference.nc` files are always generated using VMEC2000 so we
have a consistent parity target.

If you add a new case:

- include both the `input.*` and a matching `wout_*_reference.nc`,
- keep resolutions modest so CI remains fast,
- add/extend tests that validate the new functionality against the reference.
