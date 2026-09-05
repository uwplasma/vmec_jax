# Contributing to VMEX

The contributor guide is `docs/project/contributing.rst`, rendered at
https://vmex.readthedocs.io/en/latest/project/contributing.html. It covers
the development install, the test manifest, reference assets, the CI lanes,
GPU CI, and the release procedure. Three rules apply to every change:

1. Run `python tools/preflight.py` before pushing. It runs the static gates,
   the guard tests, and the tests affected by your diff. One CI attempt takes
   25-45 minutes, so a failure caught locally saves a full attempt.
2. Changed executable lines must be at least 95% covered. CI enforces this
   with `diff-cover` against `origin/main`; preflight runs the same check
   when `diff-cover` is installed.
3. Do not change the VMEC physics schema (input fields, wout variables, or
   the equations a module ports) without a parity test against the VMEC2000
   golden fixtures under `tests/`.

Report problems at https://github.com/uwplasma/vmex/issues; include the input
file and the output of `vmex --doctor`. Participation is governed by
`CODE_OF_CONDUCT.md`.
