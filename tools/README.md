# Tools

This directory contains developer-facing tools, not end-user examples.

- `fetch_assets.py`: downloads optional validation assets and verifies their
  byte size and SHA-256 before extraction. `assets/manifest.json` records the
  release URL, provenance, license, generator revision, and installed paths.
  Repository fixtures are the default; VMEC2000 goldens use the user cache:

  ```console
  python tools/fetch_assets.py
  python tools/fetch_assets.py --bundle golden-v1
  ```

- `profile_hotpaths.py`: cold-vs-warm wall-time + peak-RSS profile of the
  production hot paths (fixed-boundary solve and the differentiable
  `value_and_grad` adjoint). Backend-agnostic — the same script produces the
  CPU and GPU numbers with `--device cpu` / `--device gpu`.

Hardware parity across forward solves and boundary gradients is audited by
`benchmarks/device_parity.py`; use `--quick` for its reduced-grid smoke mode.

- `force_oracle.py`: staged force oracle along the `funct3d.f` chain. Replays
  the production iteration body and records staged internals (state/axis,
  geometry/Jacobian, bcovar fields, real-space and spectral lambda force,
  `scalxc`, `fnormL`, raw/normalized `fsql`, `faclam`, final update direction)
  at chosen iterations, straddling the `ns4 = 25` preconditioner refresh and
  the first iteration after a Jacobian retry. `record`/`check` pin the
  VMEX-only stages against recorded goldens; `cross` compares the printed
  iteration rows against a local `xvmec2000` (the only cross-code channel).
  The comparison fails at the FIRST differing chain stage; the default output
  is values-free (stage codes + PASS/FAIL) and safe for confidential decks,
  while `--details` prints values and must never be shared for one.

- `diagnose_input.py`: runs the first VMEC force evaluation without entering
  the nonlinear iteration. Its default shareable report contains only runtime
  information, pass/fail checks for field assembly, force normalization,
  Fourier projection, and preconditioning, plus a diagnostic code. It omits
  the input path and all input-derived values. `--details` exposes parsed
  controls and numerical values for local use with non-confidential decks only:
  `python tools/diagnose_input.py path/to/input.case`.

Tools may write to ignored `outputs/` or a user-selected scratch directory.
They should not write tracked artifacts unless the command is explicitly a
documentation or release-artifact promotion step.
