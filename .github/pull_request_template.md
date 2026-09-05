# What this changes

<!-- One paragraph: the behaviour before, the behaviour after, and why. Link
     the issue it closes. -->

# Evidence

<!-- The commands you ran and their outcome. For a physics change, the parity
     numbers against the VMEC2000 golden fixtures; for a performance change,
     the before/after timings and the machine they came from. -->

```console
python tools/preflight.py
```

# Checklist

- [ ] `python tools/preflight.py` passes locally. One CI attempt takes 25-45
      minutes, so a failure caught here saves a full attempt.
- [ ] Changed executable lines are at least 95% covered (`diff-cover` against
      `origin/main`, the same check CI runs).
- [ ] No VMEC physics-schema change (input fields, `wout` variables, or the
      equations a module ports) without a parity test against the VMEC2000
      golden fixtures under `tests/`.
- [ ] New or changed public behaviour is documented under `docs/`, and
      `CHANGELOG.md` records it.
- [ ] Claims added to the README or the docs are backed by a committed
      artifact or a test, not by a remembered number.

The contributor guide is `CONTRIBUTING.md` and
`docs/project/contributing.rst`. Participation is governed by
`CODE_OF_CONDUCT.md`.
