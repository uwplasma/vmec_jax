# Run a VMEC input file

`vmex input.X` solves any VMEC2000 `&INDATA` namelist or structured-JSON
deck and writes `wout_X.nc`, matching `xvmec2000`'s console output and
output-file conventions.

## Namelist or JSON

Both formats route through the same {class}`~vmex.core.input.VmecInput`:

```console
vmex input.circular_tokamak      # classic &INDATA namelist
vmex circular_tokamak.json       # structured JSON deck
```

`examples/run_from_json.py` converts a deck to JSON, reads it back, and
confirms the two representations describe one equilibrium:

```{literalinclude} ../../examples/run_from_json.py
:language: python
```

Every input key, its default, and its VMEC2000 semantics:
{doc}`/reference/input-file`. Which flags of a VMEC2000 deck are honored,
approximated, or rejected: {doc}`/reference/vmec2000-compatibility`.

## Control where output goes

```console
vmex input.circular_tokamak --outdir results/   # wout + figures into results/
vmex input.circular_tokamak --quiet             # silence the iteration table
```

Default output location is alongside the input file.

## Override convergence from the command line

```console
vmex input.circular_tokamak --ftol 1e-10       # final-stage FTOL_ARRAY
vmex input.circular_tokamak --max-iter 2000    # final-stage NITER_ARRAY cap
```

Both override only the final multigrid stage; earlier `NS_ARRAY` stages keep
their deck entries.

## Check the exit code

VMEX has a zero-crash policy: every failure maps to a typed
{class}`vmex.core.errors.VmecError`, printed as the VMEC2000 `werror`
message plus a one-line hint, and the process exits with the matching
`ier_flag` (0 success, 2 for "MORE ITERATIONS REQUIRED", ...). An
NITER-exhausted run still writes its unconverged wout through the normal
output path (`fileout.f` semantics); fatal numerical/Jacobian failures never
produce a wout. Scripting pattern:

```console
vmex input.case --quiet
case $? in
  0) echo converged ;;
  2) echo "needs more iterations" ;;
  *) echo "failed - see stderr" ;;
esac
```

Diagnosing the failures themselves is {doc}`troubleshoot`.

## The full flag set

`--plot`, `--booz`, `--scale`, `--device`, `--mode`, `--restart`,
`--jacobian-retries`, `--coils`, and the rest are tabulated in
{doc}`/reference/cli`.
