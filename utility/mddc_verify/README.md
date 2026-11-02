# mddc_verify: checks for SALMON MD, DC and RT runs

`mddc_verify` reads the output files of a finished SALMON run and says whether the run can be
trusted: it ended normally, converged, wrote only finite numbers, and obeys the physics that
the run must obey (energy conservation, energy balance under a laser, the integrator's own
identities, DC against a conventional ground state). It can also compare two runs, for
example a CPU and a GPU build, two MPI layouts or two SALMON versions, and say where they
first differ.

It was written to check GPU ports of Ehrenfest MD (`tddft_*` with `yn_md = 'y'`),
Born–Oppenheimer MD (`theory = 'dft_md'`) and divide-and-conquer DFT (`yn_dc = 'y'`), and it
also checks plain ground-state (GS) and real-time (RT) runs.

- **Standard library only.** The checks run on any Python ≥ 3.8, including `python3 -I`.
- **Figures are optional.** With matplotlib installed, `--figures DIR` writes PNG plots, each
  with a Markdown table of the plotted numbers next to it. Without matplotlib the figures are
  skipped and the verdict is unchanged.
- **NaN never passes.** Every comparison returns false when a value is NaN, ±Inf, a Fortran
  overflow field (`****`) or malformed text.
- **Missing evidence is never a pass.** It gives the verdict *unknown*.

## Usage

Run from this directory, or give the path to `verify.py`:

```sh
# one run: generic and physics checks; the mode is read from the inputfile unless given
python3 -I verify.py check RUNDIR [--mode auto|ehrenfest|dft_md|dc|gs|rt] [--gs-dir GSDIR] \
    [--json verdict.json] [--figures FIGDIR]

# DC, with its conventional reference, a buffer-size series and the RT follow-up runs
python3 -I verify.py check DC_RUN --conventional CONV_GS_RUN --buffer-dirs B10 B15 B20 \
    --rt-dc RT_FROM_DC --rt-conv RT_FROM_CONV

# a testsuite run directory: the developer's reference values are re-applied NaN-safely
python3 -I verify.py check build/testsuites/129_Si_dcdft_lcfo_chefsi --salmon-version develop

# two runs: candidate against reference
python3 -I verify.py compare CANDIDATE REFERENCE [--mode ...] [--policy exact|tolerance|both] \
    [--json cmp.json] [--figures FIGDIR]

# every tolerance, with what it applies to and why it has its value
python3 -I verify.py tolerances [--json]
```

The verdict is printed as one line per check and, with `--json`, written with all metrics.

| Exit status | Meaning |
|---|---|
| 0 | pass, or pass with warnings |
| 1 | at least one check failed |
| 2 | unknown: evidence is missing (for example no `salmon.exit`) |
| 3 | usage error |

Statuses of a single check: *pass*; *fail* (the predicate does not hold, or a value it needs is
not finite); *warn* (a softer limit is exceeded; never turns a fail into a pass); *unknown*
(the evidence is missing); *n/a* (the check does not apply, with the reason). The run's verdict
is the worst of them, in the order fail > unknown > warn > pass.

## What a run directory must contain

| File | Why |
|---|---|
| `inputfile` | the deck actually used: modes, `nt`, `dt`, output steps, thresholds, cell |
| `outputfile` | SALMON's full standard output, unfiltered (do not lower `ilevel_print`) |
| `salmon.exit` | SALMON's own exit status, one integer. With `salmon < inputfile \| tee outputfile` the shell reports `tee`'s status, so use `set -o pipefail` or `echo ${PIPESTATUS[0]} > salmon.exit`. Without this file the verdict is *unknown*. |
| `stderr.txt` | optional; scanned for runtime errors |
| `<sysname>_*.data`, `<sysname>_trj.xyz` | all text outputs of the run |
| DC: `data_dcdft/total/*`, `data_dcdft/fragments/*/<sysname>_info.data` | LCFO eigenvalues and the full-precision DC energy |

For Ehrenfest MD, keep a field-free stretch (`I_wcm2_1 = 0`, or `nt` beyond the end of the
pulse) if you want a strict energy-conservation check: under a laser only the weaker
energy-balance check applies (see below). Keep `out_rt_energy_step` and `out_rvf_rt_step`
the same in runs you want to compare.

## What it checks

**Generic (any run).**

| Check | Passes when |
|---|---|
| `exit_status` | `salmon.exit` exists and is 0 (missing: *unknown*) |
| `normal_termination` | `outputfile` contains `end SALMON` (the RT electron-count guard in `src/rt/dip.f90` ends with a plain `STOP`, which can exit with status 0) |
| `runtime_errors` | no Fortran, CUDA/OpenACC, MPI, signal or scheduler error text in `outputfile` or `stderr.txt` |
| `gs_convergence` | GS: `#GS converged at` with a residual below the threshold, and no `#GS does not converged`. DC: the last `DC #SCF` residual is below the threshold (DC mode prints no convergence message). dft_md: no `does not converged` |
| `files_present` | the mode's files exist and are not empty |
| `all_values_finite` | every number of every table, trajectory, eigenvalue file and parsed `outputfile` record is a finite, well-formed number |
| `table_integrity` | row and frame counts and the time grid match `nt`, `dt` and the output steps; no ragged or truncated lines |
| `timer_table` | the end-of-run timer table is present (NaN/Inf statistics in it only warn: they are cosmetic) |

**Ehrenfest MD and RT.** The conserved quantity is H = Eall + Tion (SALMON's `Eall` excludes
the ions' kinetic energy).

| Check | Passes when |
|---|---|
| `energy_conservation` | in the longest field-free window, max \|H(t) − H(t₀)\|/N and the drift per atom per ps are within limits. The t = 0 row is skipped when its `Tion` is SALMON's placeholder 0 (it is written before the ions' kinetic energy is computed). |
| `energy_balance` | under a laser, H(t) − H(0) equals the field work W(t) = Ω Σ (J_e − J_i)·ΔA computed from `<sys>_rt.data`, within the ambiguity of that sum (the difference between its left- and right-point forms) |
| `electron_number` | the electron count printed every output step stays at `nelec` |
| `kinematic_identity` | Tion + E_work is constant up to the exact velocity-Verlet term Σ \|F\|² dt²/(8m): an identity of SALMON's integrator, so it holds to rounding |
| `tion_vs_velocities` | `Tion` equals Σ ½ m v² from the trajectory (and `Temperature_ion` agrees) |
| `trajectory_bounds` | positions, velocities and forces finite; displacement, force, temperature and interatomic distances within sanity bounds |

**dft_md.** `md_step_scf_converged` (each MD step's final SCF residual below the threshold),
`dft_md_columns` (E_tot = Tene + Uene, constant Uene0 and E_tot0), `bo_energy_conservation`,
plus the kinematic identity, Tion and trajectory checks above.

**DC.** `dc_scf_history`; `dc_fragment_energy_consistency` (every fragment's info file has the
same total energy, equal to the last `DC #SCF` line within its e15.8 resolution);
`lcfo_output` (start and end markers, states 1..nstate present, finite and ascending, basis
at least nstate, CheFSI residual within its tolerance, complex-LCFO diagnostics ≤ 1e-10);
and, with `--conventional`, `dc_vs_conventional_energy` and `dc_spectrum_vs_conventional`
(HOMO, LUMO, mid-gap as a Fermi-level proxy, band energy). With `--buffer-dirs`,
`dc_buffer_sensitivity` (the DC error should not grow with the buffer). With `--rt-dc` and
`--rt-conv`, `rt_from_dc_vs_conventional` (excitation energy and current along the field).

**Testsuite references.** When the run directory is named like testsuite case 124, 126, 129
or 130, `stock_reference:*` re-applies that case's own reference values and tolerances,
reading the same lines, but a non-finite value always fails. Case 129's references differ
between release 2.3.0 and develop, so give `--salmon-version`.

**Comparison of two runs.** `exact` requires every numeric token of every compared file to be
identical: use it for two runs of the same build route and layout. `tolerance` applies
per-quantity limits for independent routes (CPU/GPU, versions, layouts) and reports, for
every column, the largest difference, the time of the first nonzero difference and of the
first difference above tolerance, and how fast the difference grows. Time grids must match.
Vector components share one scale, so a component that is zero by symmetry is judged
against the vector's magnitude.

## Tolerances

`python3 -I verify.py tolerances` prints every limit with what it applies to and why; the
text is in `mddc_verify/tolerances.py`. Their sources are of three kinds:

- **SALMON's own testsuite.** For example case 124 reproduces Eall to 1e-6 eV after 1000 MD
  steps; case 112 checks a current to 3.4e-5 relative; case 129 checks LCFO eigenvalues to
  1e-4 Ha.
- **Runs made while developing the tool**, on SALMON 2.3.0 CPU and GPU builds. Examples:
  - a CPU and a GPU build of the same RT run differ by at most 3.4e-13 eV/atom;
  - two CPU MPI layouts differ by 2.2e-10 eV/atom over 60 steps;
  - under a laser, the energy-balance residual is 0.15 of the work-integral ambiguity at
    two time steps, and it halves when dt halves.
  `tools/observed_noise.py` produces the same table for your own pairs of runs.
- **Physics and published results.** Examples: 1 meV/atom of energy error is 2.6% of the ion
  kinetic energy at 300 K; the DC limits use the deviations reported in Yamada and Otobe,
  arXiv:2606.21125.

Loosening a limit needs a written reason in `tolerances.py`.

## Known limits

- **Faults hidden under a laser.** One run under a laser cannot show an energy fault smaller
  than the work-integral ambiguity, about 0.1–1 meV/atom. A field-free window or a reference
  run can.
- **One wrong sample.** A single wrong but finite sample, such as one current value off by
  1e-6, passes every single-run check. Only `compare` against a reference run finds it.
- **Multi-SCF runs.** The cross-run energy limit (1e-6 eV/atom) is too tight for
  geometry-optimisation and BO-MD runs (testsuite cases 125 and 126). There each step's SCF
  stops at its threshold, so two CPU MPI layouts already differ by more than that. Judge
  such pairs by the testsuite's own check and by the spread between CPU layouts.
- **Not in SALMON's outputs.** These cannot be checked:
  - the DC Fermi level;
  - the real-orbital LCFO orthonormality;
  - per-fragment SCF convergence;
  - orbital norms during RT;
  - energies between `out_rt_energy_step` samples;
  - whether the forces are right (that needs finite differences). The kinematic identity
    only shows that Tion, E_work and the integrator agree.

## Tests and the fault-injection campaign

```sh
python3 -I -B -m unittest discover -s tests -t .        # unit tests, synthetic fixtures only
python3 -I tools/inject_faults.py OUTDIR [--quick]      # plant faults, see which checks fire
python3 -I tools/observed_noise.py OUTDIR --pair LABEL CAND REF [--field-free RUN ...]
```

The unit tests build their own run directories in temporary folders, so no SALMON run is
needed. `mddc_verify/fixtures.py` writes each file in the format of the SALMON writer it
imitates. The numbers come from small models:

- electrons driven by an Acos2 laser pulse;
- harmonic ions integrated in SALMON's velocity-Verlet order;
- chosen energies and eigenvalues for GS and DC.

Each fixture directory has a `FIXTURE.txt` that says so.

`inject_faults.py` plants 53 faults, each in a fresh copy of one fixture:

- NaN, Inf and overflow fields;
- energy drift;
- truncated and missing files;
- wrong but finite values;
- non-convergence;
- bad exit status, missing end marker, runtime errors.

It then records which checks fire, what the cross-run comparison says and, for the
testsuite-112-shaped fixture, what the testsuite's own `verification` script in this tree
says. It writes a table and, with matplotlib, a fault matrix. Clean fixtures pass, and every
fault is flagged (fail, warn or unknown) by the single-run checks, by the comparison, or both.
The single-run checks miss only single samples changed by about 1e-6, which the comparison
finds; the comparison misses exit-status, end-marker and convergence faults, which the
single-run checks find. The tool exits with 1 if a clean fixture does not pass or a fault
goes undetected.

## Layout

| File | Contents |
|---|---|
| `verify.py` | entry point (adds this directory to `sys.path`, so it works under `python3 -I`) |
| `mddc_verify/numeric.py` | NaN-safe parsing of Fortran numbers and comparisons |
| `mddc_verify/namelist.py`, `run.py` | the input deck and a run directory |
| `mddc_verify/readers.py` | readers for every output format used |
| `mddc_verify/verdict.py` | check statuses, verdict, JSON |
| `mddc_verify/generic.py`, `physics_md.py`, `physics_dc.py` | the checks |
| `mddc_verify/stock_refs.py` | testsuite reference values |
| `mddc_verify/compare.py`, `tolerances.py` | cross-run comparison; every limit with its reason |
| `mddc_verify/figures.py` | optional figures |
| `mddc_verify/fixtures.py` | synthetic run directories for the tests |
| `tests/`, `tools/` | unit tests; fault injection and observed-noise tables |
