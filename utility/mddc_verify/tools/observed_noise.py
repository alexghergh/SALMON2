#!/usr/bin/env python3
#
# Copyright 2026 SALMON developers
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Tabulate the differences observed between pairs of real RT runs (to ground tolerances).

For each pair (candidate, reference) of RT run directories, compares the two column by column
with the tolerance policy and records the largest difference per quantity class (energy per
atom, response current relative to max|J|, field relative to max|E|) and the time of the first
nonzero difference. For each field-free run it records the energy conservation. No solver run.

  python3 -I tools/observed_noise.py OUTDIR --pair LABEL CANDIDATE REFERENCE [--pair ...]
                                            [--field-free RUNDIR ...]

Writes OUTDIR/observed-noise.{json,md}. Use pairs that differ in one thing only (CPU vs GPU
build, MPI layout, SALMON version) so the table says how much that one thing moves results.
"""
import argparse
import json
import os
import sys

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)

from mddc_verify.run import Run  # noqa: E402
from mddc_verify.compare import compare_runs  # noqa: E402
from mddc_verify.verdict import Report  # noqa: E402
from mddc_verify.physics_md import check_conservation  # noqa: E402


def pair_row(label, a, b):
    ra, rb = Run(a), Run(b)
    rep, _ = compare_runs(ra, rb, "rt", "tolerance")
    exact, _ = compare_runs(ra, rb, "rt", "exact")
    natom = rb.natom or 1
    agg = {"label": label, "candidate": a, "reference": b, "natom": natom, "exact": exact.verdict(),
           "tolerance_verdict": rep.verdict()}
    for c in rep.checks:
        for col, d in c["metrics"].get("columns", {}).items():
            if d["class"] == "energy":
                en = ra.rt_energy()
                f = ra.energy_factor(en.unit(col)) if en is not None and en.has(col) else None
                if f is None:
                    continue
                v, key = (d["max_abs_diff"] or 0) * f / natom, "energy_max_diff_per_atom_ev"
            else:
                v, key = d["rel_max_diff"] or 0.0, f"{d['class']}_max_rel_diff"
            if isinstance(v, (int, float)) and v >= agg.get(key, -1):
                agg[key], agg[key + "_column"] = v, col
            fz = d.get("first_nonzero_t")
            if fz is not None and (agg.get("first_nonzero_t") is None or fz < agg["first_nonzero_t"]):
                agg["first_nonzero_t"] = fz
    return agg


def conservation_row(d):
    r = Run(d)
    rep = Report(d, "rt")
    check_conservation(r, rep)
    m = rep.checks[0]["metrics"]
    return {"run": d, "status": rep.checks[0]["status"], "dt_fs": r.dt_fs, "nt": r.nt, "natom": r.natom,
            "window_fs": m.get("window_fs"), "max_abs_per_atom_ev": m.get("max_abs_per_atom_ev"),
            "drift_mev_per_atom_per_ps": m.get("drift_mev_per_atom_per_ps")}


def _f(x, spec):
    return format(x, spec) if isinstance(x, (int, float)) else "–"


def main(outdir, pairs, field_free):
    rows = [pair_row(*p) for p in pairs]
    cons = [conservation_row(d) for d in field_free]
    os.makedirs(outdir, exist_ok=True)
    with open(os.path.join(outdir, "observed-noise.json"), "w") as fh:
        json.dump({"pairs": rows, "field_free_conservation": cons}, fh, indent=1)
    md = ["| Pair (candidate vs reference) | Exact | Max energy diff (eV/atom) | Max current diff (rel.) | "
          "Max field diff (rel.) | First nonzero diff (fs) |", "|---|---|---:|---:|---:|---:|"]
    for r in rows:
        md.append(f"| {r['label']} | {r['exact']} | {_f(r.get('energy_max_diff_per_atom_ev', 0), '.2e')} | "
                  f"{_f(r.get('response_max_rel_diff', 0), '.2e')} | {_f(r.get('field_max_rel_diff', 0), '.2e')} | "
                  f"{_f(r.get('first_nonzero_t'), '.6g') if r.get('first_nonzero_t') is not None else 'none'} |")
    md += ["", "| Field-free run | dt (fs) | Steps | Window (fs) | max abs dE / N (eV/atom) | Drift (meV/atom/ps) |",
           "|---|---:|---:|---|---:|---:|"]
    for c in cons:
        w = c["window_fs"]
        md.append(f"| {c['run']} | {_f(c['dt_fs'], '.6g')} | {c['nt']} | "
                  f"{(_f(w[0], '.3g') + '–' + _f(w[1], '.3g')) if w else '–'} | "
                  f"{_f(c['max_abs_per_atom_ev'], '.2e')} | {_f(c['drift_mev_per_atom_per_ps'], '.3g')} |")
    with open(os.path.join(outdir, "observed-noise.md"), "w") as fh:
        fh.write("\n".join(md) + "\n")
    return rows, cons, md


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("outdir")
    ap.add_argument("--pair", nargs=3, action="append", default=[], metavar=("LABEL", "CANDIDATE", "REFERENCE"))
    ap.add_argument("--field-free", nargs="*", default=[], metavar="RUNDIR")
    args = ap.parse_args()
    if not args.pair and not args.field_free:
        ap.error("give at least one --pair or --field-free run")
    print("\n".join(main(args.outdir, args.pair, args.field_free)[2]))
