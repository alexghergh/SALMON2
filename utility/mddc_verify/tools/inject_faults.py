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
"""Fault-injection campaign: do the checks catch faults planted in copies of SALMON outputs?

Fixtures (synthetic, in SALMON's formats; see mddc_verify/fixtures.py), built fresh in WORKDIR:
  rt        fixed-ion 8-atom silicon under an Acos2 pulse
  gs        converged ground state of 8-atom silicon
  ts112     the file layout of testsuite 112 (+ its stock verification script, if found)
  md_pulse  Ehrenfest MD under the pulse
  md_nve    Ehrenfest MD, field-free
  dftmd     Born-Oppenheimer dft_md
  dc        divide-and-conquer run (+ a conventional reference)
Each fault is planted in a fresh copy of one fixture. The verifier runs on the copy, and the
cross-run comparison (copy against the clean fixture) runs with both policies. For ts112 the
testsuite's own verification script runs on the same copy.

  python3 -I tools/inject_faults.py OUTDIR [--work WORKDIR] [--quick] [--only FAULT ...]
                                    [--stock-112 PATH]

Writes OUTDIR/fault-injection.{json,md} and, with matplotlib, OUTDIR/fault-matrix.{png,md}.
WORKDIR (default OUTDIR/work) holds the fixtures and mutated copies (about 0.5 GB; --quick
uses runs ten times shorter). Exit status 0 when every fault was detected by at least one layer
and every clean control passed, 1 otherwise.
"""
import argparse
import json
import os
import re
import shutil
import subprocess
import sys

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)

from mddc_verify import fixtures as fx  # noqa: E402
from mddc_verify.cli import check  # noqa: E402
from mddc_verify.compare import compare_runs  # noqa: E402
from mddc_verify.run import Run  # noqa: E402

STOCK_112 = os.path.join(HERE, "..", "..", "testsuites", "112_bulk_Si_rt_response_dp", "verification")


# --------------------------------------------------------------------------- mutators
def _data_lines(lines):
    return [i for i, l in enumerate(lines) if l.strip() and not l.lstrip().startswith("#")]


def _spans(line):
    return [(m.start(), m.end()) for m in re.finditer(r"\S+", line)]


def _pick(idx, row):
    """row: an int index into idx (negative from the end) or a fraction in [0, 1)."""
    return idx[int(row * len(idx))] if isinstance(row, float) else idx[row]


def set_token(path, row, col, new, data_row=True):
    """Replace whitespace field `col` (1-based) of a data row (int or fraction), or of raw
    line `row` (0-based) when data_row is False, keeping the field width."""
    with open(path) as fh:
        lines = fh.read().split("\n")
    i = _pick(_data_lines(lines), row) if data_row else row
    a, b = _spans(lines[i])[col - 1]
    old = lines[i][a:b]
    val = new(old) if callable(new) else new
    lines[i] = lines[i][:a] + val.rjust(b - a) + lines[i][b:]
    with open(path, "w") as fh:
        fh.write("\n".join(lines))
    return old


def fortran_add(delta):
    return lambda tok: fx.fe(float(tok) + delta)


def fortran_scale(factor):
    return lambda tok: fx.fe(float(tok) * factor)


def add_drift(path, cols, total_per_atom, natom=8):
    """Add a linear drift reaching total_per_atom*natom [eV] at the last row to the columns."""
    with open(path) as fh:
        lines = fh.read().split("\n")
    rows = _data_lines(lines)
    t_end = float(lines[rows[-1]].split()[0])
    for i in rows:
        sp = _spans(lines[i])
        t = float(lines[i][sp[0][0]:sp[0][1]])
        for c in sorted(cols, reverse=True):
            a, b = sp[c - 1]
            v = float(lines[i][a:b]) + total_per_atom * natom * t / t_end
            lines[i] = lines[i][:a] + fx.fe(v).rjust(b - a) + lines[i][b:]
    with open(path, "w") as fh:
        fh.write("\n".join(lines))


def truncate(path, fraction):
    size = os.path.getsize(path)
    with open(path, "rb+") as fh:
        fh.truncate(int(size * fraction) + 7)  # +7: usually lands mid-line


def text_replace(path, old, new, count=1):
    with open(path, errors="replace") as fh:
        s = fh.read()
    if old not in s:
        raise ValueError(f"{old!r} not in {path}")
    with open(path, "w") as fh:
        fh.write(s.replace(old, new, count))


def xyz_field(path, frame, atom, field, new):
    """frame: index or fraction; field 0..8 = x y z vx vy vz fx fy fz (fixed F18.10 layout)."""
    with open(path) as fh:
        lines = fh.read().split("\n")
    starts, i = [], 0
    while i < len(lines) and lines[i].strip():
        starts.append(i)
        i += int(lines[i].strip()) + 2
    li = _pick(starts, frame) + 2 + atom
    line = lines[li]
    off = 2 + field * 18 + (5 if field >= 3 else 0) + (5 if field >= 6 else 0)
    old = line[off:off + 18]
    val = new(old) if callable(new) else new
    lines[li] = line[:off] + val.rjust(18)[:18] + line[off + 18:]
    with open(path, "w") as fh:
        fh.write("\n".join(lines))


def f18(fn):
    return lambda tok: f"{fn(float(tok)):18.10f}"


def _stdout_rt_token(p, row, col, new):
    """Replace token `col` (1-based) of an RT step line (index or fraction) in outputfile."""
    with open(p) as fh:
        lines = fh.read().split("\n")
    h = next(i for i, l in enumerate(lines) if "time-step" in l and "electrons" in l)
    steps = [i for i in range(h + 2, len(lines)) if re.match(r"^\s*\d+\s+\d+\.\d+\s", lines[i])]
    i = _pick(steps, row)
    a, b = _spans(lines[i])[col - 1]
    lines[i] = lines[i][:a] + new.rjust(b - a) + lines[i][b:]
    with open(p, "w") as fh:
        fh.write("\n".join(lines))


def _dc_last(p, energy=None, diff=None):
    with open(p) as fh:
        lines = fh.read().split("\n")
    i = max(k for k, l in enumerate(lines) if "DC #SCF" in l)
    m = re.search(r"Total Energy =\s*(\S+)\s+diff =\s*(\S+)", lines[i])
    l = lines[i]
    if diff is not None:
        l = l[:m.start(2)] + diff.rjust(m.end(2) - m.start(2)) + l[m.end(2):]
    if energy is not None:
        l = l[:m.start(1)] + energy.rjust(m.end(1) - m.start(1)) + l[m.end(1):]
    lines[i] = l
    with open(p, "w") as fh:
        fh.write("\n".join(lines))


def _dftmd_noconv(d):
    p = os.path.join(d, "outputfile")
    with open(p) as fh:
        lines = fh.read().split("\n")
    k = next(i for i, l in enumerate(lines) if l.startswith("# 1.time[au], 2.Tene(ion)[au]")) + 20
    lines[k] = lines[k][:-15] + fx.fe(5e-6, 8, 2).rjust(15)
    lines.insert(k, "  #GS does not converged : 0.50000000E-05")
    with open(p, "w") as fh:
        fh.write("\n".join(lines))


def _drop_timer(p):
    with open(p) as fh:
        s = fh.read()
    a, b = s.index("=== application breakdown"), s.index("=== performance")
    with open(p, "w") as fh:
        fh.write(s[:a] + s[b:])


def _ne_guard(p):
    """What src/rt/dip.f90 prints before its plain STOP when the electron count is NaN."""
    with open(p) as fh:
        s = fh.read()
    a = s.index("=== application breakdown")
    with open(p, "w") as fh:
        fh.write(s[:a] + "           0 t=        4123\n           0 rbox1=   NaN\n           0 Ne=   NaN\n")


# --------------------------------------------------------------------------- campaign
SIZES = {"rt": 8000, "md_pulse": 8000, "md_nve": 4000, "dftmd": 200}


def build(root, quick=False):
    d = os.path.join(root, "fixtures")
    if os.path.exists(d):
        shutil.rmtree(d)
    os.makedirs(d)
    n = {k: v // 10 if quick else v for k, v in SIZES.items()}
    fx.rt(os.path.join(d, "rt"), nt=n["rt"])
    fx.gs(os.path.join(d, "gs"))
    fx.testsuite_112(os.path.join(d, "ts112"))
    fx.ehrenfest(os.path.join(d, "md_pulse"), pulse=True, nt=n["md_pulse"])
    fx.ehrenfest(os.path.join(d, "md_nve"), pulse=False, nt=n["md_nve"])
    fx.dft_md(os.path.join(d, "dftmd"), nt=n["dftmd"])
    fx.conventional(os.path.join(d, "conv"))
    fx.dc(os.path.join(d, "dc"), e_per_atom_offset=3e-4, buffer=12)
    return d


MODE = {"rt": "rt", "gs": "gs", "ts112": "rt", "md_pulse": "ehrenfest", "md_nve": "ehrenfest",
        "dftmd": "dft_md", "dc": "dc"}

E, R, X, DM = "Si_rt_energy.data", "Si_rt.data", "Si_trj.xyz", "Si_dft_md.data"
LCFO = os.path.join("data_dcdft", "total", "Si_eigen.data")
# (fault id, fixture, description, kind, mutate(dir))
FAULTS = [
    # --- NaN / Inf / overflow
    ("nan_eall", "rt", "NaN in one Eall value (middle row)", "nan",
     lambda d: set_token(os.path.join(d, E), 0.5, 2, "NaN")),
    ("nan_jm", "rt", "NaN in one current value Jm_x (middle row)", "nan",
     lambda d: set_token(os.path.join(d, R), 0.5, 14, "NaN")),
    ("inf_eall", "rt", "Infinity in one Eall value", "nan",
     lambda d: set_token(os.path.join(d, E), 0.5, 2, "Infinity")),
    ("overflow_jm", "rt", "Fortran overflow field ***** in Jm_x", "nan",
     lambda d: set_token(os.path.join(d, R), 0.5, 14, "*" * 23)),
    ("nan_stdout_ne", "rt", "NaN electron count in an RT step line of outputfile", "nan",
     lambda d: _stdout_rt_token(os.path.join(d, "outputfile"), 0.5, 6, "NaN")),
    ("nan_checked_112", "ts112", "NaN at the value the stock 112 check reads (Jm_z, line 167)", "nan",
     lambda d: set_token(os.path.join(d, R), 166, 16, "NaN", data_row=False)),
    ("nan_all_jm_112", "ts112", "NaN in the whole Jm_z column", "nan",
     lambda d: [set_token(os.path.join(d, R), k, 16, "NaN") for k in range(180)]),
    ("nan_tion", "md_pulse", "NaN in one Tion value", "nan",
     lambda d: set_token(os.path.join(d, E), 0.5, 4, "NaN")),
    ("nan_velocity", "md_pulse", "NaN in one velocity component (trj.xyz)", "nan",
     lambda d: xyz_field(os.path.join(d, X), 0.5, 3, 4, "NaN")),
    ("nan_force", "md_nve", "NaN in one force component (trj.xyz)", "nan",
     lambda d: xyz_field(os.path.join(d, X), 0.5, 5, 7, "NaN")),
    ("nan_dftmd_etot", "dftmd", "NaN in one E_tot value of dft_md.data", "nan",
     lambda d: set_token(os.path.join(d, DM), 0.5, 7, "NaN")),
    ("nan_dc_energy", "dc", "NaN in the last DC #SCF total energy", "nan",
     lambda d: _dc_last(os.path.join(d, "outputfile"), energy="NaN")),
    ("nan_lcfo_eigen", "dc", "NaN in one LCFO eigenvalue", "nan",
     lambda d: set_token(os.path.join(d, LCFO), 30, 2, "NaN")),
    # --- energy drift (total over the run, per atom)
    ("drift_md_1e-2", "md_nve", "Eall drift of 1e-2 eV/atom over the run (field-free MD)", "drift",
     lambda d: add_drift(os.path.join(d, E), [2, 3], 1e-2)),
    ("drift_md_1e-3", "md_nve", "Eall drift of 1e-3 eV/atom over the run", "drift",
     lambda d: add_drift(os.path.join(d, E), [2, 3], 1e-3)),
    ("drift_md_1e-4", "md_nve", "Eall drift of 1e-4 eV/atom over the run (10 meV/atom/ps at full length)",
     "drift", lambda d: add_drift(os.path.join(d, E), [2, 3], 1e-4)),
    ("drift_md_1e-5", "md_nve", "Eall drift of 1e-5 eV/atom over the run (1 meV/atom/ps, warn level)",
     "drift", lambda d: add_drift(os.path.join(d, E), [2, 3], 1e-5)),
    ("drift_laser_1e-3", "md_pulse", "Eall drift of 1e-3 eV/atom over the run (laser on)", "drift",
     lambda d: add_drift(os.path.join(d, E), [2, 3], 1e-3)),
    ("drift_laser_1e-4", "md_pulse", "Eall drift of 1e-4 eV/atom over the run (laser on)", "drift",
     lambda d: add_drift(os.path.join(d, E), [2, 3], 1e-4)),
    ("drift_rt_1e-3", "rt", "Eall drift of 1e-3 eV/atom over the run (fixed ions, laser on)", "drift",
     lambda d: add_drift(os.path.join(d, E), [2, 3], 1e-3)),
    ("drift_dftmd_1e-3", "dftmd", "E_tot drift of 1e-3 eV/atom (consistent columns)", "drift",
     lambda d: add_drift(os.path.join(d, DM), [3, 4, 7, 8, 9], 1e-3)),
    # --- truncation
    ("trunc_energy", "rt", "rt_energy.data cut at 60% (mid-line)", "truncated",
     lambda d: truncate(os.path.join(d, E), 0.6)),
    ("trunc_rt", "rt", "rt.data cut at 60%", "truncated",
     lambda d: truncate(os.path.join(d, R), 0.6)),
    ("trunc_stdout", "rt", "outputfile cut at 50% (no end SALMON, no timer table)", "truncated",
     lambda d: truncate(os.path.join(d, "outputfile"), 0.5)),
    ("trunc_xyz", "md_pulse", "trj.xyz cut mid-frame", "truncated",
     lambda d: truncate(os.path.join(d, X), 0.5)),
    ("trunc_dftmd", "dftmd", "dft_md.data cut at 50%", "truncated",
     lambda d: truncate(os.path.join(d, DM), 0.5)),
    ("trunc_112", "ts112", "Si_rt.data of 112 cut after the checked line 167", "truncated",
     lambda d: truncate(os.path.join(d, R), 0.93)),
    ("empty_energy", "rt", "rt_energy.data empty", "truncated",
     lambda d: open(os.path.join(d, E), "w").close()),
    ("missing_xyz", "md_pulse", "trj.xyz missing", "truncated",
     lambda d: os.remove(os.path.join(d, X))),
    ("missing_lcfo", "dc", "LCFO eigen file missing", "truncated",
     lambda d: os.remove(os.path.join(d, LCFO))),
    # --- wrong but finite
    ("spike_eall", "md_nve", "one Eall value +0.01 eV (single spike)", "wrong",
     lambda d: set_token(os.path.join(d, E), 0.5, 2, fortran_add(0.01))),
    ("tion_x1.01", "md_pulse", "one Tion value x1.01", "wrong",
     lambda d: set_token(os.path.join(d, E), 0.5, 4, fortran_scale(1.01))),
    ("ework_wrong", "md_nve", "one E_work value +1e-5 eV", "wrong",
     lambda d: set_token(os.path.join(d, E), 0.5, 6, fortran_add(1e-5))),
    ("velocity_x1.1", "md_pulse", "one velocity component x1.1 (trj.xyz)", "wrong",
     lambda d: xyz_field(os.path.join(d, X), 0.5, 2, 3, f18(lambda v: v * 1.1))),
    ("position_jump", "md_nve", "one atom moved by +1.2 A in one frame", "wrong",
     lambda d: xyz_field(os.path.join(d, X), 0.5, 1, 0, f18(lambda v: v + 1.2))),
    ("force_huge", "md_nve", "one force component set to 0.8 Ha/bohr", "wrong",
     lambda d: xyz_field(os.path.join(d, X), 0.5, 1, 6, f18(lambda v: 0.8))),
    ("jm_checked_112", "ts112", "Jm_z at the checked line +1e-6 (finite, wrong)", "wrong",
     lambda d: set_token(os.path.join(d, R), 166, 16, fortran_add(1e-6), data_row=False)),
    ("jm_unchecked_112", "ts112", "Jm_z at the last line +1e-6 (unchecked by stock 112)", "wrong",
     lambda d: set_token(os.path.join(d, R), -1, 16, fortran_add(1e-6))),
    ("jm_rel1e-6", "rt", "one Jm_x value x(1+1e-6)", "wrong",
     lambda d: set_token(os.path.join(d, R), 0.5, 14, fortran_scale(1 + 1e-6))),
    ("dftmd_etot_inconsistent", "dftmd", "one E_tot != Tene+Uene by 1e-6 eV", "wrong",
     lambda d: set_token(os.path.join(d, DM), 0.5, 7, fortran_add(1e-6))),
    ("dc_energy_off", "dc", "DC total energy +0.5 eV/atom in the fragment info files", "wrong",
     lambda d: [text_replace(os.path.join(d, "data_dcdft", "fragments", f"{k:06d}", "Si_info.data"),
                             "Total energy (eV) =    -3390", "Total energy (eV) =    -3374") for k in range(1, 5)]),
    ("lcfo_shift", "dc", "LCFO eigenvalues 31-66 shifted by +0.02 Ha", "wrong",
     lambda d: [set_token(os.path.join(d, LCFO), k, 2, lambda t: fx.fe(float(t) + 0.02, 16, 3))
                for k in range(30, 66)]),
    ("lcfo_basis_small", "dc", "LCFO basis dimension below nstate", "wrong",
     lambda d: text_replace(os.path.join(d, "outputfile"), "#dim=        640", "#dim=         40")),
    # --- not converged / runtime status
    ("gs_noconv", "gs", "'#GS converged' replaced by '#GS does not converged'", "noconv",
     lambda d: text_replace(os.path.join(d, "outputfile"), fx.GS_CONVERGED_LINE,
                            "  #GS does not converged : 0.12345678E-05")),
    ("gs_noconv_silent", "gs", "convergence line removed (no message either way)", "noconv",
     lambda d: text_replace(os.path.join(d, "outputfile"), fx.GS_CONVERGED_LINE + "\n", "")),
    ("dc_noconv", "dc", "last DC SCF residual 1e-6 > threshold 1e-9 (SALMON prints nothing)", "noconv",
     lambda d: _dc_last(os.path.join(d, "outputfile"), diff="0.10000000E-05")),
    ("dftmd_step_noconv", "dftmd", "one MD step with residual 5e-6 and a 'does not converged' line", "noconv",
     _dftmd_noconv),
    ("exit_nonzero", "rt", "salmon.exit = 1", "status",
     lambda d: open(os.path.join(d, "salmon.exit"), "w").write("1\n")),
    ("exit_missing", "rt", "salmon.exit missing (a pipe into tee hid the status)", "status",
     lambda d: os.remove(os.path.join(d, "salmon.exit"))),
    ("no_end_salmon", "rt", "'end SALMON' removed (e.g. the plain STOP in dip.f90)", "status",
     lambda d: text_replace(os.path.join(d, "outputfile"), "end SALMON", "")),
    ("no_timer", "rt", "timer table removed", "status",
     lambda d: _drop_timer(os.path.join(d, "outputfile"))),
    ("runtime_error", "rt", "a 'Fortran runtime error' line in outputfile", "status",
     lambda d: text_replace(os.path.join(d, "outputfile"), "end SALMON",
                            "Fortran runtime error: Floating point exception\nend SALMON")),
    ("ne_guard_stop", "rt", "dip.f90 electron-count guard message, then STOP", "status",
     lambda d: _ne_guard(os.path.join(d, "outputfile"))),
]


def run_check(name, d, fixtures):
    kw = {"conventional": os.path.join(fixtures, "conv")} if name == "dc" else {}
    _, rep = check(d, MODE[name], **kw)
    return rep


def stock_112(d, script):
    if not script or not os.path.exists(script):
        return None
    r = subprocess.run([sys.executable, "-I", "-B", script], cwd=d, capture_output=True, text=True,
                       timeout=60)
    return "accept" if r.returncode == 0 else "reject"


def _entry(fault, name, kind, desc, rep, stock, ex, tol):
    return {"fault": fault, "fixture": name, "kind": kind, "description": desc, "verdict": rep.verdict(),
            "fired": [c["id"] for c in rep.checks if c["status"] in ("fail", "unknown")],
            "warned": [c["id"] for c in rep.checks if c["status"] == "warn"],
            "stock": stock, "exact": ex, "tolerance": tol}


def main(out, work=None, quick=False, only=None, stock_script=STOCK_112, figdir=None):
    work = work or os.path.join(out, "work")
    os.makedirs(out, exist_ok=True)
    fixtures = build(work, quick)
    cases = os.path.join(work, "cases")
    if os.path.exists(cases):
        shutil.rmtree(cases)
    os.makedirs(cases)
    results = []
    for name in MODE:   # clean controls first
        d = os.path.join(fixtures, name)
        rep = run_check(name, d, fixtures)
        results.append(_entry("clean_control", name, "control", "unmodified fixture", rep,
                              stock_112(d, stock_script) if name == "ts112" else None, None, None))
    for fid, name, desc, kind, mut in FAULTS:
        if only and fid not in only:
            continue
        d = os.path.join(cases, fid)
        shutil.copytree(os.path.join(fixtures, name), d)
        mut(d)
        rep = run_check(name, d, fixtures)
        a, b = Run(d), Run(os.path.join(fixtures, name))
        try:
            ex = compare_runs(a, b, MODE[name], "exact")[0].verdict()
            tol = compare_runs(a, b, MODE[name], "tolerance")[0].verdict()
        except Exception as e:  # a fault may make a file unreadable; that is a detection
            ex = tol = f"error: {type(e).__name__}"
        results.append(_entry(fid, name, kind, desc, rep,
                              stock_112(d, stock_script) if name == "ts112" else None, ex, tol))
        print(f"{fid:<26} {rep.verdict():<8} {','.join(results[-1]['fired'])[:90]}")
    with open(os.path.join(out, "fault-injection.json"), "w") as fh:
        json.dump({"quick": quick, "results": results}, fh, indent=1)
    md = ["| Fault | Fixture | Verifier verdict | Checks that fired | Cross-run exact | Cross-run tolerance "
          "| Stock 112 |", "|---|---|---|---|---|---|---|"]
    for r in results:
        md.append(f"| {r['description']} (`{r['fault']}`) | {r['fixture']} | **{r['verdict']}** | "
                  f"{', '.join(r['fired']) or '–'}"
                  f"{(' (warn: ' + ', '.join(r['warned']) + ')') if r['warned'] else ''} | "
                  f"{r['exact'] or '–'} | {r['tolerance'] or '–'} | {r['stock'] or '–'} |")
    with open(os.path.join(out, "fault-injection.md"), "w") as fh:
        fh.write("\n".join(md) + "\n")
    try:
        from mddc_verify import figures
        figures.fault_matrix(results, figdir or out)
    except ImportError as e:
        print("figure skipped:", e)
    return results


def summary(results):
    """(clean controls that did not pass, faults that no layer detected)."""
    bad_controls = [r for r in results if r["fault"] == "clean_control" and r["verdict"] != "pass"]
    missed = [r for r in results if r["fault"] != "clean_control" and r["verdict"] == "pass"
              and r["exact"] == "pass" and r["tolerance"] == "pass"]
    return bad_controls, missed


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("outdir")
    ap.add_argument("--work", help="directory for fixtures and mutated copies (default OUTDIR/work)")
    ap.add_argument("--quick", action="store_true", help="runs ten times shorter (for tests)")
    ap.add_argument("--only", nargs="*", help="run only these fault ids")
    ap.add_argument("--stock-112", default=STOCK_112,
                    help="testsuite 112 verification script (default: the one in this SALMON tree)")
    args = ap.parse_args()
    res = main(args.outdir, args.work, args.quick, args.only, args.stock_112)
    bad, missed = summary(res)
    for r in bad:
        print(f"clean control {r['fixture']}: {r['verdict']}")
    for r in missed:
        print(f"not detected by any layer: {r['fault']}")
    sys.exit(1 if bad or missed else 0)
