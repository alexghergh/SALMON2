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
"""Layer 1: checks that apply to any SALMON run."""
import os

from .numeric import finite, fmt
from .verdict import Report, PASS, FAIL, WARN, UNKNOWN, NA

LAYER = "generic"


def expected_files(run, mode):
    sysn = run.sysname
    base = ["outputfile", "inputfile"]
    per = {
        "gs": [f"{sysn}_info.data", f"{sysn}_eigen.data"],
        "rt": [f"{sysn}_rt.data", f"{sysn}_rt_energy.data"],
        "ehrenfest": [f"{sysn}_rt.data", f"{sysn}_rt_energy.data", f"{sysn}_trj.xyz"],
        "dft_md": [f"{sysn}_dft_md.data", f"{sysn}_trj.xyz"],
        "dc": [os.path.relpath(run.dc_eigen_path(), run.dir)],
    }
    files = base + per.get(mode, [])
    if mode == "dc" and run.inputs and str(run.inputs.get("yn_dc_lcfo", "y")).lower()[:1] == "n":
        files = base
    return files


def check_exit_status(run, rep):
    from .readers import read_exit_status
    code, status = read_exit_status(run.dir)
    if status == "missing":
        return rep.add("exit_status", LAYER, UNKNOWN,
                       "salmon.exit missing: SALMON's own exit status was not recorded "
                       "(a pipe into tee hides it); this is never treated as a pass")
    if code is None:
        return rep.add("exit_status", LAYER, FAIL, f"salmon.exit {status}")
    if code != 0:
        return rep.add("exit_status", LAYER, FAIL, f"SALMON exited with status {code}",
                       {"exit_status": code})
    return rep.add("exit_status", LAYER, PASS, "exit status 0", {"exit_status": 0})


def check_termination(run, rep):
    so = run.stdout()
    if so is None:
        return rep.add("normal_termination", LAYER, FAIL, "outputfile missing or empty")
    if not so.end_salmon:
        return rep.add("normal_termination", LAYER, FAIL,
                       "no 'end SALMON' line: the run did not reach the end of main.f90 "
                       "(e.g. killed, crashed, or stopped by a plain STOP such as the RT "
                       "electron-count guard in dip.f90, which can still exit with status 0)",
                       {"lines": so.n_lines})
    return rep.add("normal_termination", LAYER, PASS, "'end SALMON' present")


def check_errors(run, rep):
    so = run.stdout()
    if so is None:
        return rep.add("runtime_errors", LAYER, UNKNOWN, "outputfile missing")
    msgs = list(so.errors)
    err = os.path.join(run.dir, "stderr.txt")
    if os.path.exists(err):
        from .readers import ERROR_PATTERNS
        with open(err, errors="replace") as fh:
            for ln, line in enumerate(fh, 1):
                for pat, label in ERROR_PATTERNS:
                    if pat.search(line):
                        msgs.append({"line": ln, "kind": label, "text": line.strip()[:160],
                                     "file": "stderr.txt"})
                        break
    if msgs:
        return rep.add("runtime_errors", LAYER, FAIL,
                       f"{len(msgs)} runtime error message(s), first: {msgs[0]['kind']} "
                       f"(line {msgs[0]['line']}): {msgs[0]['text'][:80]}", {"errors": msgs[:20]})
    return rep.add("runtime_errors", LAYER, PASS, "no runtime error messages in outputfile/stderr.txt")


def threshold_au(run):
    """SCF threshold in the units SALMON prints the residual in (inputoutput.f90:1364-1381)."""
    thr = run.threshold
    if thr is None or not finite(thr) or thr < -0.5:
        return 1e-17 if run.convergence == "rho_dne" else None
    if run.convergence == "rho_dne":
        return thr
    if run.au:
        return thr
    l2au = 1.0 / 0.52917721067
    e2au = 1.0 / 27.21138505
    if run.convergence in ("norm_rho", "norm_rho_dng"):
        return thr / l2au ** 6
    if run.convergence in ("norm_pot", "norm_pot_dng"):
        return thr * e2au ** 2 / l2au ** 6
    return None


def check_gs_convergence(run, rep, mode, cid="gs_convergence"):
    so = run.stdout()
    if so is None:
        return rep.add(cid, LAYER, UNKNOWN, "outputfile missing")
    thr = threshold_au(run)
    if mode == "gs":
        if so.gs_not_converged:
            tok, ln = so.gs_not_converged[-1]
            return rep.add(cid, LAYER, FAIL,
                           f"SALMON printed '#GS does not converged : {tok}' (line {ln}); the run "
                           "still wrote its outputs and ended normally", {"residual": tok})
        if not so.gs_converged:
            return rep.add(cid, LAYER, FAIL, "no '#GS converged at' line (and no non-convergence "
                                             "message): convergence not demonstrated")
        it, res, ln = so.gs_converged[-1]
        if not finite(res):
            return rep.add(cid, LAYER, FAIL, f"converged-line residual is not finite (line {ln})")
        if thr is not None and not (finite(res, thr) and res < thr):
            return rep.add(cid, LAYER, FAIL, f"residual {fmt(res)} >= threshold {fmt(thr)}",
                           {"residual": res, "threshold": thr})
        energies = [e for _, e in so.gs_iter_energy]
        if energies and not all(finite(e) for e in energies):
            return rep.add(cid, LAYER, FAIL, "non-finite total energy in the SCF history")
        return rep.add(cid, LAYER, PASS, f"converged at iteration {fmt(it)}, residual {fmt(res)} "
                       f"< threshold {fmt(thr)}", {"iteration": it, "residual": res, "threshold": thr})
    if mode == "dc":
        # main_dft.f90 sets ilevel_print=0 for DC, so neither '#GS converged' nor
        # '#GS does not converged' is ever printed; only the 'DC #SCF' lines show progress.
        if not so.dc_scf:
            return rep.add(cid, LAYER, FAIL, "no 'DC #SCF = ...' lines in outputfile")
        n, e, d, ln = so.dc_scf[-1]
        bad = [r for r in so.dc_scf if not finite(r[1]) or not finite(r[2])]
        if bad:
            return rep.add(cid, LAYER, FAIL, f"{len(bad)} DC SCF record(s) with non-finite energy or "
                           f"residual, first at line {bad[0][3]}")
        if thr is None:
            return rep.add(cid, LAYER, UNKNOWN, f"cannot convert threshold for convergence="
                           f"{run.convergence}")
        if not (finite(d, thr) and d < thr):
            return rep.add(cid, LAYER, FAIL,
                           f"DC SCF not converged: last residual {fmt(d)} >= threshold {fmt(thr)} "
                           f"after {fmt(n)} iterations (SALMON prints no message in DC mode)",
                           {"iterations": n, "residual": d, "threshold": thr, "nscf": run.nscf})
        return rep.add(cid, LAYER, PASS, f"DC SCF converged in {fmt(n)} iterations, residual "
                       f"{fmt(d)} < {fmt(thr)}", {"iterations": n, "residual": d, "threshold": thr,
                                                   "energy_ev": e})
    if mode == "dft_md":
        if so.gs_not_converged:
            return rep.add(cid, LAYER, FAIL, f"{len(so.gs_not_converged)} MD step(s) printed "
                           f"'#GS does not converged' (first at line {so.gs_not_converged[0][1]})",
                           {"n_not_converged": len(so.gs_not_converged)})
        return rep.add(cid, LAYER, PASS, "no '#GS does not converged' message in any MD step "
                       "(per-step residuals are checked in the physics layer)")
    return rep.add(cid, LAYER, NA, "no ground state in this run; pass --gs-dir to check the GS "
                   "it restarted from")


def check_files(run, rep, mode):
    missing, empty = [], []
    for f in expected_files(run, mode):
        p = os.path.join(run.dir, f)
        if not os.path.exists(p):
            missing.append(f)
        elif os.path.getsize(p) == 0:
            empty.append(f)
    if missing or empty:
        return rep.add("files_present", LAYER, FAIL,
                       f"missing: {missing or '-'}; empty: {empty or '-'}",
                       {"missing": missing, "empty": empty})
    return rep.add("files_present", LAYER, PASS,
                   f"{len(expected_files(run, mode))} expected files present and non-empty")


def _tables(run, mode):
    out = []
    if mode in ("rt", "ehrenfest"):
        out += [("rt.data", run.rt()), ("rt_energy.data", run.rt_energy())]
    if mode == "dft_md":
        out += [("dft_md.data", run.dft_md())]
    return [(n, t) for n, t in out if t is not None]


def check_finite(run, rep, mode):
    probs = []
    nvals = 0
    for name, t in _tables(run, mode):
        probs += [p for p in t.problems if p["kind"] in ("nonfinite", "overflow", "malformed")]
        nvals += sum(len(r) for r in t.rows)
    trj = run.trj() if mode in ("ehrenfest", "dft_md") else None
    if trj is not None:
        probs += [p for p in trj.problems if p["kind"] in ("nonfinite", "overflow", "malformed")]
        nvals += sum(9 * len(f.pos) for f in trj.frames)
    for getter in ((run.info, run.eigen) if mode == "gs" else ()):
        obj = getter()
        if obj is not None:
            pl = obj["problems"] if isinstance(obj, dict) else obj.problems
            probs += [p for p in pl if p["kind"] in ("nonfinite", "overflow", "malformed")]
            if not isinstance(obj, dict):
                nvals += sum(len(v) for v in obj.blocks.values())
    if mode == "dc":
        eg = run.dc_eigen()
        if eg is not None:
            probs += [p for p in eg.problems if p["kind"] in ("nonfinite", "overflow", "malformed")]
            nvals += sum(len(v) for v in eg.blocks.values())
    so = run.stdout()
    if so is not None:
        probs += [p for p in so.problems if p["kind"] in ("nonfinite", "overflow", "malformed")]
        probs += so.nonfinite_elsewhere
    if probs:
        kinds = sorted({p["kind"] for p in probs})
        first = probs[0]
        return rep.add("all_values_finite", LAYER, FAIL,
                       f"{len(probs)} non-finite/overflow/malformed value(s) ({', '.join(kinds)}); "
                       f"first: {first['file']} line {first['line']} token {first['token']!r}",
                       {"problems": probs[:25], "n_problems": len(probs)})
    return rep.add("all_values_finite", LAYER, PASS,
                   f"all {nvals} parsed table values and all stdout records finite",
                   {"n_values": nvals})


def _check_grid(name, times, step, dt_fs, factor, first_index):
    """times must equal (first_index + i*step)*dt within F16.8 rounding."""
    if dt_fs is None or factor is None:
        return None
    bad = []
    for i, t in enumerate(times):
        expect = (first_index + i * step) * dt_fs / factor
        if not finite(t) or abs(t - expect) > 2e-8 + 1e-9 * abs(expect):
            bad.append((i, t, expect))
            if len(bad) > 3:
                break
    return bad


def check_integrity(run, rep, mode):
    issues = []
    metrics = {}
    nt = run.nt if isinstance(run.nt, int) else None
    ers = run.out_rt_energy_step if isinstance(run.out_rt_energy_step, int) else 10
    rvs = run.out_rvf_rt_step if isinstance(run.out_rvf_rt_step, int) else 10
    for name, t in _tables(run, mode):
        for p in t.problems:
            if p["kind"] in ("ragged", "truncated_last_line", "no_header"):
                issues.append(f"{name}: {p['kind']} at line {p['line']} {p['note']}".strip())
        expect, first = None, 0
        if name == "rt.data" and nt:
            expect, step, first = nt, 1, 1
        elif name == "rt_energy.data" and nt:
            expect, step, first = nt // ers + 1, ers, 0
        elif name == "dft_md.data" and nt:
            expect, step, first = nt + 1, 1, 0
        metrics[name + "_rows"] = len(t.rows)
        if expect is not None and len(t.rows) != expect:
            issues.append(f"{name}: {len(t.rows)} rows, expected {expect} from nt={nt}")
        if expect is not None and t.has("Time"):
            fac = run.time_factor(t.unit("Time"))
            dt_out = run.dt_fs
            bad = _check_grid(name, t.col("Time"), step, dt_out, fac, first)
            if bad:
                i, got, exp = bad[0]
                issues.append(f"{name}: time column off grid at row {i + 1} ({fmt(got, 10)} vs "
                              f"{fmt(exp, 10)})")
    if mode in ("ehrenfest", "dft_md"):
        trj = run.trj()
        if trj is not None:
            for p in trj.problems:
                if p["kind"] in ("ragged", "truncated_frame", "truncated_last_line"):
                    issues.append(f"trj.xyz: {p['kind']} at line {p['line']} {p['note']}".strip())
            if nt:
                expect = nt // rvs + (1 if mode == "ehrenfest" else 0)
                metrics["trj_frames"] = len(trj.frames)
                if len(trj.frames) != expect:
                    issues.append(f"trj.xyz: {len(trj.frames)} frames, expected {expect}")
            nat = {len(f.pos) for f in trj.frames}
            if run.natom and nat and nat != {run.natom}:
                issues.append(f"trj.xyz: atom counts {sorted(nat)} != natom {run.natom}")
    so = run.stdout()
    if so is not None and nt and mode in ("rt", "ehrenfest"):
        expect = nt // ers
        metrics["stdout_rt_steps"] = len(so.rt_steps)
        if len(so.rt_steps) != expect:
            issues.append(f"outputfile: {len(so.rt_steps)} RT step lines, expected {expect}")
    if so is not None and nt and mode == "dft_md":
        metrics["stdout_md_steps"] = len(so.dft_md_steps)
        if len(so.dft_md_steps) != nt:
            issues.append(f"outputfile: {len(so.dft_md_steps)} MD step lines, expected {nt}")
    if run.inputs is None:
        return rep.add("table_integrity", LAYER, UNKNOWN,
                       "inputfile missing: expected row counts and time grid unknown", metrics)
    if issues:
        return rep.add("table_integrity", LAYER, FAIL, "; ".join(issues[:4]) +
                       (f" (+{len(issues) - 4} more)" if len(issues) > 4 else ""),
                       dict(metrics, issues=issues))
    return rep.add("table_integrity", LAYER, PASS,
                   "row/frame counts and time grids match nt, dt and the output steps", metrics)


def check_timer(run, rep):
    so = run.stdout()
    if so is None:
        return rep.add("timer_table", LAYER, UNKNOWN, "outputfile missing")
    if not so.timer_table or so.timer_rows == 0:
        return rep.add("timer_table", LAYER, FAIL, "timer table ('=== application breakdown') "
                       "missing: the run did not reach its final report")
    if so.timer_nonfinite:
        return rep.add("timer_table", LAYER, WARN,
                       f"timer table present ({so.timer_rows} rows) but has "
                       f"{len(so.timer_nonfinite)} NaN/Inf statistic(s) (cosmetic: std. dev. NaN "
                       "from rounding in releases up to 2.3.0, 'relative' Inf from a zero minimum time)",
                       {"rows": so.timer_rows, "nonfinite": so.timer_nonfinite[:10]})
    return rep.add("timer_table", LAYER, PASS, f"timer table present ({so.timer_rows} rows)",
                   {"rows": so.timer_rows})


def generic_report(run, mode, rep=None):
    rep = rep or Report(run.dir, mode)
    check_exit_status(run, rep)
    check_termination(run, rep)
    check_errors(run, rep)
    check_gs_convergence(run, rep, mode)
    check_files(run, rep, mode)
    check_finite(run, rep, mode)
    check_integrity(run, rep, mode)
    check_timer(run, rep)
    return rep
