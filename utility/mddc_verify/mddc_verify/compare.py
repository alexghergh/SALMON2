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
"""Layer 3: cross-run comparisons (CPU/GPU, 2.3.0/develop, original/modified).

Two policies:
  exact      every numeric token identical (for runs of the same build route and MPI/GPU
             layout); the first differing token is reported with its time.
  tolerance  per-quantity limits from tolerances.py; for every column the time of first
             divergence (first nonzero difference and first difference above tolerance) and the
             growth rate of |difference| (decades per fs, doubling time) are reported.
A comparison never passes if either value is non-finite.
"""
import math
import os

from .numeric import finite, fmt, linear_fit, max_abs, parse_number, OK
from .run import AU_ENERGY_EV, KB_AU
from .tolerances import TOLERANCES as T, describe
from .verdict import Report, PASS, FAIL, UNKNOWN

LAYER = "compare"
ENERGY_COLS = {"Eall", "Eall-Eall0", "Tion", "E_work", "E_kin", "E_h", "E_ion", "E_xc", "E_ion_ion",
               "Tene", "Uene", "Uene-Uene0", "E_nh", "E_tot", "E_tot-E_tot0", "Hnvt(2+3+6)",
               "Hnvt(2+5+6)'", "Enh", "Hnvt", "Hnvt'"}
FIELD_COLS = {"Ac_ext_x", "Ac_ext_y", "Ac_ext_z", "E_ext_x", "E_ext_y", "E_ext_z"}
TOT_FIELD_COLS = {"Ac_tot_x", "Ac_tot_y", "Ac_tot_z", "E_tot_x", "E_tot_y", "E_tot_z"}


def column_tolerance(label, unit, run, longitudinal=False):
    """(atol, rtol, class) for one column; rtol is relative to max|reference column|."""
    natom = run.natom or 1
    if label in ENERGY_COLS:
        f = run.energy_factor(unit) or 1.0
        return T["cross_run_energy_atol_per_atom"]["fail"] * natom / f, 0.0, "energy"
    if label == "Temperature_ion" or label == "Temperature":
        kb_ev = KB_AU * AU_ENERGY_EV
        return 2 * T["cross_run_energy_atol_per_atom"]["fail"] / (3 * kb_ev), 0.0, "energy"
    if label in FIELD_COLS or (label in TOT_FIELD_COLS and not longitudinal):
        return 0.0, T["cross_run_field_rtol"]["fail"], "field"
    return 0.0, T["cross_run_current_rtol"]["fail"], "response"


def divergence(t, a, b, atol, rtol, scale=None):
    """Per-column divergence summary. scale: magnitude for rtol (default max|b| of this column)."""
    n = min(len(a), len(b))
    diffs = []
    nonfinite = None
    for k in range(n):
        if not finite(a[k], b[k]):
            nonfinite = k
            diffs.append(float("nan"))
        else:
            diffs.append(abs(a[k] - b[k]))
    if scale is None:
        scale = max_abs([x for x in b[:n] if finite(x)]) if n else float("nan")
    thr = atol + rtol * (scale if finite(scale) else 0.0)
    first_nz = next((k for k, d in enumerate(diffs) if not finite(d) or d > 0), None)
    first_over = next((k for k, d in enumerate(diffs) if not finite(d) or d > thr), None)
    finite_d = [d for d in diffs if finite(d)]
    dmax = max(finite_d) if finite_d else float("nan")
    # growth: log10|diff| vs t over nonzero finite points from the first nonzero difference on
    pts = [(t[k], math.log10(d)) for k, d in enumerate(diffs) if finite(d) and d > 0 and finite(t[k])]
    slope = None
    if len(pts) >= 4:
        _, s, _ = linear_fit([p[0] for p in pts], [p[1] for p in pts])
        slope = s if finite(s) else None
    q = {}
    for frac in (0.25, 0.5, 0.75, 1.0):
        k = max(0, min(n - 1, int(round(frac * (n - 1)))))
        q[f"t{int(frac * 100)}"] = {"t": t[k], "diff": diffs[k]} if n else None
    return {"max_abs_diff": dmax, "scale": scale, "threshold": thr,
            "rel_max_diff": dmax / scale if finite(dmax) and finite(scale) and scale > 0 else None,
            "first_nonzero_t": t[first_nz] if first_nz is not None else None,
            "first_over_tol_t": t[first_over] if first_over is not None else None,
            "nonfinite_at_t": t[nonfinite] if nonfinite is not None else None,
            "growth_decades_per_fs": slope,
            "doubling_time_fs": (math.log10(2) / slope) if slope and slope > 0 else None,
            "checkpoints": q, "_diffs": diffs}


def _table_pairs(a, b, mode):
    files = []
    if mode in ("rt", "ehrenfest"):
        files = [("rt.data", a.rt(), b.rt()), ("rt_energy.data", a.rt_energy(), b.rt_energy())]
    elif mode == "dft_md":
        files = [("dft_md.data", a.dft_md(), b.dft_md())]
    return files


def _exact_table(name, ta, tb, rep, series):
    if ta is None or tb is None:
        return rep.add(f"exact:{name}", LAYER, FAIL, f"{name} missing in "
                       f"{'A' if ta is None else 'B'}")
    if ta.labels != tb.labels:
        return rep.add(f"exact:{name}", LAYER, FAIL, f"column layouts differ: {ta.labels} vs {tb.labels}")
    n = min(len(ta.tokens), len(tb.tokens))
    ndiff, first = 0, None
    for k in range(n):
        ra, rb = ta.tokens[k], tb.tokens[k]
        for c in range(max(len(ra), len(rb))):
            xa = ra[c] if c < len(ra) else None
            xb = rb[c] if c < len(rb) else None
            bad_a = xa is None or parse_number(xa)[1] != OK
            if xa != xb or bad_a:
                ndiff += 1
                if first is None:
                    first = (k, c, xa, xb)
    msgs = []
    if len(ta.tokens) != len(tb.tokens):
        msgs.append(f"row counts {len(ta.tokens)} vs {len(tb.tokens)}")
    if first is not None:
        k, c, xa, xb = first
        label = ta.labels[c] if c < len(ta.labels) else f"col{c + 1}"
        msgs.append(f"{ndiff} token(s) differ; first at row {k + 1} ({label}, t={ta.tokens[k][0]}): "
                    f"{xa} vs {xb}")
    if msgs:
        return rep.add(f"exact:{name}", LAYER, FAIL, "; ".join(msgs),
                       {"n_diff_tokens": ndiff, "rows_a": len(ta.tokens), "rows_b": len(tb.tokens)},
                       describe("cross_run_exact"))
    return rep.add(f"exact:{name}", LAYER, PASS, f"all {n} rows token-identical",
                   {"rows": n}, describe("cross_run_exact"))


def _exact_lines(name, pa, pb, rep, numeric_only=True):
    if not (os.path.exists(pa) and os.path.exists(pb)):
        return rep.add(f"exact:{name}", LAYER, FAIL, f"{name} missing in one run")
    with open(pa, errors="replace") as fa, open(pb, errors="replace") as fb:
        la, lb = fa.read().split("\n"), fb.read().split("\n")
    n = min(len(la), len(lb))
    ndiff, first = 0, None
    for k in range(n):
        if la[k].split() != lb[k].split():
            ndiff += 1
            if first is None:
                first = k
    if len(la) != len(lb) or ndiff:
        msg = f"{ndiff} line(s) differ" + (f", first at line {first + 1}" if first is not None else "")
        if len(la) != len(lb):
            msg += f"; line counts {len(la)} vs {len(lb)}"
        return rep.add(f"exact:{name}", LAYER, FAIL, msg, {"n_diff_lines": ndiff},
                       describe("cross_run_exact"))
    # also refuse non-finite tokens even if identical in both runs
    for k, line in enumerate(la):
        for tok in line.replace("#v=", " ").replace("#f=", " ").split():
            st = parse_number(tok)[1]
            if st in ("nonfinite", "overflow"):
                return rep.add(f"exact:{name}", LAYER, FAIL, f"identical but non-finite token "
                               f"{tok!r} at line {k + 1}")
    return rep.add(f"exact:{name}", LAYER, PASS, f"all {n} lines identical", {"lines": n},
                   describe("cross_run_exact"))


def _tol_table(name, ta, tb, run, rep, series, longitudinal):
    if ta is None or tb is None:
        return rep.add(f"tol:{name}", LAYER, FAIL, f"{name} missing in {'A' if ta is None else 'B'}")
    if ta.labels != tb.labels:
        return rep.add(f"tol:{name}", LAYER, FAIL, f"column layouts differ: {ta.labels} vs {tb.labels}")
    if not ta.has("Time"):
        return rep.add(f"tol:{name}", LAYER, FAIL, "no Time column")
    t_a, t_b = ta.col("Time"), tb.col("Time")
    n = min(len(t_a), len(t_b))
    if any(not finite(t_a[k], t_b[k]) or abs(t_a[k] - t_b[k]) > 1e-7 for k in range(n)):
        return rep.add(f"tol:{name}", LAYER, FAIL, "time grids differ: runs are not comparable "
                       "row by row (different dt or output step?)")
    tf = run.time_factor(ta.unit("Time")) or 1.0
    t = [x * tf for x in t_a[:n]]
    # vector quantities (Jm_x/y/z, E_ext_x/y/z, ...) share one scale: max over the components,
    # so a component that is zero by symmetry is judged against the vector's magnitude.
    group_scale = {}
    for label in tb.labels:
        g = label[:-2] if label[-2:] in ("_x", "_y", "_z") else label
        m = max_abs([x for x in tb.col(label)[:n] if finite(x)]) if n else float("nan")
        group_scale[g] = max(group_scale.get(g, 0.0), m) if finite(m) else group_scale.get(g, 0.0)
    cols, fails, worst = {}, [], None
    for c, label in enumerate(ta.labels):
        if label == "Time":
            continue
        atol, rtol, cls = column_tolerance(label, ta.units[c], run, longitudinal)
        g = label[:-2] if label[-2:] in ("_x", "_y", "_z") else label
        dv = divergence(t, ta.col(label), tb.col(label), atol, rtol, group_scale.get(g))
        series.setdefault(name, {})[label] = {"t": t, "diff": dv.pop("_diffs"), "threshold": dv["threshold"],
                                            "class": cls, "unit": ta.units[c]}
        dv["class"] = cls
        cols[label] = dv
        if dv["nonfinite_at_t"] is not None or dv["first_over_tol_t"] is not None:
            fails.append(label)
            if worst is None or (dv["first_over_tol_t"] or 0) < (cols[worst]["first_over_tol_t"] or 0):
                worst = label
    msgs = []
    if len(t_a) != len(t_b):
        msgs.append(f"row counts {len(t_a)} vs {len(t_b)} (compared the first {n})")
    metrics = {"columns": cols, "rows_compared": n}
    tol = {k: describe(k) for k in ("cross_run_energy_atol_per_atom", "cross_run_current_rtol",
                                    "cross_run_field_rtol")}
    if fails:
        d = cols[worst]
        msgs.insert(0, f"{len(fails)} column(s) exceed tolerance: {', '.join(fails[:6])}; first "
                    f"{worst} at t={fmt(d['first_over_tol_t'])} fs (max |d| {fmt(d['max_abs_diff'])}, "
                    f"threshold {fmt(d['threshold'])}, doubling time {fmt(d['doubling_time_fs'])} fs)")
        return rep.add(f"tol:{name}", LAYER, FAIL, "; ".join(msgs), metrics, tol)
    if msgs:
        return rep.add(f"tol:{name}", LAYER, FAIL, "; ".join(msgs), metrics, tol)
    firsts = [cols[l]["first_nonzero_t"] for l in cols if cols[l]["first_nonzero_t"] is not None]
    return rep.add(f"tol:{name}", LAYER, PASS,
                   f"{len(cols)} columns within tolerance over {n} rows; first nonzero difference at "
                   f"t={fmt(min(firsts)) if firsts else 'never'} fs", metrics, tol)


def _tol_trajectory(a, b, rep, series):
    ta, tb = a.trj(), b.trj()
    if ta is None or tb is None:
        return rep.add("tol:trj.xyz", LAYER, FAIL, "trj.xyz missing in one run")
    n = min(len(ta.frames), len(tb.frames))
    if n == 0:
        return rep.add("tol:trj.xyz", LAYER, FAIL, "no frames")
    pos_tol = T["cross_run_position_atol"]["fail"]
    vf_rtol = T["cross_run_velocity_force_rtol"]["fail"]
    t = [(f.step if f.step is not None else 0) * (a.dt_fs or 0) for f in ta.frames[:n]]
    out = {}
    for key in ("pos", "vel", "force"):
        da, scale = [], 0.0
        for fa, fb in zip(ta.frames[:n], tb.frames[:n]):
            va, vb = getattr(fa, key), getattr(fb, key)
            if len(va) != len(vb):
                return rep.add("tol:trj.xyz", LAYER, FAIL, f"atom counts differ at step {fa.step}")
            m = 0.0
            for x, y in zip(va, vb):
                for p, q in zip(x, y):
                    if not finite(p, q):
                        m = float("nan")
                        break
                    m = max(m, abs(p - q))
                    scale = max(scale, abs(q))
            da.append(m)
        if not da:
            continue
        thr = pos_tol if key == "pos" else vf_rtol * scale
        dv = divergence(t, da, [0.0] * len(da), thr, 0.0)  # da already holds |a-b| per frame
        dv["scale"], dv["threshold"] = scale, thr
        series.setdefault("trj.xyz", {})[key] = {"t": t, "diff": dv.pop("_diffs"), "threshold": thr,
                                                 "class": key, "unit": "A" if key == "pos" else "a.u."}
        out[key] = dv
    fails = [k for k, d in out.items() if d["nonfinite_at_t"] is not None or d["first_over_tol_t"] is not None]
    msgs = []
    if len(ta.frames) != len(tb.frames):
        msgs.append(f"frame counts {len(ta.frames)} vs {len(tb.frames)}")
    tol = {k: describe(k) for k in ("cross_run_position_atol", "cross_run_velocity_force_rtol")}
    if fails or msgs:
        msgs = [f"{k}: first over tolerance at t={fmt(out[k]['first_over_tol_t'])} fs, max |d| "
                f"{fmt(out[k]['max_abs_diff'])}" for k in fails] + msgs
        return rep.add("tol:trj.xyz", LAYER, FAIL, "; ".join(msgs), {"quantities": out}, tol)
    return rep.add("tol:trj.xyz", LAYER, PASS, f"positions/velocities/forces within tolerance over "
                   f"{n} frames", {"quantities": out}, tol)


def _compare_dc(a, b, rep, policy):
    sa, sb = a.stdout(), b.stdout()
    if sa is None or sb is None or not sa.dc_scf or not sb.dc_scf:
        return rep.add(f"{'exact' if policy == 'exact' else 'tol'}:dc_energy", LAYER, FAIL,
                       "DC #SCF records missing in one run")
    from .physics_dc import dc_energy_ev
    (ea, ra), (eb, rb) = dc_energy_ev(a, True), dc_energy_ev(b, True)
    natom = a.natom or 1
    metrics = {"e_a_ev": ea, "e_b_ev": eb, "iterations_a": len(sa.dc_scf), "iterations_b": len(sb.dc_scf)}
    from .physics_dc import fragment_energies
    fa = [e for _, e in fragment_energies(a)]
    fb = [e for _, e in fragment_energies(b)]
    hist_finite = all(finite(*x[:3]) for x in sa.dc_scf + sb.dc_scf)
    if policy == "exact":
        toks_a = [(r[0], r[1], r[2]) for r in sa.dc_scf]
        toks_b = [(r[0], r[1], r[2]) for r in sb.dc_scf]
        same = toks_a == toks_b and fa == fb and hist_finite and all(finite(x) for x in fa)
        st = PASS if same else FAIL
        rep.add("exact:dc_scf_history", LAYER, st, "DC SCF histories and fragment total energies " +
                ("identical" if st == PASS else "differ (or contain non-finite values)"), metrics,
                describe("cross_run_exact"))
    elif not hist_finite:
        rep.add("tol:dc_energy", LAYER, FAIL, "non-finite value in a DC SCF history", metrics)
    else:
        # the 'DC #SCF' field is e15.8: never demand more than its printed resolution
        tol = max(T["cross_run_energy_atol_per_atom"]["fail"], ((ra or 0) + (rb or 0)) / natom)
        d = (ea - eb) / natom if finite(ea, eb) else float("nan")
        metrics["diff_per_atom_ev"] = d
        metrics["print_resolution_per_atom_ev"] = ((ra or 0) + (rb or 0)) / natom
        st = PASS if finite(d) and abs(d) <= tol else FAIL
        rep.add("tol:dc_energy", LAYER, st, f"final DC energies differ by {fmt(d)} eV/atom (limit "
                f"{fmt(tol)})", metrics, describe("cross_run_energy_atol_per_atom"))
    # LCFO eigenvalues
    pa = a.path("eigen.data", sub=os.path.join("data_dcdft", "total"))
    pb = b.path("eigen.data", sub=os.path.join("data_dcdft", "total"))
    if policy == "exact":
        _exact_lines("lcfo_eigen", pa, pb, rep)
    else:
        _compare_eigen(a.dc_eigen(), b.dc_eigen(), rep, "tol:lcfo_eigen")


def _compare_eigen(ea, eb, rep, cid):
    if ea is None or eb is None:
        return rep.add(cid, LAYER, FAIL, "eigenvalue file missing in one run")
    if sorted(ea.blocks) != sorted(eb.blocks):
        return rep.add(cid, LAYER, FAIL, "different k/spin blocks")
    f = AU_ENERGY_EV if (ea.unit or "").lower() in ("a.u.", "au") else 1.0
    tol = T["cross_run_eigen_atol_ha"]["fail"] * AU_ENERGY_EV
    worst = 0.0
    for key in ea.blocks:
        va, vb = ea.values(*key), eb.values(*key)
        if len(va) != len(vb):
            return rep.add(cid, LAYER, FAIL, f"block {key}: {len(va)} vs {len(vb)} states")
        for x, y in zip(va, vb):
            if not finite(x, y):
                return rep.add(cid, LAYER, FAIL, f"non-finite eigenvalue in block {key}")
            worst = max(worst, abs(x - y) * f)
    st = PASS if worst <= tol else FAIL
    return rep.add(cid, LAYER, st, f"max eigenvalue difference {fmt(worst)} eV (limit {fmt(tol)} eV)",
                   {"max_diff_ev": worst}, describe("cross_run_eigen_atol_ha"))


def compare_runs(a, b, mode, policy="tolerance", rep=None):
    """Compare run A (candidate) against run B (reference). Returns (report, series)."""
    rep = rep or Report(f"{a.dir} vs {b.dir}", f"compare/{mode}/{policy}")
    series = {}
    tag = "exact" if policy == "exact" else "tol"   # check-id prefix, as in the per-file checks
    if a.nt != b.nt or (a.dt_fs and b.dt_fs and abs(a.dt_fs - b.dt_fs) > 1e-12):
        rep.notes.append(f"inputs differ in nt/dt: A nt={a.nt} dt={a.dt_fs} fs, B nt={b.nt} dt={b.dt_fs} fs")
    if a.natom != b.natom:
        rep.add(f"{tag}:inputs", LAYER, FAIL, f"natom differs: {a.natom} vs {b.natom}")
        return rep, series
    longitudinal = bool(a.inputs and str(a.inputs.get("trans_longi", "tr")).lower().startswith("lo"))
    if mode in ("rt", "ehrenfest", "dft_md"):
        for name, ta, tb in _table_pairs(a, b, mode):
            if policy == "exact":
                _exact_table(name, ta, tb, rep, series)
            else:
                _tol_table(name, ta, tb, a, rep, series, longitudinal)
        if mode in ("ehrenfest", "dft_md"):
            if policy == "exact":
                _exact_lines("trj.xyz", a.path("trj.xyz"), b.path("trj.xyz"), rep)
            else:
                _tol_trajectory(a, b, rep, series)
    elif mode == "gs":
        if policy == "exact":
            _exact_lines("eigen.data", a.path("eigen.data"), b.path("eigen.data"), rep)
        else:
            _compare_eigen(a.eigen(), b.eigen(), rep, "tol:eigen")
        ia, ib = a.info(), b.info()
        ea = ia.get("total_energy_ev") if ia else None
        eb = ib.get("total_energy_ev") if ib else None
        natom = a.natom or 1
        d = (ea - eb) / natom if finite(ea, eb) else float("nan")
        tol = 0.0 if policy == "exact" else T["cross_run_energy_atol_per_atom"]["fail"]
        st = PASS if finite(d) and abs(d) <= tol else FAIL
        rep.add(f"{tag}:gs_energy", LAYER, st, f"total energies differ by {fmt(d)} eV/atom "
                f"(limit {tol})", {"e_a_ev": ea, "e_b_ev": eb})
    elif mode == "dc":
        _compare_dc(a, b, rep, policy)
    else:
        rep.add(f"{tag}:mode", LAYER, UNKNOWN, f"no comparison defined for mode {mode}")
    return rep, series
