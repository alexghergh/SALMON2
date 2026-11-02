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
"""Layer 2 for divide-and-conquer DFT (theory='dft', yn_dc='y') and its RT follow-up.

What SALMON writes for DC (2.3.0 and develop c06a87e):
  outputfile    'DC #SCF = n Total Energy = E[eV] diff = d' per SCF iteration (scf_iteration_dft.f90);
                'start DC-LCFO' / 'end DC-LCFO'; 'lapack diag, #dim=' or 'eigenexa diag, #dim=';
                CheFSI: 'CheFSI cycle: n maximum residual: r' (lcfo.f90, lcfo_diag_chefsi.f90)
  data_dcdft/total/<sys>_eigen.data   LCFO eigenvalues of the whole system ('# spin= s', io esp)
  data_dcdft/fragments/NNNNNN/        per-fragment GS files and *.bin (basis, Hamiltonian, coefficients)
Not written: the DC Fermi level (ne2mu_dcdft keeps it internal), the LCFO overlap/orthonormality,
and per-fragment SCF residuals.
"""
import math
import os

from .numeric import finite, fmt, max_abs
from .run import AU_ENERGY_EV
from .tolerances import TOLERANCES as T, describe
from .verdict import Report, PASS, FAIL, WARN, UNKNOWN, NA
from .physics_md import energy_series

LAYER = "physics"


def fragment_energies(run):
    """'Total energy (eV)' from data_dcdft/fragments/*/<sys>_info.data (full precision).

    For DC, main_dft.f90 calls write_info_data while base_directory points at the fragment
    directory, so each fragment root writes the global DC total energy there.
    """
    import glob
    from .readers import read_info
    out = []
    for p in sorted(glob.glob(os.path.join(run.dir, "data_dcdft", "fragments", "*",
                                           f"{run.sysname}_info.data"))):
        out.append((p, read_info(p).get("total_energy_ev")))
    return out


def e15_resolution(x):
    """Half a unit in the last place of a Fortran e15.8 value (the 'DC #SCF' energy field)."""
    if not finite(x) or x == 0:
        return 0.0
    return 0.5 * 10 ** (math.floor(math.log10(abs(x))) + 1 - 8)


def dc_energy_ev(run, with_resolution=False):
    """DC total energy: fragment info files if present, else the last 'DC #SCF' line (e15.8)."""
    frags = fragment_energies(run)
    if frags and frags[0][1] is not None:
        e, res = frags[0][1], 1e-12 * abs(frags[0][1])
    else:
        so = run.stdout()
        if so is None or not so.dc_scf:
            return (None, None) if with_resolution else None
        e = so.dc_scf[-1][1]
        res = e15_resolution(e)
    return (e, res) if with_resolution else e


def check_fragment_energies(run, rep):
    cid = "dc_fragment_energy_consistency"
    frags = fragment_energies(run)
    if not frags:
        return rep.add(cid, LAYER, NA, "no data_dcdft/fragments/*/<sys>_info.data files")
    vals = [e for _, e in frags]
    if not all(finite(v) for v in vals):
        return rep.add(cid, LAYER, FAIL, "non-finite or missing total energy in a fragment info file")
    so = run.stdout()
    last = so.dc_scf[-1][1] if so and so.dc_scf else None
    spread = max(vals) - min(vals)
    metrics = {"n_fragments": len(vals), "spread_ev": spread, "dc_scf_last_ev": last}
    if spread > 1e-9 * max(1.0, abs(vals[0])):
        return rep.add(cid, LAYER, FAIL, f"fragments report different total energies (spread "
                       f"{fmt(spread)} eV)", metrics)
    if finite(last) and abs(last - vals[0]) > e15_resolution(last) * 1.01:
        return rep.add(cid, LAYER, FAIL, f"fragment total energy {fmt(vals[0], 12)} eV disagrees with "
                       f"the last 'DC #SCF' line {fmt(last, 12)} eV", metrics)
    return rep.add(cid, LAYER, PASS, f"{len(vals)} fragment info files agree on E = {fmt(vals[0], 12)} eV",
                   metrics)


def conventional_energy_ev(run):
    info = run.info()
    if info is not None and finite(info.get("total_energy_ev")):
        return info["total_energy_ev"]
    so = run.stdout()
    if so is not None and so.gs_iter_energy:
        return so.gs_iter_energy[-1][1]
    return None


def _eig_ev(eg):
    if eg is None or not eg.blocks:
        return None
    unit = (eg.unit or "").strip().lower()
    f = 1.0 if unit == "ev" else (AU_ENERGY_EV if unit in ("a.u.", "au") else None)
    if f is None:
        return None
    key = sorted(eg.blocks)[0]
    return [(io, e * f if finite(e) else float("nan")) for io, e, _ in eg.blocks[key]]


def check_lcfo(run, rep):
    cid = "lcfo_output"
    so = run.stdout()
    lcfo_on = not (run.inputs and str(run.inputs.get("yn_dc_lcfo", "y")).lower()[:1] == "n")
    if not lcfo_on:
        return rep.add(cid, LAYER, NA, "yn_dc_lcfo='n'")
    issues, warns, metrics = [], [], {}
    if so is None:
        return rep.add(cid, LAYER, UNKNOWN, "outputfile missing")
    if not so.lcfo_start or not so.lcfo_end:
        issues.append(f"LCFO markers: start={so.lcfo_start} end={so.lcfo_end}")
    eg = _eig_ev(run.dc_eigen())
    nstate = run.nstate
    if eg is None:
        issues.append("data_dcdft/total eigen file missing or unreadable")
    else:
        idx = [io for io, _ in eg]
        vals = [e for _, e in eg]
        metrics["n_eigenvalues"] = len(vals)
        if nstate and idx != list(range(1, nstate + 1)):
            issues.append(f"eigen file has indices {idx[:3]}..{idx[-3:]} ({len(idx)}), expected 1..{nstate}")
        if not all(finite(v) for v in vals):
            issues.append("non-finite LCFO eigenvalue")
        elif any(b < a - 1e-12 for a, b in zip(vals, vals[1:])):
            warns.append("LCFO eigenvalues not in ascending order")
    if so.lcfo_dim:
        metrics["lcfo_basis_dim"] = so.lcfo_dim[-1]
        if run.natom:
            metrics["basis_per_atom"] = so.lcfo_dim[-1] / run.natom
        if nstate and so.lcfo_dim[-1] < T["dc_lcfo_basis_min_ratio"]["fail"] * nstate:
            issues.append(f"LCFO basis dimension {so.lcfo_dim[-1]} < nstate {nstate}")
    if so.lcfo_complex:
        # develop lcfo_complex.f90 prints, per k/spin: Hermiticity error of H, LCFO basis
        # orthonormality error, eigenvector orthonormality error and eigen-residual; it stops
        # itself above lcfo_tol = 1e-10. The real LCFO path (aSi-512) prints none of these.
        worst = max((max(r[2:]) for r in so.lcfo_complex if all(finite(x) for x in r[2:])), default=None)
        metrics["complex_lcfo_max_error"] = worst
        metrics["complex_lcfo_records"] = len(so.lcfo_complex)
        if any(not all(finite(x) for x in r[2:]) for r in so.lcfo_complex):
            issues.append("non-finite complex LCFO orthonormality/residual diagnostic")
        elif worst is not None and worst > 1e-10:
            issues.append(f"complex LCFO orthonormality/residual error {fmt(worst)} > 1e-10 (lcfo_tol)")
    else:
        metrics["orthonormality"] = "not printed by the real-orbital LCFO path"
    solver = str(run.inputs.get("lcfo_eigensolver", "")).lower() if run.inputs else ""
    if solver == "chefsi":
        tolc = run.inputs.get("lcfo_diag_chefsi_residual_tolerance", 1e-7)
        r = so.chefsi_residuals
        metrics["chefsi_final_residual"] = r[-1] if r else None
        if not r or not all(finite(x) and x >= 0 for x in r):
            issues.append("CheFSI residual history missing or non-finite")
        elif r[-1] > tolc:
            issues.append(f"CheFSI final residual {fmt(r[-1])} > tolerance {fmt(tolc)}")
    if issues:
        return rep.add(cid, LAYER, FAIL, "; ".join(issues), metrics,
                       describe("dc_lcfo_basis_min_ratio"))
    if warns:
        return rep.add(cid, LAYER, WARN, "; ".join(warns), metrics)
    return rep.add(cid, LAYER, PASS, f"LCFO completed: {metrics.get('n_eigenvalues')} finite "
                   f"eigenvalues" + (f", basis {metrics['lcfo_basis_dim']} "
                                     f"({fmt(metrics.get('basis_per_atom'))}/atom)"
                                     if "lcfo_basis_dim" in metrics else ""), metrics)


def check_dc_history(run, rep):
    cid = "dc_scf_history"
    so = run.stdout()
    if so is None or not so.dc_scf:
        return rep.add(cid, LAYER, UNKNOWN, "no DC #SCF records")
    E = [r[1] for r in so.dc_scf]
    if not all(finite(e) for e in E):
        return rep.add(cid, LAYER, FAIL, "non-finite DC total energy in the SCF history")
    natom = run.natom or 1
    last = abs(E[-1] - E[-2]) / natom if len(E) > 1 else float("nan")
    metrics = {"iterations": len(E), "final_energy_ev": E[-1],
               "last_step_change_per_atom_ev": last}
    if len(E) > 1 and last > 1e-4:
        return rep.add(cid, LAYER, WARN, f"energy still changing by {fmt(last)} eV/atom in the last "
                       "DC iteration although the density residual passed", metrics)
    return rep.add(cid, LAYER, PASS, f"{len(E)} DC iterations, final E = {fmt(E[-1], 12)} eV, last "
                   f"change {fmt(last)} eV/atom", metrics)


def check_dc_vs_conventional(dc, conv, rep, conv_report=None):
    cid = "dc_vs_conventional_energy"
    if conv is None:
        return rep.add(cid, LAYER, NA, "no conventional reference given (--conventional)")
    if conv_report is not None and conv_report.verdict() not in ("pass", "warn"):
        return rep.add(cid, LAYER, UNKNOWN, f"conventional reference did not pass its own checks "
                       f"({conv_report.verdict()}); comparison not trusted")
    ed, ec = dc_energy_ev(dc), conventional_energy_ev(conv)
    natom = dc.natom
    if conv.natom != dc.natom or not natom:
        return rep.add(cid, LAYER, FAIL, f"atom counts differ (DC {dc.natom}, conventional {conv.natom})")
    if not finite(ed, ec):
        return rep.add(cid, LAYER, FAIL, f"non-finite or missing energies (DC {ed}, conventional {ec})")
    d = (ed - ec) / natom
    tl = T["dc_vs_conventional_energy_per_atom"]
    notes = []
    tdc, tcv = dc.temperature_k, conv.temperature_k
    if tdc != tcv:
        notes.append(f"electronic temperature differs: DC {tdc} K, conventional {tcv} (smearing; "
                     "negligible for a gapped system at 300 K)")
    metrics = {"e_dc_ev": ed, "e_conv_ev": ec, "diff_per_atom_ev": d, "notes": notes}
    tol = describe("dc_vs_conventional_energy_per_atom")
    if abs(d) > tl["fail"]:
        return rep.add(cid, LAYER, FAIL, f"E_DC - E_conv = {fmt(d)} eV/atom (> {tl['fail']})", metrics, tol)
    st = WARN if abs(d) > tl["warn"] else PASS
    return rep.add(cid, LAYER, st, f"E_DC - E_conv = {fmt(d)} eV/atom" +
                   (f"; {notes[0]}" if notes else ""), metrics, tol)


def spectrum_summary(eig, nelec, natom):
    """HOMO, LUMO, gap, mid-gap and band energy (eV) for spin-unpolarised occupation."""
    if eig is None or not nelec:
        return None
    vals = [e for _, e in eig]
    nocc = int(round(nelec / 2))
    if len(vals) <= nocc or not all(finite(v) for v in vals[:nocc + 1]):
        return None
    homo, lumo = vals[nocc - 1], vals[nocc]
    return {"homo": homo, "lumo": lumo, "gap": lumo - homo, "midgap": 0.5 * (homo + lumo),
            "band_energy_per_atom": 2.0 * sum(vals[:nocc]) / natom, "nocc": nocc}


def check_spectrum(dc, conv, rep):
    cid = "dc_spectrum_vs_conventional"
    if conv is None:
        return rep.add(cid, LAYER, NA, "no conventional reference")
    a = spectrum_summary(_eig_ev(dc.dc_eigen()), dc.nelec, dc.natom)
    b = spectrum_summary(_eig_ev(conv.eigen()), conv.nelec, conv.natom)
    if a is None or b is None:
        return rep.add(cid, LAYER, UNKNOWN, "eigenvalues missing, non-finite or fewer than nelec/2+1")
    th, tb = T["dc_homo_lumo_ev"], T["dc_band_energy_per_atom"]
    diffs = {k: a[k] - b[k] for k in ("homo", "lumo", "gap", "midgap", "band_energy_per_atom")}
    metrics = {"dc": a, "conventional": b, "diff": diffs,
               "note": "mid-gap is the Fermi-level proxy: SALMON does not print the DC Fermi level"}
    worst_e = max(abs(diffs[k]) for k in ("homo", "lumo", "midgap"))
    tol = {"levels": describe("dc_homo_lumo_ev"), "band": describe("dc_band_energy_per_atom")}
    if worst_e > th["fail"] or abs(diffs["band_energy_per_atom"]) > tb["fail"]:
        return rep.add(cid, LAYER, FAIL, f"HOMO/LUMO/mid-gap differ by up to {fmt(worst_e)} eV, band "
                       f"energy by {fmt(diffs['band_energy_per_atom'])} eV/atom", metrics, tol)
    st = WARN if (worst_e > th["warn"] or abs(diffs["band_energy_per_atom"]) > tb["warn"]) else PASS
    return rep.add(cid, LAYER, st, f"HOMO {fmt(diffs['homo'])}, LUMO {fmt(diffs['lumo'])}, mid-gap "
                   f"{fmt(diffs['midgap'])} eV; band energy {fmt(diffs['band_energy_per_atom'])} eV/atom",
                   metrics, tol)


def check_buffers(dcs, conv, rep):
    """Sensitivity of E_DC to the buffer size (num_rgrid_buffer)."""
    cid = "dc_buffer_sensitivity"
    if not dcs:
        return rep.add(cid, LAYER, NA, "no buffer series given (--buffer-dirs)")
    ec = conventional_energy_ev(conv) if conv is not None else None
    rows = []
    for r in dcs:
        b = r.inputs.get("num_rgrid_buffer", None, index=None) if r.inputs else None
        e = dc_energy_ev(r)
        rows.append({"dir": r.dir, "buffer": b, "e_dc_ev": e,
                     "err_per_atom_ev": (e - ec) / r.natom if finite(e, ec) and r.natom else None})
    if any(not finite(x["e_dc_ev"]) for x in rows):
        return rep.add(cid, LAYER, FAIL, "a buffer run has no finite DC energy", {"runs": rows})
    rows.sort(key=lambda x: (x["buffer"] or [0])[0] if isinstance(x["buffer"], list) else 0)
    es = [x["e_dc_ev"] / (dcs[0].natom or 1) for x in rows]
    metrics = {"runs": rows, "spread_per_atom_ev": max(es) - min(es),
               "paper_reference": "arXiv:2606.21125 Fig. 3b (I=1e9 W/cm^2): RT excitation-energy "
                                  "deviation ~8e-4 (b=10), ~3e-4 (b=15), ~2e-4 (b=20) eV/atom; "
                                  "no ground-state energies reported"}
    if ec is None:
        return rep.add(cid, LAYER, PASS, f"spread of E_DC over buffers {fmt(metrics['spread_per_atom_ev'])} "
                       "eV/atom (no conventional reference: monotonicity not checked)", metrics)
    errs = [abs(x["err_per_atom_ev"]) for x in rows]
    mono = all(b <= a * 1.05 + 1e-6 for a, b in zip(errs, errs[1:]))
    if not mono:
        return rep.add(cid, LAYER, WARN, f"|E_DC - E_conv| does not decrease with buffer size: "
                       f"{[fmt(e) for e in errs]} eV/atom", metrics, describe("dc_buffer_monotonic"))
    return rep.add(cid, LAYER, PASS, f"|E_DC - E_conv| decreases with buffer: {[fmt(e) for e in errs]} "
                   "eV/atom", metrics, describe("dc_buffer_monotonic"))


def _field_component(run):
    rt = run.rt()
    best, bmax = "z", -1.0
    for c in "xyz":
        col = rt.col(f"E_ext_{c}") if rt and rt.has(f"E_ext_{c}") else None
        if col:
            m = max_abs(col)
            if finite(m) and m > bmax:
                best, bmax = c, m
    return best


def check_rt_dc_vs_conv(rt_dc, rt_conv, rep, window_fs=None):
    """RT started from DC-LCFO orbitals vs RT from the conventional GS (paper Figs. 3-4)."""
    cid = "rt_from_dc_vs_conventional"
    if rt_dc is None or rt_conv is None:
        return rep.add(cid, LAYER, NA, "need both --rt-dc and --rt-conv")
    ea, eb = energy_series(rt_dc), energy_series(rt_conv)
    A, B = rt_dc.rt(), rt_conv.rt()
    if ea is None or eb is None or A is None or B is None:
        return rep.add(cid, LAYER, UNKNOWN, "rt.data/rt_energy.data missing in one run")
    if len(ea["t_fs"]) != len(eb["t_fs"]) or any(abs(x - y) > 1e-7 for x, y in zip(ea["t_fs"], eb["t_fs"])):
        return rep.add(cid, LAYER, FAIL, "time grids differ: the two RT runs are not comparable")
    if len(A.rows) != len(B.rows):
        return rep.add(cid, LAYER, FAIL, "rt.data lengths differ")
    natom = rt_conv.natom or 1
    tmax = window_fs if window_fs else ea["t_fs"][-1]
    ka = [k for k, t in enumerate(ea["t_fs"]) if t <= tmax + 1e-9]
    xa = [ea["Eall-Eall0"][k] / natom for k in ka]
    xb = [eb["Eall-Eall0"][k] / natom for k in ka]
    if not all(finite(v) for v in xa + xb):
        return rep.add(cid, LAYER, FAIL, "non-finite excitation energy")
    dmax = max(abs(a - b) for a, b in zip(xa, xb))
    scale = max_abs(xb)
    te = T["rt_dc_vs_conv_excitation_per_atom"]
    lim = max(te["fail_abs"], te["fail_rel"] * scale)
    c = _field_component(rt_conv)
    tf = rt_conv.time_factor(B.unit("Time"))
    ja, jb, tt = A.col(f"Jm_{c}"), B.col(f"Jm_{c}"), [x * tf for x in B.col("Time")]
    sel = [k for k, t in enumerate(tt) if t <= tmax + 1e-9]
    if not all(finite(ja[k], jb[k]) for k in sel):
        return rep.add(cid, LAYER, FAIL, "non-finite current")
    num = math.sqrt(sum((ja[k] - jb[k]) ** 2 for k in sel) / len(sel))
    den = math.sqrt(sum(jb[k] ** 2 for k in sel) / len(sel))
    nrms = num / den if den > 0 else float("inf")
    early = [k for k in sel if tt[k] <= 5.0]
    nrms5 = (math.sqrt(sum((ja[k] - jb[k]) ** 2 for k in early) / len(early)) /
             max(math.sqrt(sum(jb[k] ** 2 for k in early) / len(early)), 1e-300)) if early else None
    tj = T["rt_dc_vs_conv_current_nrms"]
    metrics = {"window_fs": tmax, "max_excitation_diff_per_atom_ev": dmax,
               "conv_max_excitation_per_atom_ev": scale, "limit_per_atom_ev": lim,
               "current_component": c, "current_nrms": nrms, "current_nrms_first_5fs": nrms5,
               "final_excitation_dc_per_atom_ev": xa[-1], "final_excitation_conv_per_atom_ev": xb[-1]}
    tol = {"energy": describe("rt_dc_vs_conv_excitation_per_atom"),
           "current": describe("rt_dc_vs_conv_current_nrms")}
    fails = []
    if dmax > lim:
        fails.append(f"excitation energy differs by {fmt(dmax)} eV/atom (> {fmt(lim)})")
    if not finite(nrms) or nrms > tj["fail"]:
        fails.append(f"current NRMS {fmt(nrms)} > {tj['fail']}")
    if fails:
        return rep.add(cid, LAYER, FAIL, "; ".join(fails), metrics, tol)
    st = WARN if (dmax > te["warn_frac"] * lim or nrms > tj["warn"]) else PASS
    return rep.add(cid, LAYER, st, f"excitation energy within {fmt(dmax)} eV/atom (limit {fmt(lim)}), "
                   f"current NRMS {fmt(nrms)} over {fmt(tmax)} fs", metrics, tol)


def dc_report(run, rep=None, conv=None, conv_report=None, buffers=None):
    rep = rep or Report(run.dir, "dc")
    check_dc_history(run, rep)
    check_fragment_energies(run, rep)
    check_lcfo(run, rep)
    check_dc_vs_conventional(run, conv, rep, conv_report)
    check_spectrum(run, conv, rep)
    if buffers:
        check_buffers(buffers, conv, rep)
    return rep
