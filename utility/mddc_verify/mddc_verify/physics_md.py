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
"""Layer 2 for Ehrenfest MD (tddft_* + yn_md='y'), Born-Oppenheimer dft_md, and plain RT.

Quantities come only from SALMON's own outputs:
  <sys>_rt_energy.data  Time, Eall, Eall-Eall0 [, components] [, Tion, Temperature_ion, E_work]
  <sys>_rt.data         Time, Ac_ext, E_ext, Ac_tot, E_tot, Jm [, Jmi]   (every step)
  <sys>_trj.xyz         positions [A], velocities [a.u.], forces [a.u.]   (every out_rvf_rt_step)
  <sys>_dft_md.data     Time, Tene, Uene, Uene-Uene0, E_work, E_nh, E_tot, E_tot-E_tot0, Hnvt, Hnvt', T
  outputfile            Mass(ik), Zps(ik); RT step lines (electron count); dft_md step lines (SCF residual)
"""
import math

from .numeric import finite, fmt, linear_fit, max_abs
from .run import AU_ENERGY_EV, AU_TIME_FS, AU_LENGTH_AA, UMASS, KB_AU
from .tolerances import TOLERANCES as T, describe
from .verdict import Report, PASS, FAIL, WARN, UNKNOWN, NA

LAYER = "physics"


# --------------------------------------------------------------------------- series
def energy_series(run):
    """Return dict of lists in eV/fs from rt_energy.data, or None."""
    t = run.rt_energy()
    if t is None or not t.has("Time"):
        return None
    tf = run.time_factor(t.unit("Time"))
    out = {"t_fs": [x * tf if finite(x) else float("nan") for x in t.col("Time")]} if tf else None
    if out is None:
        return None
    for label in ("Eall", "Eall-Eall0", "Tion", "E_work", "E_kin", "E_h", "E_ion", "E_xc", "E_ion_ion"):
        if t.has(label):
            f = run.energy_factor(t.unit(label))
            out[label] = [x * f if finite(x) else float("nan") for x in t.col(label)] if f else None
    if t.has("Temperature_ion"):
        out["Temperature_ion"] = t.col("Temperature_ion")
    return out


def _nan_to(x):
    return x if finite(x) else float("nan")


def field_work(run):
    """Cumulative field work W(t) at every rt.data step, left- and right-point rules.

    Electrons: dE/dt = -Omega J_e.E with E = -dAc/dt, so W_e = Omega sum J_e.dAc (validated on
    real fixed-ion silicon runs). Ions carry a positive-charge electric current J_i (Jmi,
    time_evolution_step.f90 calc_current_ion), so W_i = -Omega sum J_i.dAc.
    Units: A_eV_fs: J [1/(fs A^2)] x Ac [fs V/A] x Omega [A^3] = eV; a.u.: Hartree.
    """
    rt = run.rt()
    vol = run.volume_aa3()
    if rt is None or vol is None:
        return None
    tf = run.time_factor(rt.unit("Time"))
    acl = [l for l in ("Ac_tot_x", "Ac_tot_y", "Ac_tot_z")] if rt.has("Ac_tot_x") else \
        [l for l in ("Ac_ext_x", "Ac_ext_y", "Ac_ext_z")]
    el = ["E_tot_x", "E_tot_y", "E_tot_z"] if rt.has("E_tot_x") else ["E_ext_x", "E_ext_y", "E_ext_z"]
    jl = ["Jm_x", "Jm_y", "Jm_z"]
    il = ["Jmi_x", "Jmi_y", "Jmi_z"]
    if not all(rt.has(l) for l in acl + el + jl) or tf is None:
        return None
    au = (rt.unit("Jm_x") or "").strip().lower() in ("a.u.", "au")
    omega = vol / AU_LENGTH_AA ** 3 if au else vol
    efac = AU_ENERGY_EV if au else 1.0
    A = [rt.col(l) for l in acl]
    E = [rt.col(l) for l in el]
    J = [rt.col(l) for l in jl]
    I = [rt.col(l) for l in il] if all(rt.has(l) for l in il) else None
    n = len(rt.rows)
    t = [0.0] + [_nan_to(x) * tf for x in rt.col("Time")]
    prevA = [0.0, 0.0, 0.0]
    prevJ = [0.0, 0.0, 0.0]
    wl, wr, wl_sum, wr_sum, emax = [0.0], [0.0], 0.0, 0.0, 0.0
    for k in range(n):
        a = [_nan_to(A[c][k]) for c in range(3)]
        j = [_nan_to(J[c][k]) - (_nan_to(I[c][k]) if I else 0.0) for c in range(3)]
        da = [a[c] - prevA[c] for c in range(3)]
        wl_sum += omega * sum(prevJ[c] * da[c] for c in range(3)) * efac
        wr_sum += omega * sum(j[c] * da[c] for c in range(3)) * efac
        wl.append(wl_sum)
        wr.append(wr_sum)
        emax = max(emax, max(abs(_nan_to(E[c][k])) for c in range(3)))
        prevA, prevJ = a, j
    field_on = [False] + [any(_nan_to(E[c][k]) != 0.0 for c in range(3)) for k in range(n)]
    return {"t_fs": t, "W_left": wl, "W_right": wr, "max_E": emax, "field_on": field_on,
            "has_ion_current": I is not None}


def _interp_index(times, t, tol):
    """Index i with |times[i]-t| <= tol (times sorted), else None."""
    lo, hi = 0, len(times) - 1
    while lo <= hi:
        mid = (lo + hi) // 2
        if times[mid] < t - tol:
            lo = mid + 1
        elif times[mid] > t + tol:
            hi = mid - 1
        else:
            return mid
    return None


def tion_row0_unwritten(es):
    """True when the t=0 row's Tion is SALMON's placeholder, not a value.

    write_rt_energy_data (src/io/write.f90) writes the t=0 row before md%Tene is first computed
    (src/rt/md.f90), so with moving ions that row has Tion = 0 while later rows hold the ions'
    kinetic energy. H(0) = Eall(0) + 0 is then not the conserved quantity, and that row is skipped.
    Eall itself is electronic + ion-ion energy only (src/common/total_energy.f90)."""
    tion = es.get("Tion") if es else None
    return bool(tion) and len(tion) > 1 and tion[0] == 0.0 and any(x != 0.0 for x in tion[1:])


def total_energy(run, es):
    """H = Eall + Tion (Ehrenfest) or Eall (fixed ions)."""
    if es is None or es.get("Eall") is None:
        return None
    if es.get("Tion") is not None:
        return [a + b for a, b in zip(es["Eall"], es["Tion"])]
    return list(es["Eall"])


# --------------------------------------------------------------------------- checks
def _conservation(rep, cid, t_fs, H, natom, label):
    tolm, told = T["energy_conservation_max_per_atom"], T["energy_drift_per_atom_per_ps"]
    if len(H) < 3:
        return rep.add(cid, LAYER, NA, f"fewer than 3 energy rows in the window ({label})")
    if not all(finite(x) for x in H) or not all(finite(x) for x in t_fs):
        return rep.add(cid, LAYER, FAIL, f"non-finite energy or time in the window ({label})")
    d = [(h - H[0]) / natom for h in H]
    mx = max_abs(d)
    span = t_fs[-1] - t_fs[0]
    _, slope, rms = linear_fit(t_fs, d)
    drift = slope * 1e6 if finite(slope) else float("nan")  # eV/atom/fs -> meV/atom/ps
    metrics = {"window_fs": [t_fs[0], t_fs[-1]], "n_rows": len(H), "max_abs_per_atom_ev": mx,
               "drift_mev_per_atom_per_ps": drift, "rms_after_fit_per_atom_ev": rms}
    tol = {"max": describe("energy_conservation_max_per_atom"),
           "drift": describe("energy_drift_per_atom_per_ps")}
    if not finite(mx) or mx > tolm["fail"]:
        return rep.add(cid, LAYER, FAIL, f"{label}: max |dH|/N = {fmt(mx)} eV/atom > "
                       f"{tolm['fail']} over {fmt(span)} fs", metrics, tol)
    drift_applies = span >= 10.0
    if drift_applies and (not finite(drift) or abs(drift) > told["fail"]):
        return rep.add(cid, LAYER, FAIL, f"{label}: drift {fmt(drift)} meV/atom/ps > {told['fail']}",
                       metrics, tol)
    if mx > tolm["warn"] or (drift_applies and abs(drift) > told["warn"]):
        return rep.add(cid, LAYER, WARN, f"{label}: max |dH|/N = {fmt(mx)} eV/atom, drift "
                       f"{fmt(drift)} meV/atom/ps (above warn level)", metrics, tol)
    return rep.add(cid, LAYER, PASS, f"{label}: max |dH|/N = {fmt(mx)} eV/atom, drift "
                   f"{fmt(drift)} meV/atom/ps over {fmt(span)} fs"
                   + ("" if drift_applies else " (window < 10 fs: drift reported only)"),
                   metrics, tol)


def check_conservation(run, rep):
    """Energy conservation in field-free stretches (NVE)."""
    cid = "energy_conservation"
    es = energy_series(run)
    H = total_energy(run, es)
    if H is None:
        return rep.add(cid, LAYER, UNKNOWN, "rt_energy.data missing or without Eall")
    if run.ensemble == "NVT":
        return rep.add(cid, LAYER, NA, "NVT ensemble: H is not conserved (thermostat)")
    natom = run.natom or 1
    fw = field_work(run)
    t = es["t_fs"]
    impulse = run.ae_shape1 == "impulse"
    if fw is None:
        if (run.intensity in (None, 0, 0.0)) and run.e_amplitude in (None, 0, 0.0):
            free = [True] * len(t)
        else:
            return rep.add(cid, LAYER, UNKNOWN, "rt.data or the cell volume is unavailable: "
                           "cannot locate field-free windows")
    else:
        # energy row k is field-free if E == 0 at every rt.data step in (t_{k-1}, t_k]
        free = []
        j = 0
        tw = fw["t_fs"]
        on = fw["field_on"]
        prev = -1e300
        for tk in t:
            ok = True
            while j < len(tw) and tw[j] <= tk + 1e-9:
                if tw[j] > prev + 1e-9 and on[j]:
                    ok = False
                j += 1
            free.append(ok)
            prev = tk
    if impulse and free:
        free[0] = False  # the kick acts at t=0: H(0) is the pre-kick ground state
    skip0 = tion_row0_unwritten(es)
    if skip0 and free:
        free[0] = False  # Tion not yet written at t=0 (see tion_row0_unwritten)
    best, cur = (0, -1), None
    for k, f in enumerate(free):
        if f:
            cur = (cur[0], k) if cur else (k, k)
            if cur[1] - cur[0] > best[1] - best[0]:
                best = cur
        else:
            cur = None
    k0, k1 = best
    if k1 - k0 + 1 < 3:
        return rep.add(cid, LAYER, NA, "no field-free window of >= 3 energy rows (field on "
                       "throughout); the energy-balance check covers this run",
                       {"field_free_rows": sum(free)})
    label = ("Eall+Tion" if es.get("Tion") is not None else "Eall (fixed ions)")
    if skip0:
        label += ", t=0 row skipped (Tion not yet written there)"
    return _conservation(rep, cid, t[k0:k1 + 1], H[k0:k1 + 1], natom, label)


def check_energy_balance(run, rep):
    """With a laser: H(t)-H(0) must equal the field work W(t) within its quadrature ambiguity."""
    cid = "energy_balance"
    es = energy_series(run)
    H = total_energy(run, es)
    fw = field_work(run)
    if H is None or fw is None:
        return rep.add(cid, LAYER, UNKNOWN, "need rt_energy.data, rt.data and the cell (&system al)")
    if fw["max_E"] == 0.0:
        return rep.add(cid, LAYER, NA, "no field in rt.data: covered by energy_conservation")
    if run.ae_shape1 == "impulse":
        return rep.add(cid, LAYER, NA, "impulsive kick: the work integral is not resolved by the "
                       "sampled field; covered by energy_conservation after t=0")
    if run.ensemble == "NVT":
        return rep.add(cid, LAYER, NA, "NVT ensemble")
    tw = fw["t_fs"]
    resid, quad = [], []
    k_ref = 1 if tion_row0_unwritten(es) else 0  # measure from the first row with a real Tion
    i_ref = _interp_index(tw, es["t_fs"][k_ref], 1e-6) if len(es["t_fs"]) > k_ref else None
    if i_ref is None:
        return rep.add(cid, LAYER, FAIL, "the reference energy row has no rt.data step")
    for k, tk in enumerate(es["t_fs"]):
        if k < k_ref:
            continue
        i = _interp_index(tw, tk, 1e-6)
        if i is None:
            return rep.add(cid, LAYER, FAIL, f"energy row t={fmt(tk)} fs has no rt.data step")
        resid.append(H[k] - H[k_ref] - (fw["W_left"][i] - fw["W_left"][i_ref]))
        quad.append(abs((fw["W_right"][i] - fw["W_right"][i_ref]) - (fw["W_left"][i] - fw["W_left"][i_ref])))
    natom = run.natom or 1
    if not all(finite(x) for x in resid + quad):
        return rep.add(cid, LAYER, FAIL, "non-finite energy or field work")
    rmax, qmax = max_abs(resid), max(quad)
    wmax = max_abs(fw["W_left"])
    floor = 1e-9 * natom
    ratio = rmax / qmax if qmax > 0 else float("inf")
    tl = T["energy_balance_ratio"]
    metrics = {"max_abs_residual_ev": rmax, "max_quadrature_ambiguity_ev": qmax,
               "ratio": ratio, "max_abs_work_ev": wmax, "residual_end_ev": resid[-1],
               "min_detectable_energy_fault_per_atom_ev": qmax / natom,
               "includes_ion_work": fw["has_ion_current"]}
    tol = describe("energy_balance_ratio")
    if rmax <= floor:
        return rep.add(cid, LAYER, PASS, f"residual {fmt(rmax)} eV below rounding floor", metrics, tol)
    if not finite(ratio) or ratio > tl["fail"]:
        return rep.add(cid, LAYER, FAIL, f"|dH - W| up to {fmt(rmax)} eV exceeds the work-integral "
                       f"ambiguity {fmt(qmax)} eV (ratio {fmt(ratio)} > {tl['fail']})", metrics, tol)
    st = WARN if ratio > tl["warn"] else PASS
    return rep.add(cid, LAYER, st, f"|dH - W| <= {fmt(rmax)} eV = {fmt(ratio)} x the quadrature "
                   f"ambiguity ({fmt(qmax)} eV; peak field work {fmt(wmax)} eV)", metrics, tol)


def _force_term(run, trj, times_fs, dt_au):
    """Exact velocity-Verlet term sum_i |F_i(t)|^2 dt^2/(8 m_i) at the given times (eV), or None."""
    masses = run.element_masses_amu()
    if trj is None or not trj.has_force or masses is None:
        return None, None
    by_time = {}
    for f in trj.frames:
        if f.step is not None and f.step >= 0 and run.dt_fs:
            by_time[round(f.step * run.dt_fs, 9)] = f
    out = []
    for t in times_fs:
        fr = by_time.get(round(t, 9))
        if fr is None or len(fr.force) != len(masses):
            out.append(None)
            continue
        s = 0.0
        for F, m in zip(fr.force, masses):
            if not all(finite(c) for c in F):
                return None, None
            s += (F[0] ** 2 + F[1] ** 2 + F[2] ** 2) / (8.0 * UMASS * m)
        out.append(s * dt_au ** 2 * AU_ENERGY_EV)
    # bound when a time has no frame: 2 x max over frames
    fmax = 0.0
    for fr in trj.frames:
        if len(fr.force) != len(masses):
            continue
        s = sum((F[0] ** 2 + F[1] ** 2 + F[2] ** 2) / (8.0 * UMASS * m)
                for F, m in zip(fr.force, masses) if all(finite(c) for c in F))
        fmax = max(fmax, s)
    return out, 2.0 * fmax * dt_au ** 2 * AU_ENERGY_EV


def check_kinematic_identity(run, rep, tion, ework, t_fs, cid="kinematic_identity"):
    """Tion + E_work is constant up to the exact Verlet term (md.f90 part1/part2)."""
    if tion is None or ework is None:
        return rep.add(cid, LAYER, UNKNOWN, "Tion/E_work columns missing")
    if run.ensemble == "NVT" or (isinstance(run.step_velocity_scaling, int)
                                 and run.step_velocity_scaling >= 1) or run.yn_stop_system_momt == "y":
        return rep.add(cid, LAYER, NA, "thermostat, velocity scaling or momentum removal changes "
                       "velocities outside the force integration")
    if len(tion) < 3:
        return rep.add(cid, LAYER, NA, "fewer than 3 rows")
    if not all(finite(x) for x in tion + ework):
        return rep.add(cid, LAYER, FAIL, "non-finite Tion or E_work")
    dt_au = run.dt_fs / AU_TIME_FS if run.dt_fs else None
    natom = run.natom or 1
    trj = run.trj()
    terms, bound = _force_term(run, trj, t_fs, dt_au) if dt_au else (None, None)
    ref = 1  # row 0 may hold Tion=0 after a velocity restart (md%Tene not set)
    k_ref = tion[ref] + ework[ref]
    worst, worst_k, unc = 0.0, None, 0.0
    exact = 0
    for k in range(ref, len(tion)):
        dev = tion[k] + ework[k] - k_ref
        if terms is not None and terms[k] is not None and terms[ref] is not None:
            dev -= terms[k] - terms[ref]
            exact += 1
        elif bound is not None:
            unc = max(unc, bound)
        else:
            unc = float("inf")
        if abs(dev) > abs(worst):
            worst, worst_k = dev, k
    tolv = T["kinematic_identity_atol_per_atom"]["fail"] * natom + 1e-12 * max(
        max_abs(tion), max_abs(ework))
    metrics = {"max_abs_deviation_ev": abs(worst), "at_time_fs": t_fs[worst_k] if worst_k else None,
               "rows_with_exact_force_term": exact, "force_term_bound_ev": unc,
               "tolerance_ev": tolv}
    if not finite(unc):
        return rep.add(cid, LAYER, UNKNOWN, "no forces (trj.xyz) or masses (outputfile) to bound the "
                       "velocity-Verlet term", metrics, describe("kinematic_identity_atol_per_atom"))
    if abs(worst) > tolv + unc:
        return rep.add(cid, LAYER, FAIL, f"Tion+E_work changes by {fmt(abs(worst))} eV (> "
                       f"{fmt(tolv + unc)} eV): Tion/E_work columns inconsistent with the integrator",
                       metrics, describe("kinematic_identity_atol_per_atom"))
    return rep.add(cid, LAYER, PASS, f"Tion+E_work constant to {fmt(abs(worst))} eV (tolerance "
                   f"{fmt(tolv + unc)} eV)", metrics, describe("kinematic_identity_atol_per_atom"))


def check_tion_velocities(run, rep, tion, t_fs, temp=None, cid="tion_vs_velocities"):
    trj = run.trj()
    masses = run.element_masses_amu()
    if tion is None:
        return rep.add(cid, LAYER, UNKNOWN, "no Tion column")
    if trj is None or not trj.has_vel:
        return rep.add(cid, LAYER, UNKNOWN, "trj.xyz missing or without velocities")
    if masses is None:
        return rep.add(cid, LAYER, UNKNOWN, "masses not found ('Mass(ik) =' in outputfile and "
                       "atom kinds in inputfile)")
    natom = len(masses)
    rel = T["tion_from_velocities_rel"]["fail"]
    worst, n, worst_detail = 0.0, 0, None
    temp_bad = None
    for fr in trj.frames:
        if fr.step is None or fr.step < 0 or not run.dt_fs:
            continue
        k = _interp_index(t_fs, fr.step * run.dt_fs, 1e-6)
        if k is None:
            continue
        if len(fr.vel) != natom:
            return rep.add(cid, LAYER, FAIL, f"frame step {fr.step}: {len(fr.vel)} velocities for "
                           f"{natom} atoms")
        tv, floor = 0.0, 0.0
        for v, m in zip(fr.vel, masses):
            if not all(finite(c) for c in v):
                return rep.add(cid, LAYER, FAIL, f"non-finite velocity at step {fr.step}")
            mm = UMASS * m
            tv += 0.5 * mm * (v[0] ** 2 + v[1] ** 2 + v[2] ** 2)
            floor += mm * (abs(v[0]) + abs(v[1]) + abs(v[2])) * 5e-11 + 1.5 * mm * (5e-11) ** 2
        tv *= AU_ENERGY_EV
        floor *= AU_ENERGY_EV
        d = abs(tion[k] - tv)
        allowed = rel * abs(tv) + floor + 1e-14
        n += 1
        if not finite(tion[k]) or d > allowed:
            return rep.add(cid, LAYER, FAIL, f"step {fr.step}: Tion column {fmt(tion[k], 10)} eV vs "
                           f"sum m v^2/2 = {fmt(tv, 10)} eV (|d| {fmt(d)} > {fmt(allowed)})",
                           {"step": fr.step, "tion_column_ev": tion[k], "tion_from_v_ev": tv},
                           describe("tion_from_velocities_rel"))
        if d / max(allowed, 1e-300) > worst:
            worst, worst_detail = d / allowed, (fr.step, d, allowed)
        if temp is not None and finite(tion[k]):
            tk = 2.0 * tion[k] / AU_ENERGY_EV / (3.0 * natom * KB_AU)
            if not finite(temp[k]) or abs(temp[k] - tk) > 1e-9 * abs(tk) + 1e-9:
                temp_bad = (fr.step, temp[k], tk)
    if n == 0:
        return rep.add(cid, LAYER, UNKNOWN, "no trajectory frame coincides with an energy row")
    if temp_bad:
        return rep.add(cid, LAYER, FAIL, f"step {temp_bad[0]}: Temperature_ion {fmt(temp_bad[1])} K "
                       f"!= 2 Tion/(3 N kB) = {fmt(temp_bad[2])} K")
    return rep.add(cid, LAYER, PASS, f"Tion matches sum m v^2/2 at {n} frames (worst at "
                   f"{fmt(worst)} of the allowed difference)", {"frames": n,
                                                                "worst_fraction_of_allowed": worst},
                   describe("tion_from_velocities_rel"))


def _min_image(d, cell):
    if not cell:
        return d
    return [x - c * round(x / c) for x, c in zip(d, cell)]


def check_trajectory_bounds(run, rep, tion=None):
    cid = "trajectory_bounds"
    trj = run.trj()
    if trj is None or not trj.frames:
        return rep.add(cid, LAYER, UNKNOWN, "trj.xyz missing or empty")
    for fr in trj.frames:
        for vecs in (fr.pos, fr.vel, fr.force):
            for v in vecs:
                if v is None or not all(finite(c) for c in v):
                    return rep.add(cid, LAYER, FAIL, f"non-finite position/velocity/force at "
                                   f"step {fr.step}")
    cell = run.cell_aa
    first = trj.frames[0]
    maxdisp, maxf = 0.0, 0.0
    dmin0 = None
    dmin = float("inf")
    for idx, fr in enumerate(trj.frames):
        for p, p0 in zip(fr.pos, first.pos):
            maxdisp = max(maxdisp, math.sqrt(sum((a - b) ** 2 for a, b in zip(p, p0))))
        for F in fr.force:
            maxf = max(maxf, math.sqrt(sum(c * c for c in F)))
        if idx in (0, len(trj.frames) - 1) or idx % max(1, len(trj.frames) // 50) == 0:
            dm = float("inf")
            for i in range(len(fr.pos)):
                for j in range(i + 1, len(fr.pos)):
                    d = _min_image([a - b for a, b in zip(fr.pos[i], fr.pos[j])], cell)
                    dm = min(dm, math.sqrt(sum(c * c for c in d)))
            if idx == 0:
                dmin0 = dm
            dmin = min(dmin, dm)
    tmax = None
    if tion is not None and run.natom:
        tt = [2.0 * x / AU_ENERGY_EV / (3.0 * run.natom * KB_AU) for x in tion if finite(x)]
        tmax = max(tt) if tt else None
    metrics = {"max_displacement_aa": maxdisp, "max_force_au": maxf, "min_distance_aa": dmin,
               "first_frame_min_distance_aa": dmin0, "max_ion_temperature_k": tmax,
               "frames": len(trj.frames)}
    fails, warns = [], []
    td, tf, tt, tm = (T["max_displacement_angstrom"], T["max_force_au"], T["max_ion_temperature_k"],
                      T["min_distance_fraction"])
    if maxdisp > td["fail"]:
        fails.append(f"displacement {fmt(maxdisp)} A > {td['fail']}")
    elif maxdisp > td["warn"]:
        warns.append(f"displacement {fmt(maxdisp)} A > {td['warn']}")
    if trj.has_force:
        if maxf > tf["fail"]:
            fails.append(f"force {fmt(maxf)} Ha/bohr > {tf['fail']}")
        elif maxf > tf["warn"]:
            warns.append(f"force {fmt(maxf)} Ha/bohr > {tf['warn']}")
    if tmax is not None:
        if tmax > tt["fail"]:
            fails.append(f"ion temperature {fmt(tmax)} K > {tt['fail']}")
        elif tmax > tt["warn"]:
            warns.append(f"ion temperature {fmt(tmax)} K > {tt['warn']}")
    if dmin0 and finite(dmin) and dmin < tm["fail"] * dmin0:
        fails.append(f"min distance {fmt(dmin)} A < {tm['fail']} x initial {fmt(dmin0)} A")
    tol = {k: describe(k) for k in ("max_displacement_angstrom", "max_force_au",
                                    "max_ion_temperature_k", "min_distance_fraction")}
    if fails:
        return rep.add(cid, LAYER, FAIL, "; ".join(fails), metrics, tol)
    if warns:
        return rep.add(cid, LAYER, WARN, "; ".join(warns), metrics, tol)
    return rep.add(cid, LAYER, PASS, f"{len(trj.frames)} frames finite; max displacement "
                   f"{fmt(maxdisp)} A, max force {fmt(maxf)} Ha/bohr, min distance {fmt(dmin)} A",
                   metrics, tol)


def check_electron_number(run, rep):
    cid = "electron_number"
    so = run.stdout()
    if so is None or not so.rt_steps:
        return rep.add(cid, LAYER, UNKNOWN, "no RT step lines in outputfile")
    ne0 = run.nelec
    if not finite(ne0) or ne0 <= 0:
        return rep.add(cid, LAYER, UNKNOWN, "nelec not in inputfile")
    worst, at = 0.0, None
    for vals, ln in so.rt_steps:
        ne = vals[5] if len(vals) > 5 else None
        if not finite(ne):
            return rep.add(cid, LAYER, FAIL, f"non-finite electron count at outputfile line {ln}")
        d = abs(ne - ne0) / ne0
        if d > worst:
            worst, at = d, vals[0]
    tl = T["electron_number_rel"]
    metrics = {"max_rel_deviation": worst, "at_step": at, "rows": len(so.rt_steps),
               "print_resolution_rel": 1e-8 / ne0}
    if worst > tl["fail"]:
        return rep.add(cid, LAYER, FAIL, f"electron number off by {fmt(worst)} (relative) at step "
                       f"{at} > {tl['fail']}", metrics, describe("electron_number_rel"))
    st = WARN if worst > tl["warn"] else PASS
    return rep.add(cid, LAYER, st, f"max |Ne-nelec|/nelec = {fmt(worst)} over {len(so.rt_steps)} "
                   f"printed steps", metrics, describe("electron_number_rel"))


def md_report(run, rep=None):
    """Ehrenfest MD (or plain RT when the MD columns are absent)."""
    rep = rep or Report(run.dir, "ehrenfest")
    es = energy_series(run)
    check_conservation(run, rep)
    check_energy_balance(run, rep)
    check_electron_number(run, rep)
    if run.yn_md == "y":
        tion = es.get("Tion") if es else None
        ework = es.get("E_work") if es else None
        t = es["t_fs"] if es else []
        check_kinematic_identity(run, rep, tion, ework, t)
        check_tion_velocities(run, rep, tion, t, es.get("Temperature_ion") if es else None)
        check_trajectory_bounds(run, rep, tion)
    return rep


# --------------------------------------------------------------------------- dft_md
def check_dftmd_scf(run, rep):
    cid = "md_step_scf_converged"
    so = run.stdout()
    from .generic import threshold_au
    thr = threshold_au(run)
    if so is None or not so.dft_md_steps:
        return rep.add(cid, LAYER, FAIL if so is not None else UNKNOWN,
                       "no dft_md step lines (header '# 1.time[au], 2.Tene(ion)[au], ...') in outputfile")
    bad = []
    for vals, ln in so.dft_md_steps:
        res = vals[9]
        if not (finite(res, thr) and res < thr):
            bad.append((vals[0], res, ln))
    metrics = {"steps": len(so.dft_md_steps), "threshold": thr,
               "max_residual": max((v[9] for v, _ in so.dft_md_steps if finite(v[9])), default=None),
               "not_converged_messages": len(so.gs_not_converged)}
    if bad or so.gs_not_converged:
        first = bad[0] if bad else None
        msg = (f"{len(bad)} MD step(s) with final SCF residual >= threshold {fmt(thr)} or non-finite"
               + (f" (first at t={fmt(first[0])} au: {fmt(first[1])})" if first else "")
               + (f"; {len(so.gs_not_converged)} '#GS does not converged' message(s)"
                  if so.gs_not_converged else ""))
        return rep.add(cid, LAYER, FAIL, msg, metrics)
    return rep.add(cid, LAYER, PASS, f"all {len(so.dft_md_steps)} MD steps converged (max final "
                   f"residual {fmt(metrics['max_residual'])} < {fmt(thr)})", metrics)


def check_dftmd_columns(run, rep, t):
    """Internal consistency of <sys>_dft_md.data (write_dft_md_data)."""
    cid = "dft_md_columns"
    need = ["Tene", "Uene", "Uene-Uene0", "E_work", "E_nh", "E_tot", "E_tot-E_tot0"]
    if not all(t.has(c) for c in need):
        return rep.add(cid, LAYER, FAIL, f"missing columns; have {t.labels}")
    c = {k: t.col(k) for k in need}
    bad = []
    u0s, e0s = [], []
    for k in range(1, len(t.rows)):
        vals = [c[n][k] for n in need]
        if not all(finite(v) for v in vals):
            bad.append((k, "non-finite"))
            continue
        scale = max(abs(c["Uene"][k]), abs(c["E_tot"][k]), 1.0)
        if abs(c["Tene"][k] + c["Uene"][k] - c["E_tot"][k]) > 1e-12 * scale:
            bad.append((k, "E_tot != Tene + Uene"))
        u0s.append(c["Uene"][k] - c["Uene-Uene0"][k])
        e0s.append(c["E_tot"][k] - c["E_tot-E_tot0"][k])
    for name, ref in (("Uene0", u0s), ("E_tot0", e0s)):
        if ref and max(ref) - min(ref) > 1e-12 * max(1.0, max_abs(ref)) * 4:
            bad.append((None, f"{name} implied by the difference column is not constant "
                              f"(spread {fmt(max(ref) - min(ref))})"))
    if bad:
        return rep.add(cid, LAYER, FAIL, f"{len(bad)} inconsistency(ies): first {bad[0]}")
    return rep.add(cid, LAYER, PASS, "E_tot = Tene + Uene and the difference columns agree on "
                   f"{len(t.rows) - 1} rows")


def dftmd_report(run, rep=None):
    rep = rep or Report(run.dir, "dft_md")
    check_dftmd_scf(run, rep)
    t = run.dft_md()
    if t is None:
        rep.add("bo_energy_conservation", LAYER, UNKNOWN, "dft_md.data missing")
        return rep
    tf = run.time_factor(t.unit("Time"))
    efac = run.energy_factor(t.unit("E_tot")) if t.has("E_tot") else None
    if tf is None or efac is None:
        rep.add("bo_energy_conservation", LAYER, UNKNOWN, "unknown units in dft_md.data")
        return rep
    tt = [x * tf if finite(x) else float("nan") for x in t.col("Time")]
    E = [x * efac if finite(x) else float("nan") for x in t.col("E_tot")]
    check_dftmd_columns(run, rep, t)
    # row 0 is written before the first step; conservation is measured from row 1 on
    _conservation(rep, "bo_energy_conservation", tt[1:], E[1:], run.natom or 1, "E_tot=Tene+Uene")
    tion = [x * efac if finite(x) else float("nan") for x in t.col("Tene")]
    ework = [x * efac if finite(x) else float("nan") for x in t.col("E_work")]
    check_kinematic_identity(run, rep, tion, ework, tt)
    temp = t.col("Temperature_ion") if t.has("Temperature_ion") else None
    check_tion_velocities(run, rep, tion, tt, temp)
    check_trajectory_bounds(run, rep, tion)
    return rep
