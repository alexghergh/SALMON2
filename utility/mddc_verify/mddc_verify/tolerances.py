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
"""Tolerances of the checks, each with its justification.

A limit may be tightened once more MD/DC outputs exist; loosening one needs a written reason
here (never change a limit silently).

Sources used in the justifications:
  [T124] testsuites/124_periodic_H2O_md_dp/verification: Eall at step 1000, |d| <= 1e-6 eV.
  [T126] testsuites/126_periodic_H2O_dft_md_dp/verification: E_tot at step 20, |d| <= 1e-3 eV.
  [T129] develop testsuites/129_Si_dcdft_lcfo_chefsi/verification: DC total energy |d| <= 2e-2 eV
         (32 atoms = 6.3e-4 eV/atom), 12 LCFO eigenvalues |d| <= 1e-4 Ha, SCF diff <= 1e-9.
  [T112] testsuites/112_bulk_Si_rt_response_dp/verification: Jm_z |d| <= 1e-8 a.u. (3.4e-5 rel).
  [P]    Yamada & Otobe, arXiv:2606.21125v1, Figs. 3-4 (aSi-512, buffers b=c/2,3c/4,c = 10/15/20
         grid points): at I=1e9 W/cm^2 the DC excitation energy deviates from conventional by
         about 8e-4 (b=c/2), 3e-4 (3c/4) and 2e-4 (c) eV/atom at 20 fs (read off Fig. 3b, by
         eye); at I=1e12 W/cm^2 "the current and excitation energy agree almost perfectly"
         (Fig. 4, final excitation energy about 1.08 eV/atom). No ground-state energy
         difference DC vs conventional is reported.
  [OBS]  Runs made while developing this tool (SALMON 2.3.0, CPU and GPU builds): 8-atom silicon
         tddft_pulse (16^3 k-points, Acos2 2e11 W/cm^2, fixed ions) at dt = 2.5 as and 1.25 as;
         60-step silicon RT runs on two CPU MPI layouts and their GPU counterparts; and CPU/GPU
         pairs of testsuite 112 on three GPU types.
"""

TOLERANCES = {
    # ------------------------------------------------------------------ Ehrenfest MD / dft_md
    "energy_conservation_max_per_atom": {
        "warn": 1e-4, "fail": 1e-3, "unit": "eV/atom",
        "applies": "max |H(t)-H(t0)|/N in a field-free window; H = Eall+Tion (Ehrenfest) or "
                   "E_tot = Uene+Tene (dft_md)",
        "why": "1 meV/atom is 2.6% of the ion kinetic energy per atom at 300 K (3/2 kT = 38.8 "
               "meV) and 1/26 of kT; a larger error visibly distorts ion dynamics. The warn "
               "level is 10x tighter. [T124] bounds Eall reproducibility to 1e-6 eV (3 atoms), "
               "far tighter, but it is a regression value, not a conservation bound. To be "
               "calibrated on the first real 124 and silicon-MD outputs.",
    },
    "energy_drift_per_atom_per_ps": {
        "warn": 1.0, "fail": 10.0, "unit": "meV/atom/ps",
        "applies": "least-squares slope of H(t)-H(t0) per atom; reported only for windows >= 10 fs",
        "why": "Common AIMD practice: < 1 meV/atom/ps is good, ~10 meV/atom/ps marginal. Over a "
               "20 fs run the max-deviation check above is the binding one.",
    },
    "energy_balance_ratio": {
        "warn": 0.5, "fail": 1.0, "unit": "ratio",
        "applies": "max_t |dH(t) - W(t)| / max_t Q(t), where W is the field work from <sys>_rt.data "
                   "(left-point rule, W = Omega*sum J(t_{n-1}).dA_n, electrons and ions) and Q = "
                   "|W_right - W_left| is the quadrature ambiguity of W from the sampled outputs",
        "why": "Validated on fixed-ion silicon runs [OBS]: residual/Q = 0.154 at dt = 2.5 as "
               "and at dt = 1.25 as; the residual halves with dt (0.97 -> 0.49 meV), so it is "
               "discretisation, not physics. An energy error larger than the work-integral "
               "ambiguity is not explained by sampling. Sensitivity is limited: Q is about 2% of "
               "the absorbed energy (6 meV of 0.32 eV for Si at 2e11 W/cm^2).",
    },
    "kinematic_identity_atol_per_atom": {
        "fail": 1e-9, "unit": "eV/atom",
        "applies": "Tion(t)+E_work(t) - [Tion(t1)+E_work(t1)] (NVE, no velocity scaling); plus the "
                   "exact velocity-Verlet term sum_i (|F_i(t)|^2-|F_i(t1)|^2) dt^2/(8 m_i) when "
                   "forces are available every step, else that term bounded from the frames",
        "why": "Algebraic identity of SALMON's integrator (md.f90 part1/part2): E_work = -sum "
               "1/2(F_old+F_new).dR. It holds to rounding (E23.15E3 = 15 digits), so any larger "
               "mismatch is a corrupted Tion/E_work column or bookkeeping fault.",
    },
    "tion_from_velocities_rel": {
        "fail": 1e-6, "unit": "relative",
        "applies": "Tion column vs sum 1/2 m v^2 from <sys>_trj.xyz (m = 1822.9 * Mass(ik) from "
                   "outputfile); absolute floor from F18.10 rounding of v (5e-11 a.u.)",
        "why": "Same quantity written twice; differences only from output rounding.",
    },
    "electron_number_rel": {
        "warn": 1e-7, "fail": 1e-6, "unit": "relative",
        "applies": "max |Ne(t)-nelec|/nelec from the RT step lines in outputfile (F15.8)",
        "why": "[OBS] worst real value 9.1e-9 (silicon, dt=2.5 as, 8000 steps; 3.1e-10 at dt=1.25 "
               "as, i.e. the F15.8 print resolution). Warn at 11x, fail at 110x the observed worst "
               "case. SALMON itself stops only when Ne is off by 10x or NaN (dip.f90, plain STOP).",
    },
    "max_displacement_angstrom": {
        "warn": 0.5, "fail": 1.0, "unit": "Angstrom",
        "applies": "max |R_i(t)-R_i(t_first_frame)| over atoms (periodic images not unwrapped; "
                   "SALMON does not wrap positions)",
        "why": "Coherent-phonon and 300 K runs of tens of fs move Si atoms by < 0.1 A; 1 A is 43% "
               "of the Si-Si bond (2.35 A) and means bond breaking or a blow-up.",
    },
    "max_force_au": {
        "warn": 0.1, "fail": 0.5, "unit": "Hartree/bohr",
        "applies": "max |F_i| over frames",
        "why": "0.5 Ha/bohr = 25.7 eV/A; thermal Si forces are ~0.01-0.05 Ha/bohr. Larger forces "
               "mean overlapping atoms or a corrupted force.",
    },
    "max_ion_temperature_k": {
        "warn": 2000.0, "fail": 5000.0, "unit": "K",
        "applies": "kinetic temperature from Tion",
        "why": "Si melts at 1687 K; 5000 K within tens of fs means a blow-up.",
    },
    "min_distance_fraction": {
        "fail": 0.5, "unit": "fraction of the first frame's minimum interatomic distance",
        "applies": "minimum image distance in any frame",
        "why": "Atoms closer than half the starting nearest-neighbour distance overlap their "
               "pseudopotential cores.",
    },
    # ------------------------------------------------------------------ DC
    "dc_vs_conventional_energy_per_atom": {
        "warn": 1e-3, "fail": 1e-2, "unit": "eV/atom",
        "applies": "|E_DC - E_conv|/N, same structure; DC decks use temperature_k=300, the "
                   "conventional deck none (smearing at 300 K is negligible for a gapped system)",
        "why": "[P] reports no ground-state energy error. Its RT excitation-energy deviations "
               "from DC densities are 2e-4 to 8e-4 eV/atom [P]; [T129] reproduces the DC energy "
               "itself to 6.3e-4 eV/atom. Warn at 1 meV/atom (above all of these), fail at 10 "
               "meV/atom (DC no longer a usable stand-in for the conventional ground state).",
    },
    "dc_homo_lumo_ev": {
        "warn": 0.05, "fail": 0.2, "unit": "eV",
        "applies": "LCFO HOMO, LUMO and mid-gap (Fermi-level proxy) vs conventional eigenvalues",
        "why": "No published reference [P]. [T129] reproduces LCFO eigenvalues to 1e-4 Ha "
               "(2.7 meV). 0.2 eV is a quarter of the aSi LDA gap scale; provisional.",
    },
    "dc_band_energy_per_atom": {
        "warn": 5e-3, "fail": 2e-2, "unit": "eV/atom",
        "applies": "band energy 2*sum_occ eps / N, LCFO vs conventional",
        "why": "Provisional; same reasoning as the total energy, band energy is less "
               "variational so the limits are 2-5x wider.",
    },
    "dc_lcfo_basis_min_ratio": {
        "fail": 1.0, "unit": "LCFO basis dimension / nstate",
        "applies": "'lapack diag, #dim=' or 'eigenexa diag, #dim=' in outputfile",
        "why": "The LCFO basis must span at least the requested global states.",
    },
    "rt_dc_vs_conv_excitation_per_atom": {
        "warn_frac": 0.5, "fail_abs": 1e-3, "fail_rel": 0.02, "unit": "eV/atom",
        "applies": "max_t |Eex_DC(t)-Eex_conv(t)|/N <= max(fail_abs, fail_rel*max|Eex_conv|/N)",
        "why": "[P] Fig. 3 (1e9 W/cm^2): published buffers deviate by <= 8e-4 eV/atom, accepted "
               "by the authors as practical; Fig. 4 (1e12): indistinguishable, i.e. <~1% of "
               "1.08 eV/atom at plot resolution. Every published curve passes these limits.",
    },
    "rt_dc_vs_conv_current_nrms": {
        "warn": 0.03, "fail": 0.10, "unit": "rms(J_DC-J_conv)/rms(J_conv)",
        "applies": "current along the field direction over the compared window",
        "why": "[P] Figs. 3a/4a: 'overall agreement', noise near t=0 for DC. Provisional.",
    },
    "dc_buffer_monotonic": {
        "unit": "boolean (warn only)",
        "applies": "|E_DC(b)-E_conv| should not grow with buffer size b",
        "why": "[P]: 'increasing the buffer thickness b systematically improves the accuracy'.",
    },
    # ------------------------------------------------------------------ cross-run
    "cross_run_exact": {
        "unit": "token equality",
        "applies": "same build route and MPI/GPU layout: every numeric token of every compared "
                   "file identical, same row counts",
        "why": "A change that keeps the order of operations (e.g. a GPU port that follows the CPU "
               "order) must reproduce the saved outputs exactly in the same configuration; the "
               "first difference is reported with its time.",
    },
    "cross_run_energy_atol_per_atom": {
        "fail": 1e-6, "unit": "eV/atom",
        "applies": "Eall, Eall-Eall0, Tion, E_work, Uene, E_tot, DC energy, between independent "
                   "routes (CPU/GPU, 2.3.0/develop, original/modified)",
        "why": "[T124] 1e-6 eV after 1000 MD steps (3 atoms). [OBS] largest independent-route "
               "difference 2.2e-10 eV/atom (two CPU MPI layouts, 60 steps); CPU vs GPU "
               "<= 3.4e-13 eV/atom. Per-atom so it scales to aSi-512; never tighter than the "
               "printed resolution (the 'DC #SCF' energy is e15.8).",
    },
    "cross_run_current_rtol": {
        "fail": 1e-5, "unit": "relative to max|J| of the reference",
        "applies": "Jm, Jmi (and other non-field) columns; scale = max over the x/y/z components "
                   "so a component that is zero by symmetry is judged against the vector",
        "why": "[T112] 1e-8 a.u. on |Jm_z| = 3e-4 (3.4e-5 relative). [OBS] relative to the vector "
               "magnitude max|J|: CPU vs GPU <= 4.3e-14, CPU layouts 2.9e-8 (60-180 steps); 1e-5 "
               "leaves ~350x for growth over 10^4-step runs.",
    },
    "cross_run_field_rtol": {
        "fail": 1e-10, "unit": "relative to max|E|",
        "applies": "Ac_ext, E_ext (input-determined) and Ac_tot/E_tot in the transverse geometry",
        "why": "The external field is an analytic function of time; only rounding may differ.",
    },
    "cross_run_position_atol": {
        "fail": 1e-6, "unit": "Angstrom",
        "applies": "trajectory positions",
        "why": "1e-6 A is 1e-4 of a typical phonon amplitude; F18.10 resolves 1e-10 A.",
    },
    "cross_run_velocity_force_rtol": {
        "fail": 1e-4, "unit": "relative to max|v| or max|F| of the reference",
        "applies": "trajectory velocities and forces",
        "why": "Force reductions differ in order between CPU and GPU; 1e-4 relative force noise "
               "changes positions by << 1e-6 A over tens of fs. To be calibrated.",
    },
    "cross_run_eigen_atol_ha": {
        "fail": 1e-4, "unit": "Hartree",
        "applies": "GS / LCFO eigenvalues between routes",
        "why": "[T129] eigenvalue tolerance.",
    },
}


def tol(name, key="fail"):
    return TOLERANCES[name][key]


def describe(name):
    t = dict(TOLERANCES[name])
    t["name"] = name
    return t
