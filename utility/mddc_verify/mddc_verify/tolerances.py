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
}


def tol(name, key="fail"):
    return TOLERANCES[name][key]


def describe(name):
    t = dict(TOLERANCES[name])
    t["name"] = name
    return t
