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
"""Synthetic SALMON run directories for the unit tests and the fault-injection tool.

Every file is written in the format of the SALMON writer named next to it (src/io/write.f90,
src/rt/dip.f90, src/rt/main_tddft.f90, src/gs/scf_iteration_dft.f90, src/gs/main_dft_md.f90,
src/io/perflog.f90). The numbers come from small models, not from SALMON runs:

  * laser and electrons: an Acos2-shaped vector potential drives a damped oscillator (Lorentz
    model) for the electronic current. The electronic energy is the field work computed from
    the written current and vector potential, plus 0.15 of its quadrature ambiguity, so the
    energy balance holds as it does in real runs (where that ratio is about 0.15);
  * ions: a harmonic (Einstein) model of 8-atom silicon, driven by Z*E(t) under a laser and
    integrated in SALMON's velocity-Verlet order (src/rt/md.f90, part1/part2);
  * ground state, DC and LCFO: chosen energies and eigenvalues; format and logic only.

Every directory gets a FIXTURE.txt saying what it is.
"""
import math
import os
import random

from .run import AU_ENERGY_EV, AU_TIME_FS, AU_LENGTH_AA, UMASS, KB_AU

HBAR_EV_FS = 0.6582119569          # hbar [eV fs]
AU_EFIELD_V_PER_AA = 51.42206707   # atomic unit of field in V/Angstrom
E_EL0 = -858.226563761958          # electronic + ion-ion energy of the 8-atom cell [eV]

SI_RED = [(0, 0, 0), (.25, .25, .25), (.5, 0, .5), (0, .5, .5), (.5, .5, 0), (.75, .25, .75),
          (.25, .75, .75), (.75, .75, .25)]
A_SI = 5.43


# --------------------------------------------------------------------------- formatting
def fe(x, d=15, e=3):
    """Fortran Ew.dEe text without the field padding, e.g. -0.858226563761958E+003."""
    if x == 0:
        return "0." + "0" * d + "E+" + "0" * e
    s = f"{abs(x):.{d - 1}e}"
    mant, ex = s.split("e")
    digits = mant.replace(".", "")
    ex = int(ex) + 1
    return ("-" if x < 0 else "") + "0." + digits + ("E+" if ex >= 0 else "E-") + f"{abs(ex):0{e}d}"


def row(vals, breaks=()):
    """One table row as write.f90 writes it: F16.8, then 1X,E23.15E3 per value.

    `breaks` are the indices of values written by a following non-advancing WRITE (the
    current columns in <sys>_rt.data); SALMON's output has one extra blank there. The row
    ends with the blank of the closing list-directed WRITE.
    """
    out = [f"{vals[0]:16.8f}"]
    for k, v in enumerate(vals[1:], 1):
        out.append((" " if k in breaks else "") + " " + fe(v).rjust(23))
    return "".join(out) + " \n"


def _write(path, text):
    with open(path, "w") as fh:
        fh.write(text)


def _label(d, text):
    _write(os.path.join(d, "FIXTURE.txt"), text.strip() + "\n")


def banner(theory, real_orbitals=False, nproc=(1, 1, (1, 1, 1))):
    """Start of SALMON's stdout up to the pseudopotential summary (one Si species)."""
    k, ob, rg = nproc
    return "\n".join([
        "#" * 78,
        "# SALMON: Scalable Ab-initio Light-Matter simulator for Optics and Nanoscience",
        "#",
        "#                 synthetic fixture (mddc_verify), not a SALMON run",
        "#" * 78,
        "  Libxc: [disabled]",
        f"   theory= {theory}",
        f"   use of real value orbitals =   {'T' if real_orbitals else 'F'}",
        " r-space parallelization: " + ("on" if rg != (1, 1, 1) else "off"),
        " ======",
        " MPI distribution:",
        f"   nproc_k     : {k:12d}",
        f"   nproc_ob    : {ob:12d}",
        f"   nproc_rgrid : {rg[0]:12d}{rg[1]:13d}{rg[2]:13d}",
        " ===================pseudopotential data===================",
        " ik ,atom_symbol=            1 Si",
        " Zps(ik), Mlps(ik) =            4            2",
        " Mass(ik) =    28.00000000000000     ",
        " ==========================================================",
        " == Ewald ==",
        # Fortran drops the E of a three-digit exponent in an Ew.d field; this is not an overflow
        " cutoff length in G-space in ewald =  0.1870828864+102 [1/A]",
        " ===========",
    ]) + "\n"


def timer_table(sections, total):
    """'=== application breakdown [s] ===' block of src/io/perflog.f90 (readable mode)."""
    lines = ["=== application breakdown [s] ===", f"total calculation time,{total:.8f}"]
    for title, rows in sections:
        lines.append(f"{title:<30}" + "".join(f"{h:>12}" for h in
                                               ("min", "max", "average", "std. dev.", "difference",
                                                "relative")))
        for name, sec in rows:
            stats = (0.98 * sec, sec, 0.99 * sec, 0.01 * sec, 0.02 * sec, 1 / 0.98)
            lines.append(f"{name:<30}" + "".join(fe(x, 5, 2).rjust(12) for x in stats))
    lines += ["=== performance [GFLOPS] ===", "end SALMON"]
    return "\n".join(lines) + "\n"


GS_TIMERS = [("scf calculation", [("gs initialization", 0.12), ("scf iterations", 4.1),
                                  ("writing scf data", 0.01)]),
             ("in scf iterations", [("Minimization", 2.3), ("Gram Schmidt", 0.4),
                                    ("subspace-diag.", 0.6), ("calculating rho", 0.2)])]
RT_TIMERS = [("rt calculation", [("rt initialization", 0.13), ("rt iterations", 49.4)]),
             ("in rt iterations", [("time propagation", 36.9), ("calculating rho", 4.2),
                                   ("calculating curr", 5.8)])]


# --------------------------------------------------------------------------- laser model
def laser_response(nt, dt_fs, tw_fs=20.0, omega_ev=1.2, e0_v_per_aa=0.1227, kappa=-0.31):
    """Field and electronic current at steps 1..nt (A_eV_fs units), plus the energy history.

    A_x(t) = -(E0/w) cos^2(pi (t - tw/2)/tw) sin(w (t - tw/2)) for 0 < t < tw (zero outside),
    E_x = -dA_x/dt. The current J_x = dP/dt with P'' = -w0^2 P - g P' + kappa E (RK4, four
    substeps per time step); the sign of kappa makes the electrons absorb energy over the
    pulse, as in a real run. Returns dict with lists t, A, E, J (x components) and the
    electronic energy E_el at every step n = 0..nt:
        E_el(n) = E_EL0 + W_left(n) + 0.15 (W_right(n) - W_left(n)),
    where W_left/W_right are the left/right-point sums of Omega J.dA used by the verifier.
    """
    w = omega_ev / HBAR_EV_FS
    w0, g = 3.4 / HBAR_EV_FS, 0.1
    omega = A_SI ** 3

    def field(t):
        if not 0.0 < t < tw_fs:
            return 0.0, 0.0
        x = math.pi * (t - 0.5 * tw_fs) / tw_fs
        ph = w * (t - 0.5 * tw_fs)
        env = math.cos(x) ** 2
        a = -(e0_v_per_aa / w) * env * math.sin(ph)
        e = (e0_v_per_aa / w) * (-(math.pi / tw_fs) * math.sin(2 * x) * math.sin(ph) + env * w * math.cos(ph))
        return a, e

    def deriv(t, p, q):
        return q, -w0 * w0 * p - g * q + kappa * field(t)[1]

    p = q = 0.0
    t_list, a_list, e_list, j_list = [], [], [], []
    sub = 4
    h = dt_fs / sub
    for n in range(1, nt + 1):
        for s in range(sub):
            t0 = (n - 1) * dt_fs + s * h
            k1 = deriv(t0, p, q)
            k2 = deriv(t0 + h / 2, p + h / 2 * k1[0], q + h / 2 * k1[1])
            k3 = deriv(t0 + h / 2, p + h / 2 * k2[0], q + h / 2 * k2[1])
            k4 = deriv(t0 + h, p + h * k3[0], q + h * k3[1])
            p += h / 6 * (k1[0] + 2 * k2[0] + 2 * k3[0] + k4[0])
            q += h / 6 * (k1[1] + 2 * k2[1] + 2 * k3[1] + k4[1])
        a, e = field(n * dt_fs)
        t_list.append(n * dt_fs)
        a_list.append(a)
        e_list.append(e)
        j_list.append(q)
    e_el = [E_EL0]
    wl = wr = 0.0
    prev_a = prev_j = 0.0
    for a, j in zip(a_list, j_list):
        wl += omega * prev_j * (a - prev_a)
        wr += omega * j * (a - prev_a)
        e_el.append(E_EL0 + wl + 0.15 * (wr - wl))
        prev_a, prev_j = a, j
    return {"t": t_list, "A": a_list, "E": e_list, "J": j_list, "E_el": e_el}


def electron_count(t_fs, nelec=32.0, tw_fs=20.0):
    """Printed electron number: a slow loss of 3e-7 over the pulse (about the real size)."""
    return nelec - 3e-7 * min(1.0, t_fs / tw_fs) ** 2


RT_HEADER = ["# Real time calculation: ", "# Ac_ext: External vector potential field",
             "# E_ext: External electric field", "# Ac_tot: Total vector potential field",
             "# E_tot: Total electric field", "# Jm: Matter current density (electrons)"]


def rt_data_header(units, ions=False):
    """Header of <sys>_rt.data (write_rt_data_3d); units = (time, ac, field, current)."""
    ut, ua, ue, uj = units
    cols = [(1, "Time", ut)] + [(2 + i, f"Ac_ext_{c}", ua) for i, c in enumerate("xyz")] + \
        [(5 + i, f"E_ext_{c}", ue) for i, c in enumerate("xyz")] + \
        [(8 + i, f"Ac_tot_{c}", ua) for i, c in enumerate("xyz")] + \
        [(11 + i, f"E_tot_{c}", ue) for i, c in enumerate("xyz")]
    line = "#" + "".join(f" {k}:{n}[{u}]" for k, n, u in cols)
    line += " " + "".join(f" {14 + i}:Jm_{c}[{uj}]" for i, c in enumerate("xyz"))
    if ions:
        line += " " + "".join(f" {17 + i}:Jmi_{c}[{uj}]" for i, c in enumerate("xyz"))
    head = RT_HEADER + (["# Jmi: Matter current density (ions)"] if ions else [])
    return "\n".join(head) + "\n" + line + " \n"


UNITS_EV_FS = ("fs", "fs*V/Angstrom", "V/Angstrom", "1/fs*Angstrom^2")
UNITS_AU = ("a.u.", "a.u.", "a.u.", "a.u.")
J_AU_TO_EV_FS = (AU_LENGTH_AA / AU_TIME_FS) / AU_LENGTH_AA ** 3   # a.u. current -> 1/(fs A^2)

RT_INPUT = """&calculation
  theory = 'tddft_pulse'
/
&control
  sysname = 'Si'
/
&units
  unit_system = 'A_eV_fs'
/
&system
  yn_periodic = 'y'
  al = 5.43, 5.43, 5.43
  nstate = 32
  nelec = 32
  nelem = 1
  natom = 8
/
&pseudo
  izatom(1) = 14
  file_pseudo(1) = './Si_rps.dat'
  lmax_ps(1) = 2
  lloc_ps(1) = 2
/
&functional
  xc = 'PZ'
/
&rgrid
  num_rgrid = 16, 16, 16
/
&kgrid
  num_kgrid = 4, 4, 4
/
&tgrid
  nt = {nt}
  dt = {dt}
/
&emfield
  ae_shape1 = 'Acos2'
  I_wcm2_1 = {intensity}
  tw1 = 20.0
  omega1 = 1.2
  epdir_re1 = 1.0, 0.0, 0.0
/
"""

SI_COOR = "&atomic_red_coor\n" + "".join(f"  'Si' {a} {b} {c} 1\n" for a, b, c in SI_RED) + "/\n"


def rt_stdout(steps, theory="tddft_pulse"):
    """outputfile of an RT run; steps = [(step, time_fs, (jx, jy, jz) a.u., ne, etot_ev)]."""
    lines = [banner(theory).rstrip("\n"), " ",
             " " + "time-step".rjust(10) + "time[fs] ".rjust(11) + "Current(xyz)[a.u.]".rjust(48)
             + "electrons".rjust(15) + "Total energy[eV] ".rjust(18),
             "#" + "-" * 70]
    for n, t, j, ne, e in steps:   # src/rt/dip.f90: (i8,f14.8, 3e16.8, f15.8,f18.8)
        lines.append(f"{n:8d}{t:14.8f}" + "".join(fe(x, 8, 2).rjust(16) for x in j) + f"{ne:15.8f}{e:18.8f}")
    return "\n".join(lines) + "\n" + timer_table(RT_TIMERS, 50.1)


def rt(dst, nt=8000, dt_fs=0.0025, pulse=True):
    """Fixed-ion RT run of 8-atom silicon under an Acos2 pulse (or field-free), A_eV_fs units."""
    os.makedirs(dst, exist_ok=True)
    m = laser_response(nt, dt_fs) if pulse else None
    out = [rt_data_header(UNITS_EV_FS)]
    for n in range(1, nt + 1):
        if pulse:
            a, e, j = m["A"][n - 1], m["E"][n - 1], m["J"][n - 1]
        else:
            a = e = j = 0.0
        out.append(row([n * dt_fs, a, 0, 0, e, 0, 0, a, 0, 0, e, 0, 0, j, 0, 0], breaks=(13,)))
    _write(os.path.join(dst, "Si_rt.data"), "".join(out))
    e_el = m["E_el"] if pulse else [E_EL0] * (nt + 1)
    lines = ["# Real time calculation: \n", "# Eall: Total energy\n", "# Eall0: Initial total energy\n",
             "# 1:Time[fs] 2:Eall[eV] 3:Eall-Eall0[eV] \n"]
    steps = []
    for n in range(0, nt + 1, 10):
        lines.append(row([n * dt_fs, e_el[n], e_el[n] - e_el[0]]))
        if n:
            jx = (m["J"][n - 1] if pulse else 0.0) / J_AU_TO_EV_FS
            steps.append((n, n * dt_fs, (jx, 0.0, 0.0), electron_count(n * dt_fs), e_el[n]))
    _write(os.path.join(dst, "Si_rt_energy.data"), "".join(lines))
    _write(os.path.join(dst, "outputfile"), rt_stdout(steps))
    _write(os.path.join(dst, "inputfile"),
           RT_INPUT.format(nt=nt, dt=dt_fs, intensity="2e11" if pulse else "0d0") + SI_COOR)
    _write(os.path.join(dst, "salmon.exit"), "0\n")
    _label(dst, f"SYNTHETIC fixed-ion RT fixture ({'Acos2 pulse' if pulse else 'field-free'}), "
                f"8-atom Si, nt={nt}, dt={dt_fs} fs. Field: analytic Acos2 pulse; current: driven damped "
                "oscillator; energy: field work + 0.15 x its quadrature ambiguity. SALMON formats.")
    return dst


GS_INPUT = """&calculation
  theory = 'dft'
/
&control
  sysname = 'Si'
/
&units
  unit_system = 'a.u.'
/
&system
  yn_periodic = 'y'
  al = 10.26d0, 10.26d0, 10.26d0
  nstate = 32
  nelec = 32
  nelem = 1
  natom = 8
/
&pseudo
  izatom(1) = 14
  file_pseudo(1) = './Si_rps.dat'
  lmax_ps(1) = 2
  lloc_ps(1) = 2
/
&functional
  xc = 'PZ'
/
&rgrid
  num_rgrid = 12, 12, 12
/
&kgrid
  num_kgrid = 2, 2, 2
/
&scf
  ncg = 4
  nscf = 120
  threshold = 1.d-9
/
"""

GS_CONVERGED_LINE = "  #GS converged at    57  : 0.85514873E-09"


def gs_spectrum(nk=8, nstate=32, seed=2):
    """Eigenvalues [Ha] per k-point: 16 occupied bands below a 0.02 Ha gap, ascending."""
    rnd = random.Random(seed)
    out = []
    for k in range(nk):
        occ = sorted(-0.21 + 0.42 * (i / 15) ** 0.7 + rnd.uniform(-2e-3, 2e-3) for i in range(16))
        emp = sorted(0.25 + 0.3 * (i / 15) + rnd.uniform(-2e-3, 2e-3) for i in range(nstate - 16))
        out.append(occ + emp)
    return out


def gs(dst, e_tot=-858.2265707953129, n_iter=57):
    """Converged ground state of 8-atom silicon (2x2x2 k-points) in SALMON's GS formats."""
    os.makedirs(dst, exist_ok=True)
    lines = [banner("dft").rstrip("\n")]
    for n in range(1, n_iter + 1):   # scf_iteration_dft.f90 format 100/101
        lines.append(f" iter={n:6d}     Total Energy={e_tot + 400 * 0.7 ** n:19.8f}     "
                     f"Gap={0.6 - 0.5 * 0.8 ** n:15.8f}")
    lines.append(GS_CONVERGED_LINE)
    _write(os.path.join(dst, "outputfile"), "\n".join(lines) + "\n" + timer_table(GS_TIMERS, 4.3))
    spec = gs_spectrum()
    eg = ["#esp: single-particle energies (eigen energies)", "#occ: occupation numbers, io: orbital index",
          "# 1:io, 2:esp[a.u.], 3:occ"]
    for k, vals in enumerate(spec, 1):     # write_eigen: '(1x,i5,2(e26.16e3))'
        eg.append(f"k= {k:5d},  spin= {1:5d}")
        eg += [f" {i + 1:5d}" + fe(v, 16, 3).rjust(26) + fe(2.0 if i < 16 else 0.0, 16, 3).rjust(26)
               for i, v in enumerate(vals)]
    _write(os.path.join(dst, "Si_eigen.data"), "\n".join(eg) + "\n")
    info = [f" Total number of iteration = {n_iter:12d}", " ", f" Number of states = {32:12d}",
            f" Number of electrons = {32:12d}", " ", f" Total energy (eV) = {e_tot!r:>22}     ",
            " 1-particle energies (eV)"]
    vals = spec[0]
    for p in range(0, len(vals), 4):       # write_info_data: format(1x,4(i5,f15.4,2x))
        info.append(" " + "".join(f"{i + 1:5d}{vals[i] * AU_ENERGY_EV:15.4f}  " for i in range(p, min(p + 4, len(vals)))))
    _write(os.path.join(dst, "Si_info.data"), "\n".join(info) + "\n \n")
    _write(os.path.join(dst, "inputfile"), GS_INPUT + "&atomic_coor\n" +
           "".join(f"  'Si' {a * 10.26:.4f} {b * 10.26:.4f} {c * 10.26:.4f} 1\n" for a, b, c in SI_RED) + "/\n")
    _write(os.path.join(dst, "salmon.exit"), "0\n")
    _label(dst, f"SYNTHETIC ground-state fixture: 8-atom Si, 2x2x2 k-points, {n_iter} SCF iterations, "
                "chosen eigenvalues and energies (format and logic only). SALMON formats.")
    return dst


# --------------------------------------------------------------------------- molecular dynamics
def _ion_model(nt, dt_fs, efield_au, seed=7, t0_k=300.0, freq_thz=15.6, z=4.0):
    """Velocity Verlet in SALMON's order. efield_au(n) -> (Ex,Ey,Ez) at step n (n=0..nt)."""
    m = UMASS * 28.0
    k = m * (2 * math.pi * freq_thz * 1e12 * AU_TIME_FS * 1e-15) ** 2
    dt = dt_fs / AU_TIME_FS
    r0 = [[c * A_SI / AU_LENGTH_AA for c in p] for p in SI_RED]
    rnd = random.Random(seed)
    v = [[rnd.gauss(0, 1) for _ in range(3)] for _ in r0]
    vcom = [sum(x[c] for x in v) / len(v) for c in range(3)]
    v = [[x[c] - vcom[c] for c in range(3)] for x in v]
    t_now = sum(0.5 * m * sum(c * c for c in x) for x in v)
    t_target = 1.5 * len(v) * KB_AU * t0_k
    s = math.sqrt(t_target / t_now) if t0_k > 0 else 0.0
    v = [[c * s for c in x] for x in v]
    r = [list(p) for p in r0]

    def force(rr, n):
        e = efield_au(n)
        return [[-k * (rr[i][c] - r0[i][c]) + z * e[c] for c in range(3)] for i in range(len(rr))]

    def pot(rr):
        return sum(0.5 * k * (rr[i][c] - r0[i][c]) ** 2 for i in range(len(rr)) for c in range(3))

    f = force(r, 0)
    ework = 0.0
    out = [{"r": [x[:] for x in r], "v": [x[:] for x in v], "f": [x[:] for x in f], "U": pot(r),
            "T": sum(0.5 * m * sum(c * c for c in x) for x in v), "W": 0.0, "vhalf": [x[:] for x in v]}]
    for n in range(1, nt + 1):
        v = [[v[i][c] + f[i][c] / m * 0.5 * dt for c in range(3)] for i in range(len(v))]
        rl, fl = [x[:] for x in r], [x[:] for x in f]
        r = [[r[i][c] + v[i][c] * dt for c in range(3)] for i in range(len(r))]
        vhalf = [x[:] for x in v]
        f = force(r, n)
        for i in range(len(r)):
            for c in range(3):
                ework -= 0.5 * (fl[i][c] + f[i][c]) * (r[i][c] - rl[i][c])
        v = [[v[i][c] + f[i][c] / m * 0.5 * dt for c in range(3)] for i in range(len(v))]
        out.append({"r": [x[:] for x in r], "v": [x[:] for x in v], "f": [x[:] for x in f],
                    "U": pot(r), "T": sum(0.5 * m * sum(c * c for c in x) for x in v), "W": ework,
                    "vhalf": vhalf})
    return out, m, z


def _xyz_frame(comment, st):
    """One frame of <sys>_trj.xyz (write_xyz, 'rvf': a2,3f18.10,"  #v=",3f18.10,"  #f=",3f18.10)."""
    lines = [f"{len(st['r']):12d}", " " + comment]
    for i in range(len(st["r"])):
        p = [x * AU_LENGTH_AA for x in st["r"][i]]
        lines.append("Si" + "".join(f"{x:18.10f}" for x in p) + "  #v=" +
                     "".join(f"{x:18.10f}" for x in st["v"][i]) + "  #f=" +
                     "".join(f"{x:18.10f}" for x in st["f"][i]))
    return "\n".join(lines) + "\n"


def _temperature(st, natom=8):
    return 2 * st["T"] / (3 * natom * KB_AU)


MD_INPUT_EXTRA = """&md
  ensemble = 'NVE'
  yn_set_ini_velocity = 'y'
  temperature0_ion_k = 300.0d0
/
"""


def ehrenfest(dst, pulse=True, nt=8000, dt_fs=0.0025):
    """Ehrenfest MD of 8-atom silicon: the laser model above plus harmonic ions."""
    os.makedirs(dst, exist_ok=True)
    m = laser_response(nt, dt_fs) if pulse else None

    def efield(n):
        if not pulse or n == 0:
            return (0.0, 0.0, 0.0)
        return (m["E"][n - 1] / AU_EFIELD_V_PER_AA, 0.0, 0.0)

    states, mass, z = _ion_model(nt, dt_fs, efield)
    vol_au = A_SI ** 3 / AU_LENGTH_AA ** 3
    out = [rt_data_header(UNITS_EV_FS, ions=True)]
    for n in range(1, nt + 1):
        a, e, j = (m["A"][n - 1], m["E"][n - 1], m["J"][n - 1]) if pulse else (0.0, 0.0, 0.0)
        # calc_current_ion uses the half-step velocity (before md part2)
        ji = [sum(z * st[c] for st in states[n]["vhalf"]) / vol_au * J_AU_TO_EV_FS for c in range(3)]
        out.append(row([n * dt_fs, a, 0, 0, e, 0, 0, a, 0, 0, e, 0, 0, j, 0, 0] + ji, breaks=(13, 16)))
    _write(os.path.join(dst, "Si_rt.data"), "".join(out))
    e_el = m["E_el"] if pulse else [E_EL0] * (nt + 1)
    lines = ["# Real time calculation: \n", "# Eall: Total energy\n", "# Eall0: Initial total energy\n",
             "# Tion: Kinetic energy of ions\n", "# Temperature_ion: Temperature of ions\n",
             "# E_work: Work energy of ions(sum f*dr)\n",
             "# 1:Time[fs] 2:Eall[eV] 3:Eall-Eall0[eV]# 4:Tion[eV] 5:Temperature_ion[K] 6:E_work[eV] \n"]
    eall0 = e_el[0] + states[0]["U"] * AU_ENERGY_EV
    steps = []
    for n in range(0, nt + 1, 10):
        eall = e_el[n] + states[n]["U"] * AU_ENERGY_EV
        lines.append(row([n * dt_fs, eall, eall - eall0, states[n]["T"] * AU_ENERGY_EV,
                          _temperature(states[n]), states[n]["W"] * AU_ENERGY_EV]))
        if n:
            jx = (m["J"][n - 1] if pulse else 0.0) / J_AU_TO_EV_FS
            steps.append((n, n * dt_fs, (jx, 0.0, 0.0), electron_count(n * dt_fs), eall))
    _write(os.path.join(dst, "Si_rt_energy.data"), "".join(lines))
    # initialization_rt.f90 writes the t=0 frame, time_evolution_step.f90 one per out_rvf_rt_step
    frames = [_xyz_frame(f"#rt   step={-1:8d}   time{fe(0.0, 6, 2).rjust(16)}", states[0])]
    for n in range(10, nt + 1, 10):
        frames.append(_xyz_frame(f"#rt   step={n:8d}   time={fe(n * dt_fs / AU_TIME_FS, 6, 2).rjust(16)}"
                                 f"   T={_temperature(states[n]):12.4f}", states[n]))
    _write(os.path.join(dst, "Si_trj.xyz"), "".join(frames))
    inp = RT_INPUT.format(nt=nt, dt=dt_fs, intensity="2e11" if pulse else "0d0")
    inp = inp.replace("  theory = 'tddft_pulse'\n", "  theory = 'tddft_pulse'\n  yn_md = 'y'\n")
    _write(os.path.join(dst, "inputfile"), inp + MD_INPUT_EXTRA + SI_COOR)
    _write(os.path.join(dst, "outputfile"), rt_stdout(steps))
    _write(os.path.join(dst, "salmon.exit"), "0\n")
    _label(dst, f"SYNTHETIC Ehrenfest-MD fixture ({'Acos2 pulse' if pulse else 'field-free'}), nt={nt}, "
                f"dt={dt_fs} fs. Electrons: the laser model of fixtures.rt; ions: harmonic Einstein model "
                "(15.6 THz, Z=4, m=28 amu, 300 K start) in SALMON's velocity-Verlet order. SALMON formats.")
    return dst


def dft_md(dst, nt=200, dt_fs=0.2):
    """Born-Oppenheimer MD (theory='dft_md') of the same ion model; every step 'converged'."""
    os.makedirs(dst, exist_ok=True)
    states, mass, z = _ion_model(nt, dt_fs, lambda n: (0.0, 0.0, 0.0), seed=11)
    e_gs = -858.2265707953129
    hdr = ["# DFT-MD: adiabatic (ground state) molecular dynamics\n"]
    for a, b in (("Tene", "Kinetic energy of atoms(ions)"), ("Uene", "Potential energy"),
                 ("Uene0", "Initial potential energy"), ("E_work", "Work energy (sum F*dR)"),
                 ("E_nh", "Energy of NH thermostat"), ("E_tot", "Total energy(=Tene+Uene)"),
                 ("E_tot0", "Initial total energy"), ("Hnvt", "Hamiltonian with NH thermostat"),
                 ("Hnvt'", "Hnvt using E_work"), ("Temperature", "Temperature of atoms")):
        hdr.append(f"# {a}: {b}\n")
    hdr.append("# 1:Time[fs] 2:Tene[eV] 3:Uene[eV] 4:Uene-Uene0[eV] 5:E_work[eV] 6:E_nh[eV] "
               "7:E_tot[eV] 8:E_tot-E_tot0[eV] 9:Hnvt(2+3+6)[eV] 10:Hnvt(2+5+6)'[eV] 11:Temperature_ion[K]\n")
    lines = list(hdr)
    u0 = e_gs + states[0]["U"] * AU_ENERGY_EV
    et0 = u0 + states[0]["T"] * AU_ENERGY_EV
    so = ["# 1.time[au], 2.Tene(ion)[au], 3.Uene(gs)[au], 4.Ework[au], 5.Etot[au], 6.Etot-Etot0[au], "
          "7.Enh[au], 8.Htot[au], 9.Temperature_ion[K], 10.convergency[au]"]
    rnd = random.Random(3)
    for n in range(nt + 1):
        T_ = states[n]["T"] * AU_ENERGY_EV
        U_ = e_gs + states[n]["U"] * AU_ENERGY_EV
        W_ = states[n]["W"] * AU_ENERGY_EV
        temp = _temperature(states[n])
        lines.append(row([n * dt_fs, T_, U_, U_ - u0, W_, 0.0, T_ + U_, T_ + U_ - et0, U_, T_ + W_, temp]))
        if n >= 1:   # main_dft_md.f90: format(1x,f10.4, 7e20.10E3,f12.3,e15.8)
            res = 10 ** rnd.uniform(-9.5, -8.2)
            so.append(f" {n * dt_fs / AU_TIME_FS:10.4f}" + "".join(fe(x / AU_ENERGY_EV, 10, 3).rjust(20) for x in
                                                               (T_, U_, W_, T_ + U_, T_ + U_ - et0, 0.0, U_))
                      + f"{temp:12.3f}" + fe(res, 8, 2).rjust(15))
    _write(os.path.join(dst, "Si_dft_md.data"), "".join(lines))
    frames = []
    for n in range(1, nt + 1):    # main_dft_md.f90 format 110
        frames.append(_xyz_frame(f"#md-gs  step={n:8d}   time{fe(n * dt_fs / AU_TIME_FS, 6, 2).rjust(16)}",
                                 states[n]))
    _write(os.path.join(dst, "Si_trj.xyz"), "".join(frames))
    inp = GS_INPUT.replace("theory = 'dft'", "theory = 'dft_md'").replace(
        "unit_system = 'a.u.'", "unit_system = 'A_eV_fs'").replace(
        "al = 10.26d0, 10.26d0, 10.26d0", "al = 5.43, 5.43, 5.43").replace(
        "  threshold = 1.d-9\n", f"  threshold = 1.d-8\n/\n&tgrid\n  dt = {dt_fs}\n  nt = {nt}\n")
    inp += MD_INPUT_EXTRA.replace("/\n", "  step_update_ps = 1\n/\n") + \
        "&analysis\n  yn_out_rvf_rt = 'y'\n  out_rvf_rt_step = 1\n/\n" + SI_COOR
    _write(os.path.join(dst, "inputfile"), inp)
    _write(os.path.join(dst, "outputfile"), banner("dft_md") + "\n".join(so) + "\n" + timer_table(GS_TIMERS, 812.0))
    _write(os.path.join(dst, "salmon.exit"), "0\n")
    _label(dst, f"SYNTHETIC dft_md fixture: harmonic Einstein-model Si8 Born-Oppenheimer MD, nt={nt}, "
                f"dt={dt_fs} fs, every MD step 'converged'. SALMON formats.")
    return dst


# --------------------------------------------------------------------------- divide and conquer
def _dc_spectrum(n=66, shift=0.0, noise=0.0, seed=5):
    rnd = random.Random(seed)
    occ = [-0.20 + 0.24 * (i / 63) ** 0.8 for i in range(64)]
    emp = [0.075 + 0.17 * (i / max(1, n - 65)) for i in range(n - 64)]
    return [v + shift + rnd.uniform(-noise, noise) for v in occ + emp]


DC_INPUT = """&calculation
  theory='dft'
  yn_dc='y'
/
&dc
  num_fragment=4,1,1
  nstate_frag=64
  num_rgrid_buffer={buffer},0,0
/
&control
  sysname='Si'
/
&units
  unit_system='a.u.'
/
&system
  yn_periodic='y'
  al=41.04d0,10.26d0,10.26d0
  nstate=66
  nelec=128
  nelem=1
  natom={natom}
  temperature_k=300d0
/
&scf
  nscf=500
  threshold=1d-9
/
"""


def dc(dst, e_per_atom_offset=6e-4, natom=32, buffer=12, nscf_conv=True, conv_e=-3390.6646):
    """DC run (32-atom Si in 4 fragments, the geometry and sizes of develop's test 129)."""
    os.makedirs(os.path.join(dst, "data_dcdft", "total"), exist_ok=True)
    e_dc = conv_e + e_per_atom_offset * natom
    lines = [banner("dft").rstrip("\n")]
    n_it = 60 if nscf_conv else 500
    for n in range(1, n_it + 1):
        d = 1e-2 * 0.6 ** n if nscf_conv else max(1e-2 * 0.97 ** n, 3e-8)
        e = e_dc + 5.0 * 0.6 ** n
        # scf_iteration_dft.f90: '(a, i6 ,3x , a, e15.8, 3x, a, e15.8)'
        lines.append(f"DC #SCF = {n:6d}   Total Energy = {fe(e, 8, 2).rjust(15)}   "
                     f"diff = {fe(d, 8, 2).rjust(15)}")
    lines += [" start DC-LCFO", " basis functions operation: done", " Hamiltonian matrix: done",
              " lapack diag, #dim=        640", " diagonalization: done", " end DC-LCFO"]
    _write(os.path.join(dst, "outputfile"), "\n".join(lines) + "\n" + timer_table(GS_TIMERS, 96.0))
    for frag in range(1, 5):
        fd = os.path.join(dst, "data_dcdft", "fragments", f"{frag:06d}")
        os.makedirs(fd, exist_ok=True)
        # main_dft.f90 writes the global DC energy into every fragment directory
        _write(os.path.join(fd, "Si_info.data"),
               f" Total number of iteration = {n_it:12d}\n \n Number of states = {64:12d}\n"
               f" Number of electrons = {32:12d}\n \n Total energy (eV) = {e_dc + 5.0 * 0.6 ** n_it!r:>22}     \n")
    vals = _dc_spectrum(shift=2e-5)
    eg = ["#esp: single-particle energies (eigen energies) calculated by DC-LCFO method",
          "#io: orbital index", "# 1:io, 2:esp[a.u.]", "# spin=     1"]
    eg += [f" {i + 1:5d}{fe(v, 16, 3).rjust(26)}" for i, v in enumerate(vals)]
    _write(os.path.join(dst, "data_dcdft", "total", "Si_eigen.data"), "\n".join(eg) + "\n")
    _write(os.path.join(dst, "inputfile"), DC_INPUT.format(buffer=buffer, natom=natom))
    _write(os.path.join(dst, "salmon.exit"), "0\n")
    _label(dst, f"SYNTHETIC DC fixture (format and logic only, no physics): 32-atom Si geometry of "
                f"develop test 129, E_DC = E_conv + {e_per_atom_offset} eV/atom, buffer {buffer}.")
    return dst


CONV_INPUT = """&calculation
  theory='dft'
/
&control
  sysname='Si'
/
&units
  unit_system='a.u.'
/
&system
  yn_periodic='y'
  al=41.04d0,10.26d0,10.26d0
  nstate=66
  nelec=128
  nelem=1
  natom={natom}
/
&scf
  threshold=1d-9
/
"""


def conventional(dst, natom=32, e=-3390.6646):
    """Conventional GS of the DC fixture's structure, as the DC comparison's reference."""
    os.makedirs(dst, exist_ok=True)
    lines = [banner("dft").rstrip("\n")]
    for n in range(1, 41):
        lines.append(f" iter={n:6d}     Total Energy={e + 3.0 * 0.7 ** n:19.8f}     Gap={0.6:15.8f}")
    lines.append(f"  #GS converged at{41:6d}  :" + fe(5e-10, 8, 2).rjust(15))
    _write(os.path.join(dst, "outputfile"), "\n".join(lines) + "\n" + timer_table(GS_TIMERS, 210.0))
    vals = _dc_spectrum()
    eg = ["#esp: single-particle energies (eigen energies)", "#occ: occupation numbers, io: orbital index",
          "# 1:io, 2:esp[a.u.], 3:occ", f"k= {1:5d},  spin= {1:5d}"]
    eg += [f" {i + 1:5d}" + fe(v, 16, 3).rjust(26) + fe(2.0 if i < 64 else 0.0, 16, 3).rjust(26)
           for i, v in enumerate(vals)]
    _write(os.path.join(dst, "Si_eigen.data"), "\n".join(eg) + "\n")
    _write(os.path.join(dst, "Si_info.data"),
           f" Total number of iteration = {40:12d}\n \n Number of states = {66:12d}\n"
           f" Number of electrons = {128:12d}\n \n Total energy (eV) = {e!r:>22}     \n")
    _write(os.path.join(dst, "inputfile"), CONV_INPUT.format(natom=natom))
    _write(os.path.join(dst, "salmon.exit"), "0\n")
    _label(dst, "SYNTHETIC conventional GS fixture for the DC comparison (format only).")
    return dst


def rt_from_dc(dst, nt=800, ex_offset_per_atom=2e-4, noise=0.01):
    """The RT fixture, perturbed like an RT run started from DC-LCFO orbitals."""
    rt(dst, nt=nt)
    from .readers import read_table
    en = read_table(os.path.join(dst, "Si_rt_energy.data"))
    rtd = read_table(os.path.join(dst, "Si_rt.data"))
    with open(os.path.join(dst, "Si_rt_energy.data")) as fh:
        out = [l for l in fh if l.startswith("#")]
    for r_ in en.rows:
        dE = ex_offset_per_atom * 8 * (1 - math.exp(-r_[0] / 1.0))
        out.append(row([r_[0], r_[1] + dE, r_[2] + dE]))
    _write(os.path.join(dst, "Si_rt_energy.data"), "".join(out))
    with open(os.path.join(dst, "Si_rt.data")) as fh:
        out = [l for l in fh if l.startswith("#")]
    jmax = max(abs(r_[13]) for r_ in rtd.rows)
    for r_ in rtd.rows:
        rr = list(r_)
        rr[13] += noise * jmax * math.exp(-r_[0] / 0.5) * math.sin(40 * r_[0])
        out.append(row(rr, breaks=(13,)))
    _write(os.path.join(dst, "Si_rt.data"), "".join(out))
    _label(dst, f"SYNTHETIC 'RT from DC' fixture: the RT fixture with an excitation-energy offset rising to "
                f"{ex_offset_per_atom} eV/atom and early current noise ({noise} of max|J|).")
    return dst
