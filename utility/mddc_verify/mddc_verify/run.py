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
"""A SALMON run directory: its input parameters and lazily read outputs."""
import os

from .namelist import Inputs
from . import readers

AU_TIME_FS = 0.02418884326505      # src/io/inputoutput.f90
AU_ENERGY_EV = 27.21138505
AU_LENGTH_AA = 0.52917721067
UMASS = 1822.9                     # src/common/const.f90 (atomic mass unit in m_e)
KB_AU = 1.38064852e-23 / 4.359744650e-18   # kB/hartree2J as in md.f90 [Ha/K]

MODES = ("ehrenfest", "dft_md", "dc", "gs", "rt")


class Run:
    def __init__(self, rundir, sysname=None):
        self.dir = os.path.abspath(rundir)
        self.input_path = os.path.join(self.dir, "inputfile")
        self.inputs = Inputs.from_file(self.input_path) if os.path.exists(self.input_path) else None
        g = (self.inputs or Inputs({})).get
        self.sysname = sysname or (self.inputs.get("sysname", "default") if self.inputs else "default")
        self.theory = str(g("theory", "")).lower()
        self.unit_system = str(g("unit_system", "au")).lower()
        self.au = self.unit_system in ("au", "a.u.")
        self.yn_md = str(g("yn_md", "n")).lower()[:1]
        self.yn_dc = str(g("yn_dc", "n")).lower()[:1]
        if self.theory == "dft_md":
            self.yn_md = "y"
        self.natom = g("natom", None)
        self.nelec = g("nelec", None)
        self.nstate = g("nstate", None)
        self.nt = g("nt", None)
        dt = g("dt", None)
        self.dt_fs = (dt * AU_TIME_FS if self.au else dt) if isinstance(dt, (int, float)) else None
        self.out_rt_energy_step = g("out_rt_energy_step", 10)
        self.out_rvf_rt_step = g("out_rvf_rt_step", 10)
        self.yn_out_rvf_rt = str(g("yn_out_rvf_rt", "n")).lower()[:1]
        self.ensemble = str(g("ensemble", "NVE")).upper()
        self.thermostat = str(g("thermostat", "nothing")).lower()
        self.step_velocity_scaling = g("step_velocity_scaling", -1)
        self.yn_stop_system_momt = str(g("yn_stop_system_momt", "n")).lower()[:1]
        self.nscf = g("nscf", 300)
        self.threshold = g("threshold", None)
        self.convergence = str(g("convergence", "rho_dne")).lower()
        self.temperature_k = g("temperature_k", None)
        self.temperature = g("temperature", None)
        self.intensity = g("i_wcm2_1", None)
        self.e_amplitude = g("e_amplitude1", None)
        self.ae_shape1 = str(g("ae_shape1", "none")).lower()
        self.yn_conventional_from_dcdft = str(g("yn_conventional_from_dcdft", "n")).lower()[:1]
        al = g("al", None, index=None)
        self.cell_aa = None
        if isinstance(al, list) and len(al) == 3 and all(isinstance(x, (int, float)) for x in al):
            f = AU_LENGTH_AA if self.au else 1.0
            self.cell_aa = [x * f for x in al]
        self._cache = {}

    # ------------------------------------------------------------------ helpers
    def mode(self):
        if self.theory == "dft_md":
            return "dft_md"
        if self.yn_dc == "y" and self.theory == "dft":
            return "dc"
        if self.theory.startswith("tddft") or self.theory.endswith("maxwell_tddft"):
            return "ehrenfest" if self.yn_md == "y" else "rt"
        if self.theory == "dft":
            return "gs"
        return None

    def volume_aa3(self):
        if self.cell_aa:
            return self.cell_aa[0] * self.cell_aa[1] * self.cell_aa[2]
        return None

    def path(self, suffix, sub=None):
        base = os.path.join(self.dir, sub) if sub else self.dir
        return os.path.join(base, f"{self.sysname}_{suffix}")

    def exists(self, p):
        return os.path.isfile(p) and os.path.getsize(p) > 0

    def _get(self, key, fn, p):
        if key not in self._cache:
            self._cache[key] = fn(p) if self.exists(p) else None
        return self._cache[key]

    def stdout(self):
        return self._get("stdout", readers.read_stdout, os.path.join(self.dir, "outputfile"))

    def rt(self):
        return self._get("rt", readers.read_table, self.path("rt.data"))

    def rt_energy(self):
        return self._get("rt_energy", readers.read_table, self.path("rt_energy.data"))

    def dft_md(self):
        return self._get("dft_md", readers.read_table, self.path("dft_md.data"))

    def trj(self):
        return self._get("trj", readers.read_xyz, self.path("trj.xyz"))

    def info(self):
        return self._get("info", readers.read_info, self.path("info.data"))

    def eigen(self):
        return self._get("eigen", readers.read_eigen, self.path("eigen.data"))

    def dc_eigen_path(self):
        """Real LCFO: <sys>_eigen.data; complex LCFO (develop): <sys>_lcfo_complex_eigen.data."""
        sub = os.path.join("data_dcdft", "total")
        p = self.path("eigen.data", sub=sub)
        q = self.path("lcfo_complex_eigen.data", sub=sub)
        return q if (not os.path.exists(p) and os.path.exists(q)) else p

    def dc_eigen(self):
        return self._get("dc_eigen", readers.read_eigen, self.dc_eigen_path())

    def energy_factor(self, unit):
        """Multiply a value in the file's energy unit to get eV."""
        if unit is None:
            return None
        u = unit.strip().lower()
        if u == "ev":
            return 1.0
        if u in ("a.u.", "au", "hartree"):
            return AU_ENERGY_EV
        return None

    def time_factor(self, unit):
        u = (unit or "").strip().lower()
        if u == "fs":
            return 1.0
        if u in ("a.u.", "au"):
            return AU_TIME_FS
        return None

    def element_masses_amu(self):
        """Mass per atom (amu as SALMON uses it), from outputfile 'Mass(ik) =' and the kinds."""
        so = self.stdout()
        if not so or not so.masses:
            return None
        kinds = self.atom_kinds()
        if kinds is None:
            return [so.masses[0]] * self.natom if len(so.masses) == 1 and self.natom else None
        try:
            return [so.masses[k - 1] for k in kinds]
        except IndexError:
            return None

    def atom_kinds(self):
        """Kind index (1-based) for each atom from &atomic_(red_)coor or file_atom_coor."""
        lines = []
        if self.inputs:
            _, lines = self.inputs.coordinate_lines()
            fname = self.inputs.get("file_atom_coor", None)
            if not lines and fname:
                p = os.path.join(self.dir, str(fname))
                if os.path.exists(p):
                    with open(p, errors="replace") as fh:
                        lines = [l.strip() for l in fh if l.strip() and not l.strip().startswith("!")]
        kinds = []
        for l in lines:
            toks = l.replace(",", " ").split()
            if len(toks) >= 5:
                try:
                    kinds.append(int(toks[4]))
                except ValueError:
                    return None
        if self.natom and len(kinds) == self.natom:
            return kinds
        return None
