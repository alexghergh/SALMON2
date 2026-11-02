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
"""Physics checks for Ehrenfest MD, Born-Oppenheimer dft_md and plain RT (layer 2)."""
import io
import os
import shutil
import sys
import tempfile
import unittest
from contextlib import redirect_stdout

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from tests.common import TempDirCase, read_lines, replace_field, write  # noqa: E402
from mddc_verify import fixtures, physics_md  # noqa: E402
from mddc_verify.cli import check, main  # noqa: E402
from mddc_verify.run import Run, AU_TIME_FS  # noqa: E402
from mddc_verify.tolerances import TOLERANCES, describe  # noqa: E402
from mddc_verify.verdict import Report  # noqa: E402


def add_drift(path, cols, rate_per_fs, natom=8):
    """Add rate*t*natom [eV] to the given 1-based columns of every data row."""
    lines = read_lines(path)
    for i, l in enumerate(lines):
        if not l.strip() or l.lstrip().startswith("#"):
            continue
        toks = l.split()
        t = float(toks[0])
        for c in cols:
            toks[c - 1] = fixtures.fe(float(toks[c - 1]) + rate_per_fs * t * natom)
        lines[i] = fixtures.row([float(x) for x in toks]).rstrip("\n")
    write(path, "\n".join(lines))


class Tion0Placeholder(unittest.TestCase):
    """SALMON writes Tion = 0 on the t=0 row of <sys>_rt_energy.data when ions start with
    velocities (write_rt_energy_data runs before md%Tene is first set). The numbers are the
    first rows of a real Ehrenfest-MD run of 8-atom silicon (SALMON 2.3.0)."""
    ES = {"t_fs": [0.0, 0.025, 0.05, 0.075],
          "Eall": [-857.863204819865, -857.865524556065, -857.864182739476, -857.862858560268],
          "Tion": [0.0, 0.310041101855481, 0.310019447045488, 0.310023869768566]}

    def test_detected(self):
        self.assertTrue(physics_md.tion_row0_unwritten(self.ES))
        self.assertFalse(physics_md.tion_row0_unwritten({"Tion": [0.31, 0.31, 0.31]}))
        self.assertFalse(physics_md.tion_row0_unwritten({"Tion": [0.0, 0.0, 0.0]}))
        self.assertFalse(physics_md.tion_row0_unwritten({"Eall": [1.0]}))

    def test_jump_not_counted_as_drift(self):
        H = physics_md.total_energy(None, self.ES)
        jump = (H[1] - H[0]) / 8
        self.assertGreater(jump, 0.03)  # the spurious 0.039 eV/atom a naive check reports
        rest = max(abs(h - H[1]) for h in H[1:]) / 8
        self.assertLess(rest, 1e-3)     # what is left once row 0 is skipped


class MDRuns(TempDirCase):
    @classmethod
    def setUpClass(cls):
        cls.base = tempfile.mkdtemp()
        cls.nve = fixtures.ehrenfest(os.path.join(cls.base, "nve"), pulse=False, nt=400)
        cls.laser = fixtures.ehrenfest(os.path.join(cls.base, "laser"), pulse=True, nt=800)
        cls.rt = fixtures.rt(os.path.join(cls.base, "rt"), nt=800)
        cls.bo = fixtures.dft_md(os.path.join(cls.base, "bo"), nt=60)

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.base)

    def test_clean_runs_pass(self):
        for d in (self.nve, self.laser, self.rt, self.bo):
            _, rep = check(d)
            self.assertIn(rep.verdict(), ("pass", "warn"), "\n".join(rep.summary_lines()))
        self.assertEqual(Run(self.nve).mode(), "ehrenfest")
        self.assertEqual(Run(self.bo).mode(), "dft_md")

    def test_laser_energy_balance(self):
        _, rep = check(self.rt)
        c = next(c for c in rep.checks if c["id"] == "energy_balance")
        self.assertEqual(c["status"], "pass")
        self.assertAlmostEqual(c["metrics"]["ratio"], 0.15, delta=0.02)  # by construction
        _, rep = check(self.laser)
        c = next(c for c in rep.checks if c["id"] == "energy_balance")
        self.assertIn(c["status"], ("pass", "warn"))
        self.assertTrue(c["metrics"]["includes_ion_work"])

    def test_energy_drift_fails(self):
        d = self.copy(self.nve, "drift")
        add_drift(os.path.join(d, "Si_rt_energy.data"), [2, 3], 2e-3)  # 2 meV/atom/fs
        self.assertEqual(check(d)[1].status_of("energy_conservation"), "fail")

    def test_energy_error_under_laser_fails(self):
        d = self.copy(self.rt, "balance")
        add_drift(os.path.join(d, "Si_rt_energy.data"), [2, 3], 5e-3)
        self.assertEqual(check(d)[1].status_of("energy_balance"), "fail")

    def test_wrong_tion_fails(self):
        d = self.copy(self.laser, "tion")
        replace_field(os.path.join(d, "Si_rt_energy.data"), 40, 4, lambda t: fixtures.fe(float(t) * 1.01))
        _, rep = check(d)
        self.assertEqual(rep.status_of("tion_vs_velocities"), "fail")
        self.assertEqual(rep.status_of("kinematic_identity"), "fail")

    def test_position_jump_fails(self):
        d = self.copy(self.nve, "jump")
        p = os.path.join(d, "Si_trj.xyz")
        lines = read_lines(p)
        i = 10 * 20 + 2          # frame 20 (10 lines per frame), first atom
        x = float(lines[i][2:20]) + 1.2
        lines[i] = lines[i][:2] + f"{x:18.10f}" + lines[i][20:]
        write(p, "\n".join(lines))
        self.assertEqual(check(d)[1].status_of("trajectory_bounds"), "fail")

    def test_electron_number(self):
        d = self.copy(self.rt, "ne")
        p = os.path.join(d, "outputfile")
        lines = read_lines(p)
        i = next(k for k, l in enumerate(lines) if l.startswith("     400"))
        lines[i] = lines[i].replace("32.00000000", "31.99000000")
        write(p, "\n".join(lines))
        self.assertEqual(check(d)[1].status_of("electron_number"), "fail")

    def test_kinematic_force_term_matters(self):
        # dt = 0.2 fs: the exact Verlet term sum |F|^2 dt^2/(8m) is far above the rounding tolerance
        run = Run(self.bo)
        t = run.dft_md()
        tt = list(t.col("Time"))
        terms, bound = physics_md._force_term(run, run.trj(), tt, run.dt_fs / AU_TIME_FS)
        spread = max(x for x in terms if x is not None) - min(x for x in terms if x is not None)
        self.assertGreater(spread, 1e-7)
        rep = Report(self.bo, "dft_md")
        physics_md.check_kinematic_identity(run, rep, list(t.col("Tene")), list(t.col("E_work")), tt)
        self.assertEqual(rep.checks[0]["status"], "pass")

    def test_dft_md_inconsistent_columns_fail(self):
        d = self.copy(self.bo, "cols")
        replace_field(os.path.join(d, "Si_dft_md.data"), 30, 7, lambda t: fixtures.fe(float(t) + 1e-6))
        self.assertEqual(check(d)[1].status_of("dft_md_columns"), "fail")

    def test_dft_md_step_not_converged_fails(self):
        d = self.copy(self.bo, "noconv")
        p = os.path.join(d, "outputfile")
        lines = read_lines(p)
        k = next(i for i, l in enumerate(lines) if l.startswith("# 1.time[au]")) + 20
        lines[k] = lines[k][:-15] + fixtures.fe(5e-6, 8, 2).rjust(15)
        write(p, "\n".join(lines))
        self.assertEqual(check(d)[1].status_of("md_step_scf_converged"), "fail")


class Tolerances(unittest.TestCase):
    def test_every_limit_is_documented(self):
        for name in TOLERANCES:
            d = describe(name)
            self.assertTrue(d["why"] and d["applies"] and d["unit"], name)

    def test_cli_lists_them(self):
        buf = io.StringIO()
        with redirect_stdout(buf):
            self.assertEqual(main(["tolerances"]), 0)
        self.assertIn("energy_conservation_max_per_atom", buf.getvalue())


if __name__ == "__main__":
    unittest.main()
