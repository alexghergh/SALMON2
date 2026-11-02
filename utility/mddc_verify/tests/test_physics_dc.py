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
"""Physics checks for divide-and-conquer runs and their RT follow-up (layer 2)."""
import os
import shutil
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from tests.common import TempDirCase, read_lines, replace_field, write  # noqa: E402
from mddc_verify import fixtures, physics_dc  # noqa: E402
from mddc_verify.cli import check  # noqa: E402
from mddc_verify.run import Run  # noqa: E402


class DCRuns(TempDirCase):
    @classmethod
    def setUpClass(cls):
        cls.base = tempfile.mkdtemp()
        cls.conv = fixtures.conventional(os.path.join(cls.base, "conv"))
        cls.dc = fixtures.dc(os.path.join(cls.base, "dc"))
        cls.buffers = [fixtures.dc(os.path.join(cls.base, f"b{b}"), e_per_atom_offset=off, buffer=b)
                       for b, off in ((10, 8e-4), (15, 3e-4), (20, 2e-4))]

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.base)

    def test_clean_pass(self):
        self.assertEqual(Run(self.dc).mode(), "dc")
        _, rep = check(self.dc, conventional=self.conv, buffer_dirs=self.buffers)
        self.assertIn(rep.verdict(), ("pass", "warn"), "\n".join(rep.summary_lines()))
        self.assertEqual(rep.status_of("dc_buffer_sensitivity"), "pass")

    def test_not_converged_fails(self):
        # DC mode prints no convergence message: only the last 'DC #SCF' residual tells
        d = self.copy(self.dc, "dcnc")
        p = os.path.join(d, "outputfile")
        lines = read_lines(p)
        i = max(k for k, l in enumerate(lines) if "DC #SCF" in l)
        lines[i] = lines[i].rsplit("diff =", 1)[0] + "diff =  0.10000000E-05"
        write(p, "\n".join(lines))
        self.assertEqual(check(d, conventional=self.conv)[1].status_of("gs_convergence"), "fail")

    def test_energy_far_from_conventional_fails(self):
        d = fixtures.dc(os.path.join(self.d, "far"), e_per_atom_offset=0.5)
        self.assertEqual(check(d, conventional=self.conv)[1].status_of("dc_vs_conventional_energy"), "fail")

    def test_fragment_files_must_agree(self):
        d = self.copy(self.dc, "frag")
        p = os.path.join(d, "data_dcdft", "fragments", "000002", "Si_info.data")
        write(p, "\n".join(read_lines(p)).replace("Total energy (eV) =    -3390", "Total energy (eV) =    -3391"))
        self.assertEqual(check(d)[1].status_of("dc_fragment_energy_consistency"), "fail")

    def test_lcfo_spectrum_shift_fails(self):
        d = self.copy(self.dc, "shift")
        p = os.path.join(d, "data_dcdft", "total", "Si_eigen.data")
        for k in range(30, 66):
            replace_field(p, k, 2, lambda t: fixtures.fe(float(t) + 0.02, 16, 3))
        self.assertEqual(check(d, conventional=self.conv)[1].status_of("dc_spectrum_vs_conventional"), "fail")

    def test_lcfo_nan_and_small_basis_fail(self):
        d = self.copy(self.dc, "lcfo")
        replace_field(os.path.join(d, "data_dcdft", "total", "Si_eigen.data"), 30, 2, "NaN")
        p = os.path.join(d, "outputfile")
        write(p, "\n".join(read_lines(p)).replace("#dim=        640", "#dim=         40"))
        _, rep = check(d)
        self.assertEqual(rep.status_of("lcfo_output"), "fail")
        self.assertEqual(rep.status_of("all_values_finite"), "fail")

    def test_e15_resolution(self):
        self.assertAlmostEqual(physics_dc.e15_resolution(-3390.6646), 5e-5)


class RTFromDC(TempDirCase):
    def test_close_and_far(self):
        conv = fixtures.rt(os.path.join(self.d, "conv"), nt=800)
        near = fixtures.rt_from_dc(os.path.join(self.d, "near"), nt=800, ex_offset_per_atom=2e-4, noise=0.01)
        far = fixtures.rt_from_dc(os.path.join(self.d, "far"), nt=800, ex_offset_per_atom=5e-3, noise=0.01)
        _, rep = check(self.d + "/near", rt_dc=near, rt_conv=conv)
        self.assertIn(rep.status_of("rt_from_dc_vs_conventional"), ("pass", "warn"))
        _, rep = check(self.d + "/far", rt_dc=far, rt_conv=conv)
        self.assertEqual(rep.status_of("rt_from_dc_vs_conventional"), "fail")


if __name__ == "__main__":
    unittest.main()
