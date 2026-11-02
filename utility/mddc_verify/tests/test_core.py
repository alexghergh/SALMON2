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
"""Core: NaN-safe numbers, the input-file reader and the output-format readers."""
import math
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from tests.common import TempDirCase, write  # noqa: E402
from mddc_verify import numeric, namelist, readers  # noqa: E402


class Numeric(unittest.TestCase):
    def test_parse(self):
        P = numeric.parse_number
        self.assertEqual(P("NaN")[1], numeric.NONFINITE)
        self.assertEqual(P("-Infinity")[1], numeric.NONFINITE)
        self.assertEqual(P("inf")[1], numeric.NONFINITE)
        self.assertEqual(P("*******")[1], numeric.OVERFLOW)
        self.assertEqual(P("1e999")[1], numeric.NONFINITE)
        self.assertEqual(P("1_0")[1], numeric.MALFORMED)      # float() would accept this
        self.assertAlmostEqual(P("1.5D-3")[0], 1.5e-3)
        self.assertAlmostEqual(P("0.1870828864+102")[0], 0.1870828864e102)
        self.assertAlmostEqual(P("-0.858226563761958E+003")[0], -858.226563761958)

    def test_nan_never_passes(self):
        nan = float("nan")
        self.assertFalse(numeric.close(nan, 1.0, 1e9))
        self.assertFalse(numeric.close(1.0, nan, 1e9))
        self.assertFalse(numeric.le(nan, 1.0))
        self.assertFalse(numeric.le(1.0, nan))
        self.assertTrue(numeric.close(1.0, 1.0 + 1e-12, 1e-9))
        self.assertTrue(math.isnan(numeric.max_abs([1.0, nan])))
        self.assertTrue(all(math.isnan(x) for x in numeric.linear_fit([0, 1, 2], [0, nan, 2])))


class Namelist(TempDirCase):
    def test_parse(self):
        p = os.path.join(self.d, "inputfile")
        write(p, """&calculation
  theory = 'dft'   ! comment, with = sign
  yn_dc = y
/
&system
  al = 21.9679712862 , 21.9679712862 , 21.9679712862
  nelec  = 2048
/
&scf
  threshold = 1.0d-9
/
&pseudo
  file_pseudo(1) = './Si_rps.dat'
/
&rgrid
  num_rgrid(1:3) = 80,80,80
/
&atomic_coor
  'Si'  0.010  0.000  0.000  1
/
""")
        inp = namelist.Inputs.from_file(p)
        self.assertEqual(inp.get("theory"), "dft")
        self.assertEqual(inp.get("yn_dc"), "y")
        self.assertEqual(inp.get("al", index=None), [21.9679712862] * 3)
        self.assertEqual(inp.get("nelec"), 2048)
        self.assertAlmostEqual(inp.get("threshold"), 1e-9)
        self.assertEqual(inp.get("file_pseudo(1)"), "./Si_rps.dat")
        self.assertEqual(inp.get("num_rgrid", index=None), [80, 80, 80])
        self.assertEqual(inp.coordinate_lines()[0], "atomic_coor")


class Readers(TempDirCase):
    def test_table_problems(self):
        p = os.path.join(self.d, "x_rt_energy.data")
        write(p, "# 1:Time[fs] 2:Eall[eV] 3:Eall-Eall0[eV]# 4:Tion[eV]\n"
                 "  0.0 -0.1E+001 0.0 0.1\n  0.1 NaN 0.0 0.1\n  0.2 -0.1E+001 0.0\n  0.3 -0.1")
        t = readers.read_table(p)
        self.assertEqual(t.labels, ["Time", "Eall", "Eall-Eall0", "Tion"])
        kinds = {x["kind"] for x in t.problems}
        self.assertIn("nonfinite", kinds)
        self.assertIn("ragged", kinds)
        self.assertIn("truncated_last_line", kinds)

    def test_xyz_fixed_width_glued(self):
        p = os.path.join(self.d, "x_trj.xyz")
        a = f"{1.0:18.10f}{-123456.1234567890:18.10f}{2.0:18.10f}"
        self.assertNotIn(" -123456", a)  # fields touch
        line = "Si" + a + "  #v=" + f"{0.1:18.10f}" * 3 + "  #f=" + f"{0.2:18.10f}" * 3
        write(p, f"           1\n #rt   step=      10   time=    0.100000E+01   T=    300.0000\n{line}\n")
        tr = readers.read_xyz(p)
        self.assertEqual(len(tr.frames), 1)
        self.assertAlmostEqual(tr.frames[0].pos[0][1], -123456.123456789)
        self.assertEqual(tr.frames[0].step, 10)
        self.assertEqual(tr.problems, [])

    def test_xyz_truncated(self):
        p = os.path.join(self.d, "x_trj.xyz")
        line = "Si" + f"{0.0:18.10f}" * 3
        write(p, f"           2\n comment\n{line}\n")
        tr = readers.read_xyz(p)
        self.assertEqual(len(tr.frames), 0)
        self.assertIn("truncated_frame", {x["kind"] for x in tr.problems})

    def test_stdout(self):
        p = os.path.join(self.d, "outputfile")
        write(p, "*** Unified Pseudopotential Format ***\n Mass(ik) =    28.0\n"
                 "DC #SCF =      1   Total Energy = -0.33906646E+04   diff =  0.10000000E-01\n"
                 "DC #SCF =      2   Total Energy = -0.33906646E+04   diff =             NaN\n"
                 "  #GS does not converged : 0.12E-03\n"
                 " start DC-LCFO complex\n CheFSI diag, #dim=        640\n end DC-LCFO complex\n"
                 "complex LCFO k/spin, hermitian/basis/eigen/residual:     1     1   1.0E-12   2.0E-12   3.0E-12   4.0E-11\n"
                 "=== application breakdown [s] ===\n"
                 "rt iterations   0.1E+01 0.1E+01 0.1E+01 NaN 0.0E+00 Inf\n=== performance [GFLOPS] ===\n"
                 "Fortran runtime error: boom\nend SALMON\n")
        so = readers.read_stdout(p)
        self.assertTrue(so.end_salmon)
        self.assertEqual(len(so.dc_scf), 2)
        self.assertIsNone(so.dc_scf[1][2])                 # the NaN residual is not a number
        self.assertTrue(so.problems)                       # ... and it is recorded as a problem
        self.assertEqual(len(so.gs_not_converged), 1)
        self.assertEqual(so.masses, [28.0])
        self.assertEqual(len(so.timer_nonfinite), 2)
        self.assertEqual(len(so.errors), 1)
        self.assertEqual(so.nonfinite_elsewhere, [])       # the banner is not an overflow field
        self.assertEqual(len(so.lcfo_complex), 1)
        self.assertTrue(so.lcfo_start and so.lcfo_end)     # develop's complex-LCFO markers
        self.assertEqual(so.lcfo_dim, [640])

    def test_eigen_formats(self):
        p = os.path.join(self.d, "a.data")
        write(p, "# 1:io, 2:esp[a.u.]\n# spin=     1\n     1  -0.1E+000\n     2   0.2E+000\n")
        e = readers.read_eigen(p)
        self.assertEqual(e.values(1, 1), [-0.1, 0.2])
        q = os.path.join(self.d, "b.data")
        write(q, "# columns: ik ispin iband eigenvalue_Ha\n 1 1 1 -0.18\n 2 1 1 -0.17\n")
        e = readers.read_eigen(q)
        self.assertEqual(sorted(e.blocks), [(1, 1), (2, 1)])

    def test_exit_status(self):
        self.assertEqual(readers.read_exit_status(self.d), (None, "missing"))
        write(os.path.join(self.d, "salmon.exit"), "139\n")
        self.assertEqual(readers.read_exit_status(self.d), (139, "ok"))


if __name__ == "__main__":
    unittest.main()
