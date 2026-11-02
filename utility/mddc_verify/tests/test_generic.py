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
"""Generic run checks (layer 1) on synthetic RT and ground-state runs, and the CLI exit codes."""
import io
import os
import shutil
import sys
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from tests.common import TempDirCase, read_lines, replace_field, write  # noqa: E402
from mddc_verify import fixtures, generic  # noqa: E402
from mddc_verify.cli import check, main  # noqa: E402
from mddc_verify.run import Run  # noqa: E402


class FortranFormat(unittest.TestCase):
    def test_fe(self):
        self.assertEqual(fixtures.fe(-858.226563761958), "-0.858226563761958E+003")
        self.assertEqual(fixtures.fe(0.0), "0.000000000000000E+000")
        self.assertEqual(fixtures.fe(2.5e-8, 8, 2), "0.25000000E-07")

    def test_rt_row_layout(self):
        # write.f90: F16.8, 1X,E23.15E3 ..., one extra blank before the current columns
        line = fixtures.row([0.0025] + [1.0] * 15, breaks=(13,))
        self.assertTrue(line.startswith("      0.00250000  0.100000000000000E+001"))
        self.assertIn("E+001   0.1", line)
        self.assertTrue(line.endswith(" \n"))


class GenericChecks(TempDirCase):
    @classmethod
    def setUpClass(cls):
        cls.base = tempfile.mkdtemp()
        cls.rt = fixtures.rt(os.path.join(cls.base, "rt"), nt=400)
        cls.gs = fixtures.gs(os.path.join(cls.base, "gs"))

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.base)

    def test_clean_runs_pass(self):
        for d, mode in ((self.rt, "rt"), (self.gs, "gs")):
            _, rep = check(d, mode)
            self.assertEqual(rep.verdict(), "pass", "\n".join(rep.summary_lines()))

    def test_mode_inferred_from_inputfile(self):
        self.assertEqual(Run(self.rt).mode(), "rt")
        self.assertEqual(Run(self.gs).mode(), "gs")

    def test_missing_exit_status_is_unknown(self):
        d = self.copy(self.rt, "noexit")
        os.remove(os.path.join(d, "salmon.exit"))
        _, rep = check(d)
        self.assertEqual(rep.status_of("exit_status"), "unknown")
        self.assertEqual(rep.verdict(), "unknown")

    def test_nonzero_exit_fails(self):
        d = self.copy(self.rt, "exit1")
        write(os.path.join(d, "salmon.exit"), "1\n")
        self.assertEqual(check(d)[1].status_of("exit_status"), "fail")

    def test_nan_in_a_table_fails(self):
        d = self.copy(self.rt, "nan")
        replace_field(os.path.join(d, "Si_rt.data"), 200, 14, "NaN")
        _, rep = check(d)
        self.assertEqual(rep.status_of("all_values_finite"), "fail")
        self.assertEqual(rep.verdict(), "fail")

    def test_nan_in_stdout_fails(self):
        d = self.copy(self.rt, "nanne")
        p = os.path.join(d, "outputfile")
        lines = read_lines(p)
        i = next(k for k, l in enumerate(lines) if l.startswith("      20"))
        lines[i] = lines[i].replace("32.00000000", "        NaN")
        write(p, "\n".join(lines))
        self.assertEqual(check(d)[1].status_of("all_values_finite"), "fail")

    def test_truncated_table_fails(self):
        d = self.copy(self.rt, "trunc")
        p = os.path.join(d, "Si_rt.data")
        with open(p, "rb+") as fh:
            fh.truncate(int(os.path.getsize(p) * 0.6) + 7)
        self.assertEqual(check(d)[1].status_of("table_integrity"), "fail")

    def test_no_end_marker_fails(self):
        d = self.copy(self.rt, "noend")
        p = os.path.join(d, "outputfile")
        write(p, "\n".join(read_lines(p)).replace("end SALMON", ""))
        self.assertEqual(check(d)[1].status_of("normal_termination"), "fail")

    def test_runtime_error_fails(self):
        d = self.copy(self.rt, "err")
        write(os.path.join(d, "stderr.txt"), "Fortran runtime error: Floating point exception\n")
        self.assertEqual(check(d)[1].status_of("runtime_errors"), "fail")

    def test_gs_not_converged_fails(self):
        d = self.copy(self.gs, "noconv")
        p = os.path.join(d, "outputfile")
        write(p, "\n".join(read_lines(p)).replace(fixtures.GS_CONVERGED_LINE,
                                                  "  #GS does not converged : 0.12345678E-05"))
        self.assertEqual(check(d)[1].status_of("gs_convergence"), "fail")

    def test_gs_silent_non_convergence_fails(self):
        d = self.copy(self.gs, "silent")
        p = os.path.join(d, "outputfile")
        write(p, "\n".join(read_lines(p)).replace(fixtures.GS_CONVERGED_LINE + "\n", ""))
        self.assertEqual(check(d)[1].status_of("gs_convergence"), "fail")

    def test_timer_nan_only_warns(self):
        d = self.copy(self.gs, "timer")
        p = os.path.join(d, "outputfile")
        lines = read_lines(p)
        i = next(k for k, l in enumerate(lines) if l.startswith("scf iterations"))
        lines[i] = lines[i][:30] + "".join(f"{x:>12}" for x in lines[i][30:].split()[:3]) + \
            f"{'NaN':>12}" + "".join(f"{x:>12}" for x in lines[i][30:].split()[4:])
        write(p, "\n".join(lines))
        _, rep = check(d)
        self.assertEqual(rep.status_of("timer_table"), "warn")
        self.assertEqual(rep.verdict(), "warn")

    def test_threshold_units(self):
        r = Run(self.gs)
        self.assertEqual(generic.threshold_au(r), 1e-9)


class CommandLine(TempDirCase):
    def run_main(self, argv):
        buf = io.StringIO()
        with redirect_stdout(buf):
            code = main(argv)
        return code, buf.getvalue()

    def test_exit_codes_and_json(self):
        d = fixtures.gs(os.path.join(self.d, "gs"))
        out = os.path.join(self.d, "v.json")
        code, text = self.run_main(["check", d, "--json", out])
        self.assertEqual(code, 0, text)
        self.assertTrue(text.startswith("verdict: PASS"))
        with open(out) as fh:
            self.assertIn('"verdict": "pass"', fh.read())
        os.remove(os.path.join(d, "salmon.exit"))
        self.assertEqual(self.run_main(["check", d])[0], 2)
        write(os.path.join(d, "salmon.exit"), "0\n")
        replace_field(os.path.join(d, "Si_eigen.data"), 3, 2, "NaN")
        self.assertEqual(self.run_main(["check", d])[0], 1)

    def test_usage_error_is_3(self):
        with redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit) as cm:
                main(["check"])
        self.assertEqual(cm.exception.code, 3)
        os.makedirs(os.path.join(self.d, "empty"))
        with redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit) as cm:
                main(["check", os.path.join(self.d, "empty")])   # no inputfile: mode unknown
        self.assertEqual(cm.exception.code, 3)


if __name__ == "__main__":
    unittest.main()
