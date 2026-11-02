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
"""Cross-run comparison (layer 3): exact and tolerance policies."""
import os
import shutil
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from io import StringIO

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from tests.common import TempDirCase, read_lines, replace_field, write  # noqa: E402
from mddc_verify import fixtures  # noqa: E402
from mddc_verify.cli import main  # noqa: E402
from mddc_verify.compare import compare_runs  # noqa: E402
from mddc_verify.run import Run  # noqa: E402


class Compare(TempDirCase):
    @classmethod
    def setUpClass(cls):
        cls.base = tempfile.mkdtemp()
        cls.md = fixtures.ehrenfest(os.path.join(cls.base, "md"), pulse=True, nt=800)
        cls.gs = fixtures.gs(os.path.join(cls.base, "gs"))
        cls.dc = fixtures.dc(os.path.join(cls.base, "dc"))
        cls.bo = fixtures.dft_md(os.path.join(cls.base, "bo"), nt=40)

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.base)

    def test_identical_runs_pass_both_policies(self):
        for ref, mode in ((self.md, "ehrenfest"), (self.gs, "gs"), (self.dc, "dc"), (self.bo, "dft_md")):
            d = self.copy(ref, "same-" + mode)
            for policy in ("exact", "tolerance"):
                rep = compare_runs(Run(d), Run(ref), mode, policy)[0]
                self.assertEqual(rep.verdict(), "pass", "\n".join(rep.summary_lines()))

    def test_tiny_current_change(self):
        # one current sample x(1+1e-9): not identical, but well within the tolerance
        d = self.copy(self.md, "cur")
        replace_field(os.path.join(d, "Si_rt.data"), 300, 14, lambda t: fixtures.fe(float(t) * (1 + 1e-9)))
        a, b = Run(d), Run(self.md)
        ex = compare_runs(a, b, "ehrenfest", "exact")[0]
        self.assertEqual(ex.verdict(), "fail")
        self.assertIn("row 301", ex.checks[0]["message"])          # the first difference is located
        self.assertEqual(compare_runs(a, b, "ehrenfest", "tolerance")[0].verdict(), "pass")

    def test_large_current_change_fails_tolerance(self):
        d = self.copy(self.md, "cur2")
        replace_field(os.path.join(d, "Si_rt.data"), 300, 14, lambda t: fixtures.fe(float(t) * 1.01))
        rep = compare_runs(Run(d), Run(self.md), "ehrenfest", "tolerance")[0]
        self.assertEqual(rep.status_of("tol:rt.data"), "fail")

    def test_nan_never_matches(self):
        d = self.copy(self.md, "nan")
        replace_field(os.path.join(d, "Si_rt_energy.data"), 20, 2, "NaN")
        for policy in ("exact", "tolerance"):
            self.assertEqual(compare_runs(Run(d), Run(self.md), "ehrenfest", policy)[0].verdict(), "fail")

    def test_dc_energy_beyond_print_resolution_fails(self):
        d = fixtures.dc(os.path.join(self.d, "dc2"), e_per_atom_offset=6e-4 + 1e-3)
        self.assertEqual(compare_runs(Run(d), Run(self.dc), "dc", "tolerance")[0].status_of("tol:dc_energy"),
                         "fail")

    def test_cli_compare(self):
        d = self.copy(self.gs, "cli")
        with redirect_stdout(StringIO()):
            self.assertEqual(main(["compare", d, self.gs, "--json", os.path.join(self.d, "c.json")]), 0)
            p = os.path.join(d, "Si_info.data")
            write(p, "\n".join(read_lines(p)).replace("-858.2265707953129", "-858.2264707953129"))
            self.assertEqual(main(["compare", d, self.gs, "--policy", "tolerance"]), 1)
            buf = StringIO()
            with redirect_stdout(buf):
                main(["compare", d, self.gs])           # both policies: exact and tolerance fail
            self.assertIn("tolerance policy verdict: FAIL", buf.getvalue())


if __name__ == "__main__":
    unittest.main()
