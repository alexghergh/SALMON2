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
"""Optional figures: run only where matplotlib is installed; the checks never need it."""
import importlib.util
import io
import os
import sys
import unittest
from contextlib import redirect_stdout

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from tests.common import TempDirCase  # noqa: E402
from mddc_verify import fixtures  # noqa: E402
from mddc_verify.cli import main  # noqa: E402

HAVE_MPL = importlib.util.find_spec("matplotlib") is not None


class Figures(TempDirCase):
    def run_main(self, argv):
        with redirect_stdout(io.StringIO()) as buf:
            code = main(argv)
        return code, buf.getvalue()

    @unittest.skipUnless(HAVE_MPL, "matplotlib not installed")
    def test_check_and_compare_figures(self):
        md = fixtures.ehrenfest(os.path.join(self.d, "md"), pulse=True, nt=400)
        figs = os.path.join(self.d, "figs")
        self.assertEqual(self.run_main(["check", md, "--figures", figs])[0], 0)
        self.assertTrue(os.path.exists(os.path.join(figs, "energy.png")))
        self.assertTrue(os.path.exists(os.path.join(figs, "energy.md")))   # numbers next to the PNG

    @unittest.skipIf(HAVE_MPL, "matplotlib installed")
    def test_figures_skipped_without_matplotlib(self):
        gs = fixtures.gs(os.path.join(self.d, "gs"))
        code, text = self.run_main(["check", gs, "--figures", os.path.join(self.d, "figs")])
        self.assertEqual(code, 0)                  # the verdict does not depend on figures
        self.assertIn("figures skipped", text)


if __name__ == "__main__":
    unittest.main()
