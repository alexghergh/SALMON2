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
"""The developer's testsuite reference values, re-applied NaN-safely."""
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from tests.common import TempDirCase, write  # noqa: E402
from mddc_verify import stock_refs  # noqa: E402
from mddc_verify.run import Run  # noqa: E402
from mddc_verify.verdict import Report  # noqa: E402

CASE = "124_periodic_H2O_md_dp"


class StockReferences(TempDirCase):
    def run_124(self, value):
        d = os.path.join(self.d, CASE)
        lines = ["# header"] * 107 + [f"  0.25000000  {value}  0.0"]   # line 108, field 2
        write(os.path.join(d, "H2O_rt_energy.data"), "\n".join(lines) + "\n")
        rep = Report(d, "ehrenfest")
        self.assertEqual(stock_refs.detect_case(d), CASE)
        stock_refs.reference_checks(Run(d), rep, CASE)
        return rep.checks[0]["status"]

    def test_reference_value_passes(self):
        self.assertEqual(self.run_124("-0.466289445223927E+003"), "pass")

    def test_nan_and_wrong_values_fail(self):
        self.assertEqual(self.run_124("NaN"), "fail")
        self.assertEqual(self.run_124("-0.466289455223927E+003"), "fail")

    def test_version_needed_where_references_differ(self):
        d = os.path.join(self.d, "129_Si_dcdft_lcfo_chefsi")
        os.makedirs(d)
        rep = Report(d, "dc")
        stock_refs.reference_checks(Run(d), rep, "129_Si_dcdft_lcfo_chefsi")
        self.assertEqual(rep.checks[0]["status"], "unknown")


if __name__ == "__main__":
    unittest.main()
