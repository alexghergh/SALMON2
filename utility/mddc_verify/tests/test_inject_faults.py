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
"""The fault-injection campaign, in quick mode on a sample of faults."""
import io
import os
import sys
import unittest
from contextlib import redirect_stdout

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from tests.common import ROOT, TempDirCase  # noqa: E402

sys.path.insert(0, os.path.join(ROOT, "tools"))
import inject_faults  # noqa: E402


class FaultInjection(TempDirCase):
    SAMPLE = ["nan_eall", "nan_checked_112", "drift_md_1e-3", "trunc_xyz", "jm_rel1e-6",
              "dc_noconv", "gs_noconv", "exit_missing"]

    def test_quick_campaign(self):
        with redirect_stdout(io.StringIO()):
            res = inject_faults.main(os.path.join(self.d, "out"), quick=True, only=self.SAMPLE,
                                     stock_script=None)
        by = {r["fault"]: r for r in res if r["fault"] != "clean_control"}
        self.assertEqual(sorted(by), sorted(self.SAMPLE))
        bad_controls, missed = inject_faults.summary(res)
        self.assertEqual(bad_controls, [])
        self.assertEqual(missed, [])
        self.assertEqual(by["nan_checked_112"]["verdict"], "fail")
        self.assertEqual(by["exit_missing"]["verdict"], "unknown")
        self.assertEqual(by["jm_rel1e-6"]["verdict"], "pass")       # single-run checks cannot see it
        self.assertEqual(by["jm_rel1e-6"]["exact"], "fail")         # ... the cross-run layer does
        for name in ("fault-injection.json", "fault-injection.md"):
            self.assertTrue(os.path.exists(os.path.join(self.d, "out", name)))

    def test_every_fault_has_a_known_fixture(self):
        ids = [f[0] for f in inject_faults.FAULTS]
        self.assertEqual(len(ids), len(set(ids)))
        self.assertTrue(all(f[1] in inject_faults.MODE for f in inject_faults.FAULTS))


if __name__ == "__main__":
    unittest.main()
