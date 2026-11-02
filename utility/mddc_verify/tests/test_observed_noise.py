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
"""The observed-noise table of tools/observed_noise.py."""
import io
import os
import sys
import unittest
from contextlib import redirect_stdout

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from tests.common import ROOT, TempDirCase  # noqa: E402
from mddc_verify import fixtures  # noqa: E402

sys.path.insert(0, os.path.join(ROOT, "tools"))
import observed_noise  # noqa: E402


class ObservedNoise(TempDirCase):
    def test_pairs_and_conservation(self):
        a = fixtures.rt(os.path.join(self.d, "a"), nt=400)
        b = fixtures.rt(os.path.join(self.d, "b"), nt=400)
        nve = fixtures.rt(os.path.join(self.d, "nve"), nt=400, pulse=False)
        with redirect_stdout(io.StringIO()):
            rows, cons, md = observed_noise.main(os.path.join(self.d, "out"), [("same", a, b)], [nve])
        self.assertEqual(rows[0]["exact"], "pass")
        self.assertEqual(rows[0].get("first_nonzero_t"), None)
        self.assertEqual(cons[0]["status"], "pass")
        self.assertTrue(os.path.exists(os.path.join(self.d, "out", "observed-noise.md")))


if __name__ == "__main__":
    unittest.main()
