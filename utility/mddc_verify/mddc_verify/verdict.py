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
"""Check results and the overall verdict.

Status meanings:
  pass     the predicate held, with finite values
  fail     the predicate did not hold, or a value it needed was non-finite/malformed
  warn     the predicate held, but a softer limit was exceeded (never turns a fail into a pass)
  unknown  the evidence the check needs is missing (e.g. no salmon.exit); never counts as a pass
  n/a      the check does not apply to this run (reason given), e.g. no field-free window

Overall verdict: fail > unknown > warn > pass. A run is accepted only on 'pass' or 'warn'.
"""
import json
import math

PASS, FAIL, WARN, UNKNOWN, NA = "pass", "fail", "warn", "unknown", "n/a"
_RANK = {FAIL: 4, UNKNOWN: 3, WARN: 2, PASS: 1, NA: 0}


def _clean(x):
    """JSON-safe: non-finite floats become strings so they stay visible."""
    if isinstance(x, float) and not math.isfinite(x):
        return str(x)
    if isinstance(x, dict):
        return {k: _clean(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_clean(v) for v in x]
    return x


class Report:
    def __init__(self, subject, mode):
        self.subject = subject
        self.mode = mode
        self.checks = []
        self.notes = []
        self.figures = []

    def add(self, cid, layer, status, message, metrics=None, tolerance=None):
        assert status in _RANK, status
        self.checks.append({"id": cid, "layer": layer, "status": status, "message": message,
                            "metrics": metrics or {}, "tolerance": tolerance})
        return status

    def extend(self, other):
        self.checks.extend(other.checks)
        self.notes.extend(other.notes)

    def verdict(self):
        worst = max((_RANK[c["status"]] for c in self.checks), default=_RANK[UNKNOWN])
        for k, v in _RANK.items():
            if v == worst:
                return PASS if k == NA else k
        return UNKNOWN

    def counts(self):
        out = {}
        for c in self.checks:
            out[c["status"]] = out.get(c["status"], 0) + 1
        return out

    def status_of(self, cid):
        for c in self.checks:
            if c["id"] == cid:
                return c["status"]
        return None

    def to_dict(self):
        return _clean({"subject": self.subject, "mode": self.mode, "verdict": self.verdict(),
                       "counts": self.counts(), "checks": self.checks, "notes": self.notes,
                       "figures": self.figures})

    def to_json(self):
        return json.dumps(self.to_dict(), indent=1, sort_keys=False)

    def summary_lines(self):
        lines = [f"verdict: {self.verdict().upper()}  ({self.mode}: {self.subject})"]
        for c in self.checks:
            lines.append(f"  [{c['status']:>7}] {c['layer']:<8} {c['id']:<34} {c['message']}")
        return lines
