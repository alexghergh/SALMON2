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
"""Minimal reader for SALMON input files (Fortran namelists plus coordinate blocks).

Only what the verifier needs: group -> {key: [values]} with lower-case keys, and the raw lines
of non key=value groups such as &atomic_coor. Values keep Fortran semantics: 1d-9 -> 1e-9,
quoted strings -> str, bare words (y, n) -> str.
"""
import re

from .numeric import parse_number, OK

_KEY = re.compile(r"([A-Za-z_][A-Za-z0-9_]*(?:\([^)]*\))?)\s*=")


def _strip_comment(line):
    out, quote = [], None
    for ch in line:
        if quote:
            out.append(ch)
            if ch == quote:
                quote = None
        elif ch in "'\"":
            quote = ch
            out.append(ch)
        elif ch == "!":
            break
        else:
            out.append(ch)
    return "".join(out)


def _split_values(text):
    vals, cur, quote = [], [], None
    for ch in text:
        if quote:
            if ch == quote:
                vals.append(("str", "".join(cur)))
                cur, quote = [], None
            else:
                cur.append(ch)
        elif ch in "'\"":
            quote = ch
            cur = []
        elif ch in ", \t":
            if cur:
                vals.append(("raw", "".join(cur)))
                cur = []
        else:
            cur.append(ch)
    if cur:
        vals.append(("raw", "".join(cur)))
    out = []
    for kind, v in vals:
        if kind == "str":
            out.append(v)
            continue
        num, status = parse_number(v)
        if status == OK:
            out.append(int(num) if re.fullmatch(r"[+-]?\d+", v) else num)
        else:
            out.append(v)
    return out


def _assignments(line):
    """Split 'a=1, b=2' into [('a','1,'), ('b','2')], ignoring '=' inside quotes."""
    positions = []
    quote = None
    for i, ch in enumerate(line):
        if quote:
            if ch == quote:
                quote = None
            continue
        if ch in "'\"":
            quote = ch
            continue
        m = _KEY.match(line, i)
        if m and (i == 0 or not (line[i - 1].isalnum() or line[i - 1] in "_(")):
            positions.append((i, m.end(), m.group(1)))
    out = []
    for n, (start, vstart, key) in enumerate(positions):
        end = positions[n + 1][0] if n + 1 < len(positions) else len(line)
        out.append((key, line[vstart:end]))
    return out


def read_namelist(path):
    """Return dict: group -> {'keys': {key: values}, 'lines': [raw non-assignment lines]}."""
    groups = {}
    current = None
    with open(path, errors="replace") as fh:
        for raw in fh:
            line = _strip_comment(raw).strip()
            if not line:
                continue
            if line.startswith("&"):
                name = line[1:].split()[0].lower() if len(line) > 1 else ""
                current = groups.setdefault(name, {"keys": {}, "lines": []})
                rest = line[1 + len(name):].strip()
                if rest and rest != "/":
                    for key, val in _assignments(rest):
                        current["keys"][_norm_key(key)] = _split_values(val)
                if rest.endswith("/"):
                    current = None
                continue
            if line == "/":
                current = None
                continue
            if current is None:
                continue
            assigns = _assignments(line)
            if assigns:
                for key, val in assigns:
                    current["keys"][_norm_key(key)] = _split_values(val)
            else:
                current["lines"].append(line)
    return groups


def _norm_key(key):
    key = key.strip().lower()
    # num_rgrid(1:3) -> num_rgrid ; file_pseudo(1) stays indexed
    m = re.fullmatch(r"([a-z_0-9]+)\((\d+):(\d+)\)", key)
    if m:
        return m.group(1)
    return key


class Inputs:
    """Convenience accessors over a parsed SALMON input."""

    def __init__(self, groups):
        self.groups = groups

    @classmethod
    def from_file(cls, path):
        return cls(read_namelist(path))

    def get(self, key, default=None, group=None, index=0):
        key = key.lower()
        groups = [group] if group else list(self.groups)
        for g in groups:
            vals = self.groups.get(g, {}).get("keys", {}).get(key)
            if vals is not None:
                if index is None:
                    return vals
                return vals[index] if index < len(vals) else default
        return default

    def yn(self, key, default="n"):
        v = self.get(key, default)
        return str(v).strip().lower()[:1] if v is not None else default

    def coordinate_lines(self):
        for g in ("atomic_coor", "atomic_red_coor"):
            if g in self.groups:
                return g, self.groups[g]["lines"]
        return None, []
