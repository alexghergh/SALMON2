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
"""Shared helpers for the unit tests (stdlib only)."""
import os
import shutil
import tempfile
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def write(path, text):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as fh:
        fh.write(text)


def read_lines(path):
    with open(path) as fh:
        return fh.read().split("\n")


def replace_field(path, data_row, field, new):
    """Replace whitespace field `field` (1-based) of data row `data_row` (0-based, '#' lines
    skipped; negative counts from the end), keeping the field width."""
    lines = read_lines(path)
    rows = [i for i, l in enumerate(lines) if l.strip() and not l.lstrip().startswith("#")]
    i = rows[data_row]
    toks = lines[i].split()
    old = toks[field - 1]
    val = new(old) if callable(new) else new
    start = 0
    for _ in range(field - 1):
        start = lines[i].index(toks[_], start) + len(toks[_])
    a = lines[i].index(old, start)
    lines[i] = lines[i][:a] + val.rjust(len(old)) + lines[i][a + len(old):]
    write(path, "\n".join(lines))
    return old


class TempDirCase(unittest.TestCase):
    """A test case with a temporary directory removed afterwards."""

    def setUp(self):
        self.d = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.d)

    def copy(self, src, name):
        dst = os.path.join(self.d, name)
        if os.path.exists(dst):
            shutil.rmtree(dst)
        shutil.copytree(src, dst)
        return dst
