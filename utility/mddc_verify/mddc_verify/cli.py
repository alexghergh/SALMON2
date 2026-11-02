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
"""Command line interface.

  verify.py check RUNDIR [--mode auto|ehrenfest|dft_md|dc|gs|rt] [--gs-dir DIR] [--json OUT]

Exit status: 0 = pass or warn, 1 = fail, 2 = unknown (missing evidence), 3 = usage error.
"""
import argparse
import os
import sys

from .run import Run, MODES
from .verdict import Report
from .generic import generic_report

EXIT = {"pass": 0, "warn": 0, "fail": 1, "unknown": 2}
EXIT_USAGE = 3


class _Parser(argparse.ArgumentParser):
    """argparse exits with 2 on a usage error; 2 means 'unknown' here, so use 3."""

    def error(self, message):
        self.print_usage(sys.stderr)
        self.exit(EXIT_USAGE, f"{self.prog}: error: {message}\n")


def _mode(run, requested):
    if requested and requested != "auto":
        return requested
    m = run.mode()
    if m is None:
        sys.stderr.write(f"cannot infer the mode from {run.input_path}; pass --mode\n")
        raise SystemExit(EXIT_USAGE)
    return m


def check(rundir, mode="auto", gs_dir=None, sysname=None):
    """Verify one run directory; returns (Run, Report)."""
    run = Run(rundir, sysname)
    mode = _mode(run, mode)
    rep = Report(run.dir, mode)
    generic_report(run, mode, rep)
    if gs_dir:
        g = Run(gs_dir, sysname)
        sub = generic_report(g, "gs")
        st = sub.verdict()
        rep.add("start_state_gs", "generic", st if st in ("pass", "fail", "unknown", "warn") else "unknown",
                f"ground state at {gs_dir}: {st} "
                f"({', '.join(c['id'] + '=' + c['status'] for c in sub.checks if c['status'] != 'pass')})"
                if st != "pass" else f"ground state at {gs_dir}: all generic checks pass",
                {"checks": sub.checks})
    return run, rep


def main(argv=None):
    ap = _Parser(prog="verify.py", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("check", help="verify one run directory (generic checks)")
    c.add_argument("rundir")
    c.add_argument("--mode", default="auto", choices=("auto",) + MODES)
    c.add_argument("--sysname")
    c.add_argument("--gs-dir", help="ground-state run the RT/MD run restarted from")
    c.add_argument("--json", help="write the verdict JSON here")
    args = ap.parse_args(argv)

    if args.cmd == "check":
        run, rep = check(args.rundir, args.mode, args.gs_dir, args.sysname)
    for line in rep.summary_lines():
        print(line)
    for n in rep.notes:
        print("  note:", n)
    if args.json:
        os.makedirs(os.path.dirname(os.path.abspath(args.json)), exist_ok=True)
        with open(args.json, "w") as fh:
            fh.write(rep.to_json())
    return EXIT.get(rep.verdict(), 2)


if __name__ == "__main__":
    sys.exit(main())
