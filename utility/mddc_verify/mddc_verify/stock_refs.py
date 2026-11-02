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
"""The developer's own testsuite reference values, compared NaN-safely.

Values and tolerances are copied from the stock `verification` scripts of release 2.3.0 and of
develop-2.0.0 (autumn 2026). Most stock scripts in those versions fail only when
abs(x - ref) > tol, which is False for NaN; here a non-finite value always fails. Selected lines are read exactly as the stock script reads
them (raw line number, whitespace field), so a pass here means the stock check would also pass.
"""
import os

from .numeric import parse_number, close, fmt, OK
from .readers import read_eigen
from .verdict import PASS, FAIL, UNKNOWN, NA

LAYER = "reference"

# case -> version -> list of (kind, spec...)
REFS = {
    "124_periodic_H2O_md_dp": {
        "any": [("line", "H2O_rt_energy.data", 108, 2, -0.466289445223927E+003, 1e-6,
                 "Eall at step 1000 [eV]")]},
    "126_periodic_H2O_dft_md_dp": {
        "any": [("line", "H2O_dft_md.data", 33, 7, -0.466675122289146E+003, 1e-3,
                 "E_tot at MD step 20 [eV]")]},
    "129_Si_dcdft_lcfo_chefsi": {
        "2.3.0": [("eigen", "data_dcdft/total/Si_eigen.data", {1: 0.1188965147061483,
                                                             33: 0.7334642415599671,
                                                             66: 1.536012514823663}, 1e-7)],
        "develop": [("eigen", "data_dcdft/total/Si_eigen.data",
                     {1: -0.1997182961, 8: -0.04542507571, 25: -0.03789661523, 26: 0.03123561295,
                      40: 0.1122554328, 42: 0.1240134227, 43: 0.1240134228, 44: 0.1240134235,
                      45: 0.1240134258, 46: 0.1243798737, 47: 0.1243798737, 66: 0.2481420191}, 1e-4),
                    ("dc_energy", -3390.6646, 2e-2)]},
    "130_Si_dcdft_lcfo_complex": {
        "develop": [("dc_energy", -1702.2269, 1e-3),
                    ("eigen_k", "data_dcdft/total/Si_2x1x1_dc_lcfo_complex_lcfo_complex_eigen.data",
                     {(1, 1, 1): -1.8262711562304929e-1, (1, 1, 43): 3.3893699888591583e-1,
                      (2, 1, 1): -1.8262711562315798e-1, (2, 1, 43): 3.3893678852570441e-1,
                      (3, 1, 1): -1.8262711562315800e-1, (3, 1, 43): 3.3893684399354629e-1,
                      (4, 1, 1): -1.8262711562304754e-1, (4, 1, 43): 3.3893684854022965e-1}, 1e-4)]},
}


def detect_case(rundir):
    base = os.path.basename(os.path.normpath(rundir))
    return base if base in REFS else None


def reference_checks(run, rep, case, version=None):
    refs = REFS.get(case)
    if refs is None:
        return rep.add("stock_reference", LAYER, NA, f"no stored reference for {case}")
    specs = refs.get("any") or refs.get(version or "")
    if specs is None:
        return rep.add("stock_reference", LAYER, UNKNOWN, f"{case}: references differ between "
                       f"versions {sorted(refs)}; pass --salmon-version")
    for spec in specs:
        kind = spec[0]
        if kind == "line":
            _, fname, line, field, ref, tol, what = spec
            p = os.path.join(run.dir, fname)
            if not os.path.exists(p):
                rep.add(f"stock_reference:{fname}", LAYER, FAIL, f"{fname} missing")
                continue
            with open(p, errors="replace") as fh:
                lines = fh.read().split("\n")
            tok = lines[line - 1].split()[field - 1] if len(lines) >= line and \
                len(lines[line - 1].split()) >= field else None
            v, st = parse_number(tok) if tok else (None, "missing")
            ok = st == OK and close(v, ref, tol)
            rep.add(f"stock_reference:{fname}:{line}", LAYER, PASS if ok else FAIL,
                    f"{what}: {tok} vs reference {ref} (tol {tol})" + ("" if st == OK else f" [{st}]"),
                    {"value": v, "reference": ref, "tolerance": tol, "token": tok})
        elif kind == "dc_energy":
            _, ref, tol = spec
            so = run.stdout()
            e = so.dc_scf[-1][1] if so and so.dc_scf else None
            ok = close(e, ref, tol)
            rep.add("stock_reference:dc_energy", LAYER, PASS if ok else FAIL,
                    f"last 'DC #SCF' energy {fmt(e, 12)} eV vs reference {ref} (tol {tol})",
                    {"value": e, "reference": ref, "tolerance": tol})
        elif kind in ("eigen", "eigen_k"):
            _, fname, table, tol = spec
            p = os.path.join(run.dir, fname)
            if not os.path.exists(p):
                rep.add("stock_reference:eigen", LAYER, FAIL, f"{fname} missing")
                continue
            eg = read_eigen(p)
            bad = []
            for key, ref in table.items():
                if kind == "eigen":
                    vals = dict((io, e) for io, e, _ in eg.blocks.get(sorted(eg.blocks)[0], [])) \
                        if eg.blocks else {}
                    v = vals.get(key)
                else:
                    vals = dict((io, e) for io, e, _ in eg.blocks.get((key[0], key[1]), []))
                    v = vals.get(key[2])
                if not close(v, ref, tol):
                    bad.append((key, v, ref))
            rep.add("stock_reference:eigen", LAYER, FAIL if bad else PASS,
                    f"{len(table) - len(bad)}/{len(table)} reference eigenvalues within {tol}"
                    + (f"; first mismatch {bad[0]}" if bad else ""), {"mismatches": bad[:5]})
    return rep
