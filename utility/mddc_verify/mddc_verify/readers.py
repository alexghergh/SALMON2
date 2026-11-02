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
"""Readers for SALMON's output files (release 2.3.0 and develop-2.0.0).

Formats were read from the writers, which are the same in both versions:
  src/io/write.f90  write_rt_data_3d / write_rt_data_0d  -> <sys>_rt.data
                    write_rt_energy_data                 -> <sys>_rt_energy.data
                    write_dft_md_data                    -> <sys>_dft_md.data
                    write_xyz                            -> <sys>_trj.xyz   (a2,3f18.10,"  #v=",3f18.10,"  #f=",3f18.10)
                    write_info_data / write_eigen        -> <sys>_info.data / <sys>_eigen.data
  src/gs/dc/lcfo.f90 (write of esp_tot)                  -> data_dcdft/total/<sys>_eigen.data
  src/gs/dc/lcfo_complex.f90 (develop)                   -> data_dcdft/total/<sys>_lcfo_complex_eigen.data
  stdout (outputfile): src/main.f90 "end SALMON"; src/gs/scf_iteration_dft.f90 "#GS converged",
      "#GS does not converged", "DC #SCF = ..."; src/rt/dip.f90 RT step lines (electron count);
      src/gs/main_dft_md.f90 per-MD-step lines; src/io/perflog.f90 timer table.

Every reader records problems (non-finite tokens, overflow fields, malformed or ragged rows,
truncated last line) instead of raising, so the generic layer can report them all.
"""
import os
import re

from .numeric import parse_number, OK, NONFINITE, OVERFLOW, MALFORMED, looks_numeric

_COLDEF = re.compile(r"(\d+):([^\s\[]+)\[([^\]]*)\]")


class Problem(dict):
    def __init__(self, path, line, kind, token="", field=None, note=""):
        super().__init__(file=os.path.basename(path), line=line, kind=kind, token=token,
                         field=field, note=note)


def _ends_with_newline(path):
    try:
        with open(path, "rb") as fh:
            fh.seek(0, os.SEEK_END)
            if fh.tell() == 0:
                return True
            fh.seek(-1, os.SEEK_END)
            return fh.read(1) == b"\n"
    except OSError:
        return False


class Table:
    """A numeric .data table with '#' header lines and an indexed column definition line."""

    def __init__(self, path):
        self.path = path
        self.labels, self.units = [], []
        self.rows, self.line_numbers, self.tokens = [], [], []
        self.problems = []
        self.header = []

    def index(self, label):
        try:
            return self.labels.index(label)
        except ValueError:
            return None

    def has(self, label):
        return label in self.labels

    def col(self, label):
        i = self.index(label)
        if i is None:
            return None
        return [r[i] if i < len(r) else None for r in self.rows]

    def unit(self, label):
        i = self.index(label)
        return self.units[i] if i is not None else None

    @property
    def ncols(self):
        return len(self.labels)


def read_table(path):
    t = Table(path)
    with open(path, errors="replace") as fh:
        for ln, raw in enumerate(fh, 1):
            line = raw.rstrip("\n")
            s = line.strip()
            if not s:
                continue
            if s.startswith("#"):
                t.header.append(s)
                defs = _COLDEF.findall(s)
                if defs:
                    t.labels = [d[1] for d in defs]
                    t.units = [d[2] for d in defs]
                continue
            toks = s.split()
            vals = []
            for k, tok in enumerate(toks, 1):
                v, st = parse_number(tok)
                if st != OK:
                    t.problems.append(Problem(path, ln, st, tok, k))
                vals.append(v)
            if t.labels and len(toks) != len(t.labels):
                t.problems.append(Problem(path, ln, "ragged", "", None,
                                          f"{len(toks)} fields, header defines {len(t.labels)}"))
            t.rows.append(vals)
            t.tokens.append(toks)
            t.line_numbers.append(ln)
    if not t.labels:
        t.problems.append(Problem(path, 0, "no_header", note="no column-definition header line"))
    if not _ends_with_newline(path):
        t.problems.append(Problem(path, t.line_numbers[-1] if t.line_numbers else 0,
                                  "truncated_last_line", note="file does not end with a newline"))
    return t


# --------------------------------------------------------------------------- trajectory
class Frame:
    def __init__(self, step, time, comment):
        self.step, self.time, self.comment = step, time, comment
        self.names, self.pos, self.vel, self.force = [], [], [], []
        self.temperature = None


class Trajectory:
    def __init__(self, path):
        self.path = path
        self.frames = []
        self.problems = []
        self.has_vel = False
        self.has_force = False


_STEP = re.compile(r"step=\s*(\d+)")
_TIME = re.compile(r"time\s*=?\s*([-+0-9.EeDd]+)")
_TEMP = re.compile(r"\bT=\s*(\S+)")


def _fixed_fields(body, nfields, width=18):
    return [body[i * width:(i + 1) * width] for i in range(nfields)]


def _parse_vec(path, ln, fields, problems, offset):
    out = []
    for k, f in enumerate(fields):
        tok = f.strip()
        v, st = parse_number(tok) if tok else (None, MALFORMED)
        if st != OK:
            problems.append(Problem(path, ln, st, tok, offset + k))
        out.append(v)
    return out


def _parse_atom_line(path, ln, line, problems):
    """Parse 'Si  x y z  #v= vx vy vz  #f= fx fy fz' using the fixed a2,3f18.10 layout.

    F18.10 fields can touch without a separating blank (a value of -123456.x fills all 18
    characters), so fixed-width slicing is used whenever the section has the exact width.
    """
    name = line[:2].strip()
    parts = re.split(r"\s+#([vf])=", line[2:])
    sections = {"r": parts[0]}
    for i in range(1, len(parts) - 1, 2):
        sections[parts[i]] = parts[i + 1]
    vecs = {}
    for key, txt in sections.items():
        txt = txt.rstrip()
        fields = _fixed_fields(txt, 3) if len(txt) == 54 else None
        if fields is None or any(not f.strip() for f in fields):
            fields = txt.split()
            if len(fields) != 3:
                problems.append(Problem(path, ln, "ragged", txt.strip()[:60], None,
                                        f"expected 3 values in the {key!r} section"))
                fields = (fields + ["", "", ""])[:3]
        vecs[key] = _parse_vec(path, ln, fields, problems, {"r": 2, "v": 5, "f": 8}[key])
    return name, vecs


def read_xyz(path):
    tr = Trajectory(path)
    with open(path, errors="replace") as fh:
        lines = fh.read().split("\n")
    if lines and lines[-1] == "":
        lines.pop()
    i, n = 0, len(lines)
    while i < n:
        head = lines[i].strip()
        if not head:
            i += 1
            continue
        try:
            nat = int(head)
        except ValueError:
            tr.problems.append(Problem(path, i + 1, "malformed", head[:40], None, "expected atom count"))
            break
        if i + 1 >= n:
            tr.problems.append(Problem(path, i + 1, "truncated_frame", "", None, "frame has no comment line"))
            break
        comment = lines[i + 1]
        ms, mt, mT = _STEP.search(comment), _TIME.search(comment), _TEMP.search(comment)
        fr = Frame(int(ms.group(1)) if ms else None,
                   parse_number(mt.group(1))[0] if mt else None, comment)
        if mT:
            fr.temperature, st = parse_number(mT.group(1))
            if st != OK:
                tr.problems.append(Problem(path, i + 2, st, mT.group(1), None, "frame temperature"))
        complete = True
        for a in range(nat):
            idx = i + 2 + a
            if idx >= n:
                tr.problems.append(Problem(path, idx + 1, "truncated_frame", "", None,
                                           f"frame at line {i + 1} ends after {a} of {nat} atoms"))
                complete = False
                break
            name, vecs = _parse_atom_line(path, idx + 1, lines[idx], tr.problems)
            fr.names.append(name)
            fr.pos.append(vecs.get("r"))
            if "v" in vecs:
                tr.has_vel = True
                fr.vel.append(vecs["v"])
            if "f" in vecs:
                tr.has_force = True
                fr.force.append(vecs["f"])
        if not complete:
            break
        tr.frames.append(fr)
        i += 2 + nat
    if not _ends_with_newline(path):
        tr.problems.append(Problem(path, n, "truncated_last_line", note="file does not end with a newline"))
    return tr


# --------------------------------------------------------------------------- eigen / info
class Eigen:
    def __init__(self, path):
        self.path = path
        self.unit = None
        self.blocks = {}   # (k, spin) -> list of (io, esp, occ or None)
        self.problems = []
        self.complex_format = False

    def values(self, k=1, spin=1):
        return [e for _, e, _ in self.blocks.get((k, spin), [])]


def read_eigen(path):
    """GS <sys>_eigen.data (k=..,spin=.. blocks; io esp occ) or DC-LCFO total eigen (# spin= n)."""
    eg = Eigen(path)
    k, spin = 1, 1
    with open(path, errors="replace") as fh:
        for ln, raw in enumerate(fh, 1):
            s = raw.strip()
            if not s:
                continue
            if s.startswith("#"):
                if s.startswith("# columns: ik ispin iband eigenvalue_Ha"):
                    eg.complex_format = True
                    eg.unit = "a.u."
                m = re.search(r"esp\[([^\]]+)\]", s)
                if m:
                    eg.unit = m.group(1)
                m = re.search(r"spin=\s*(\d+)", s)
                if m:
                    spin = int(m.group(1))
                continue
            m = re.match(r"k=\s*(\d+),\s*spin=\s*(\d+)", s)
            if m:
                k, spin = int(m.group(1)), int(m.group(2))
                continue
            toks = s.split()
            if eg.complex_format:
                # develop lcfo_complex.f90: 'ik ispin iband eigenvalue_Ha'
                try:
                    kk, ss, io = int(toks[0]), int(toks[1]), int(toks[2])
                except (ValueError, IndexError):
                    eg.problems.append(Problem(path, ln, "malformed", s[:40], 1))
                    continue
                v, st = parse_number(toks[3]) if len(toks) == 4 else (None, "ragged")
                if st != OK:
                    eg.problems.append(Problem(path, ln, st, toks[3] if len(toks) > 3 else "", 4))
                eg.blocks.setdefault((kk, ss), []).append((io, v, None))
                continue
            try:
                io = int(toks[0])
            except (ValueError, IndexError):
                eg.problems.append(Problem(path, ln, "malformed", toks[0] if toks else "", 1))
                continue
            vals = []
            for c, tok in enumerate(toks[1:], 2):
                v, st = parse_number(tok)
                if st != OK:
                    eg.problems.append(Problem(path, ln, st, tok, c))
                vals.append(v)
            if not vals:
                eg.problems.append(Problem(path, ln, "ragged", s, None, "no eigenvalue"))
                continue
            eg.blocks.setdefault((k, spin), []).append((io, vals[0], vals[1] if len(vals) > 1 else None))
    if not _ends_with_newline(path):
        eg.problems.append(Problem(path, 0, "truncated_last_line"))
    return eg


def read_info(path):
    """<sys>_info.data: iterations, total energy (eV), counts, box, forces."""
    info = {"path": path, "problems": [], "forces": []}
    force_unit = None
    in_force = False
    with open(path, errors="replace") as fh:
        for ln, raw in enumerate(fh, 1):
            s = raw.strip()
            m = re.match(r"Total number of iteration\s*=\s*(\S+)", s)
            if m:
                info["iterations"] = int(m.group(1)) if m.group(1).lstrip("-").isdigit() else None
            m = re.match(r"Total energy \(eV\)\s*=\s*(\S+)", s)
            if m:
                v, st = parse_number(m.group(1))
                info["total_energy_ev"] = v
                if st != OK:
                    info["problems"].append(Problem(path, ln, st, m.group(1), None, "total energy"))
            m = re.match(r"Number of (states|electrons|atoms)\s*=\s*(\S+)", s)
            if m:
                info["n_" + m.group(1)] = parse_number(m.group(2))[0]
            if s.startswith("Force ["):
                in_force, force_unit = True, s
                continue
            if in_force:
                toks = s.split()
                if len(toks) == 4 and toks[0].isdigit():
                    vec = []
                    for c, tok in enumerate(toks[1:], 2):
                        v, st = parse_number(tok)
                        if st != OK:
                            info["problems"].append(Problem(path, ln, st, tok, c, "force"))
                        vec.append(v)
                    info["forces"].append(vec)
                elif s:
                    in_force = False
            elif s and not s.startswith("Size") and not s.startswith("Grid"):
                # one-particle energies: "  1  -3.7594   2  -3.6997 ..."
                toks = s.split()
                if len(toks) >= 2 and len(toks) % 2 == 0 and all(looks_numeric(t) for t in toks):
                    for c in range(1, len(toks), 2):
                        v, st = parse_number(toks[c])
                        if st != OK:
                            info["problems"].append(Problem(path, ln, st, toks[c], c + 1, "eigenvalue"))
    info["force_unit"] = force_unit
    if not _ends_with_newline(path):
        info["problems"].append(Problem(path, 0, "truncated_last_line"))
    return info


# --------------------------------------------------------------------------- stdout
ERROR_PATTERNS = [
    (re.compile(r"Fortran runtime error", re.I), "Fortran runtime error"),
    (re.compile(r"\bforrtl\b", re.I), "Intel Fortran runtime message"),
    (re.compile(r"NVFORTRAN-[FS]-"), "nvfortran fatal message"),
    (re.compile(r"Segmentation fault|SIGSEGV|SIGBUS|SIGFPE|Floating point exception", re.I),
     "signal"),
    (re.compile(r"CUDA_ERROR|cudaError|call to cu\w+ returned error|Failing in Thread", re.I),
     "CUDA/OpenACC runtime error"),
    (re.compile(r"MPI_ABORT|MPI_Abort|application called MPI_Abort|BAD TERMINATION", re.I),
     "MPI abort"),
    (re.compile(r"^\s*ERROR STOP|^\s*error stop", re.I), "ERROR STOP"),
    (re.compile(r"Killed|out of memory|oom-kill|DUE TO TIME LIMIT|CANCELLED", re.I),
     "killed by system or scheduler"),
    # src/rt/dip.f90 subdip: electron-count guard prints "<rank> Ne= <value>" then plain STOP.
    (re.compile(r"^\s*\d+\s+(Ne=|rbox1=)"), "RT electron-count guard (dip.f90) stopped the run"),
]


class Stdout:
    def __init__(self, path):
        self.path = path
        self.end_salmon = False
        self.gs_converged = []        # (iter, residual)
        self.gs_not_converged = []    # residual token(s)
        self.gs_iter_energy = []      # (iter, E_tot eV)
        self.dc_scf = []              # (n, E_tot eV, diff)
        self.rt_steps = []            # (step, time, j1, j2, j3, Ne, Etot or None)
        self.dft_md_steps = []        # 10 numbers per MD step
        self.timer_table = False
        self.timer_rows = 0
        self.timer_nonfinite = []
        self.nonfinite_elsewhere = []
        self.errors = []
        self.masses = []              # Mass(ik) per element, in amu
        self.zps = []                 # valence charge per element
        self.lcfo_start = False
        self.lcfo_end = False
        self.chefsi_residuals = []
        self.lcfo_dim = []
        self.lcfo_complex = []        # (ik, ispin, herm_err, basis_err, ortho_err, resid_err)
        self.problems = []
        self.n_lines = 0


_DC_SCF = re.compile(r"DC #SCF =\s*(\S+)\s+Total Energy =\s*(\S+)\s+diff =\s*(\S+)")
_GS_CONV = re.compile(r"#GS converged at\s*(\S+)\s*:\s*(\S+)")
_GS_NOCONV = re.compile(r"#GS does not converged\s*:\s*(\S*)")
_GS_ITER = re.compile(r"^\s*iter=\s*(\d+)\s+Total Energy=\s*(\S+)")
_MASS = re.compile(r"Mass\(ik\)\s*=\s*(\S+)")
_ZPS = re.compile(r"Zps\(ik\), Mlps\(ik\)\s*=\s*(\S+)")
_CHEFSI = re.compile(r"CheFSI cycle:\s*\d+\s+maximum residual:\s*(\S+)")
_LCFO_DIM = re.compile(r"(lapack|eigenexa|CheFSI) diag, #dim=\s*(\d+)")
_NONFIN_TOKEN = re.compile(r"(?<![A-Za-z_])([+-]?(?:NaN|nan|Infinity|Inf|inf)|\*{3,})(?![A-Za-z_])")


def _num(path, ln, tok, problems, note):
    v, st = parse_number(tok)
    if st != OK:
        problems.append(Problem(path, ln, st, tok, None, note))
    return v


_ERR_ANY = re.compile("|".join(f"(?:{p.pattern})" for p, _ in ERROR_PATTERNS), re.I | re.M)
_NONFIN_QUICK = re.compile(r"[Nn][Aa][Nn]|[Ii][Nn][Ff]|\*\*\*")


def read_stdout(path):
    """One pass over SALMON's stdout. Lines are pre-filtered by cheap substring tests so that
    multi-GB outputfiles (e.g. with NV_ACC_NOTIFY kernel traces) stay tractable."""
    out = Stdout(path)
    in_timer = False
    rt_header = False
    md_header = False
    with open(path, errors="replace") as fh:
        for ln, raw in enumerate(fh, 1):
            out.n_lines = ln
            if raw.startswith("launch CUDA kernel") or raw.startswith("upload CUDA data") \
                    or raw.startswith("download CUDA data"):
                continue  # OpenACC runtime trace (NV_ACC_NOTIFY), not SALMON output
            line = raw.rstrip("\n")
            s = line.strip()
            if not s:
                continue
            if s == "end SALMON":
                out.end_salmon = True
                in_timer = False
                continue
            if in_timer:
                if s.startswith("=== performance"):
                    in_timer = False
                    continue
                toks = s.split()
                nums = [t for t in toks if looks_numeric(t)]
                if len(nums) >= 6:
                    out.timer_rows += 1
                    for t in nums:
                        st = parse_number(t)[1]
                        if st in (NONFINITE, OVERFLOW):
                            out.timer_nonfinite.append(Problem(path, ln, st, t, None, "timer table"))
                continue
            if s.startswith("=== application breakdown"):
                out.timer_table = True
                in_timer = True
                continue
            c0 = s[0]
            if c0.isdigit() or c0 in "-+.":
                toks = s.split()
                if rt_header and len(toks) in (6, 7, 8) and toks[0].isdigit() \
                        and all(looks_numeric(t) for t in toks[1:]):
                    vals = [int(toks[0])] + [_num(path, ln, t, out.problems, "RT step line")
                                             for t in toks[1:]]
                    out.rt_steps.append((vals, ln))
                    continue
                if md_header and len(toks) == 10 and all(looks_numeric(t) for t in toks):
                    out.dft_md_steps.append(([_num(path, ln, t, out.problems, "dft_md step line")
                                              for t in toks], ln))
                    continue
            if _ERR_ANY.search(line):
                for pat, label in ERROR_PATTERNS:
                    if pat.search(line):
                        out.errors.append({"line": ln, "kind": label, "text": s[:160]})
                        break
            if "DC #SCF" in line:
                m = _DC_SCF.search(line)
                if m:
                    n = _num(path, ln, m.group(1), out.problems, "DC #SCF iteration")
                    e = _num(path, ln, m.group(2), out.problems, "DC total energy")
                    d = _num(path, ln, m.group(3), out.problems, "DC SCF residual")
                    out.dc_scf.append((n, e, d, ln))
                    continue
            if "#GS" in line:
                m = _GS_CONV.search(line)
                if m:
                    out.gs_converged.append((_num(path, ln, m.group(1), out.problems, "GS iteration"),
                                             _num(path, ln, m.group(2), out.problems, "GS residual"), ln))
                    continue
                m = _GS_NOCONV.search(line)
                if m:
                    out.gs_not_converged.append((m.group(1), ln))
                    continue
            if s.startswith("iter="):
                m = _GS_ITER.match(line)
                if m:
                    out.gs_iter_energy.append((int(m.group(1)),
                                               _num(path, ln, m.group(2), out.problems, "GS iteration energy")))
            if "(ik)" in line:
                m = _MASS.search(line)
                if m:
                    out.masses.append(_num(path, ln, m.group(1), out.problems, "Mass(ik)"))
                m = _ZPS.search(line)
                if m:
                    out.zps.append(_num(path, ln, m.group(1), out.problems, "Zps(ik)"))
            if "LCFO" in s:
                # lcfo.f90 prints 'start DC-LCFO'; develop's lcfo_complex.f90 'start DC-LCFO complex'
                if s in ("start DC-LCFO", "start DC-LCFO complex"):
                    out.lcfo_start = True
                elif s in ("end DC-LCFO", "end DC-LCFO complex"):
                    out.lcfo_end = True
            if "CheFSI cycle" in line:
                m = _CHEFSI.search(line)
                if m:
                    out.chefsi_residuals.append(_num(path, ln, m.group(1), out.problems, "CheFSI residual"))
            if "complex LCFO k/spin" in line:
                toks = line.split(":", 1)[1].split()
                if len(toks) == 6:
                    out.lcfo_complex.append([int(toks[0]), int(toks[1])] +
                                            [_num(path, ln, t, out.problems, "complex LCFO diagnostic")
                                             for t in toks[2:]])
            if "diag, #dim=" in line:
                m = _LCFO_DIM.search(line)
                if m:
                    out.lcfo_dim.append(int(m.group(2)))
            if "time-step" in s and "electrons" in s:
                rt_header = True
                continue
            if s.startswith("# 1.time[au], 2.Tene(ion)[au]"):
                md_header = True
                continue
            if _NONFIN_QUICK.search(line):
                for m in _NONFIN_TOKEN.finditer(line):
                    if m.group(1).startswith("*") and not re.search(r"\d", line):
                        continue  # banner text such as '*** Unified Pseudopotential Format ***'
                    out.nonfinite_elsewhere.append(Problem(path, ln, "nonfinite", m.group(1), None,
                                                           s[:120]))
    return out


def read_exit_status(rundir):
    """Convention of this tool: <rundir>/salmon.exit holds SALMON's own exit status (one integer)."""
    p = os.path.join(rundir, "salmon.exit")
    if not os.path.exists(p):
        return None, "missing"
    with open(p, errors="replace") as fh:
        first = fh.readline().strip()
    try:
        return int(first), "ok"
    except ValueError:
        return None, f"malformed: {first[:40]!r}"
