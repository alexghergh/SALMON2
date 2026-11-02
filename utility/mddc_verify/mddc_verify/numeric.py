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
"""NaN-safe number parsing and comparison.

SALMON writes numbers with Fortran edit descriptors (E23.15E3, F18.10, E15.8, list-directed).
Python's float() accepts things a Fortran writer never produces (``1_0``, ``infinity``) and
silently accepts ``NaN``. Here every token is classified, and every comparison helper returns
False when any operand is not finite, so a NaN can never satisfy a check.
"""
import math
import re

# A Fortran real as SALMON writes it: optional sign, digits with optional point, optional
# exponent with E or D. Also the exponent-without-letter form that Fortran uses when a
# three-digit exponent does not fit an Ew.d field (e.g. 0.1870828864+102).
_REAL = re.compile(r"^[+-]?(\d+\.?\d*|\.\d+)([EeDd][+-]?\d+)?$")
_REAL_NOE = re.compile(r"^([+-]?(?:\d+\.?\d*|\.\d+))([+-]\d{3})$")
_NONFINITE = re.compile(r"^[+-]?(nan|inf|infinity)$", re.IGNORECASE)
_OVERFLOW = re.compile(r"^\*+$")

OK, NONFINITE, OVERFLOW, MALFORMED = "ok", "nonfinite", "overflow", "malformed"


def parse_number(token):
    """Return (value, status). value is None unless status is OK."""
    t = token.strip()
    if _NONFINITE.match(t):
        return None, NONFINITE
    if _OVERFLOW.match(t):
        return None, OVERFLOW
    if _REAL.match(t):
        v = float(t.replace("D", "E").replace("d", "e"))
        if not math.isfinite(v):  # e.g. 1E+999
            return None, NONFINITE
        return v, OK
    m = _REAL_NOE.match(t)
    if m:
        v = float(m.group(1) + "E" + m.group(2))
        if not math.isfinite(v):
            return None, NONFINITE
        return v, OK
    return None, MALFORMED


def looks_numeric(token):
    """True for anything a numeric field could contain, including NaN/Inf/**** tokens."""
    _, status = parse_number(token)
    return status != MALFORMED


def finite(*xs):
    """True only if every argument is a finite real number."""
    for x in xs:
        if x is None or isinstance(x, bool):
            return False
        try:
            if not math.isfinite(x):
                return False
        except TypeError:
            return False
    return True


def le(x, limit):
    """x <= limit, False for any non-finite operand."""
    return finite(x, limit) and x <= limit


def close(a, b, atol, rtol=0.0, scale=None):
    """|a-b| <= atol + rtol*scale (scale defaults to |b|). False for non-finite operands."""
    if scale is None:
        scale = abs(b) if finite(b) else float("nan")
    return finite(a, b, atol, rtol, scale) and abs(a - b) <= atol + rtol * scale


def max_abs(values):
    """Max |v| over values; NaN if any value is non-finite or the list is empty."""
    m = float("nan")
    first = True
    for v in values:
        if not finite(v):
            return float("nan")
        if first or abs(v) > m:
            m = abs(v)
            first = False
    return m


def linear_fit(xs, ys):
    """Least-squares y = a + b*x. Returns (a, b, rms_residual); NaNs if not computable."""
    n = len(xs)
    if n < 2 or not all(finite(x) for x in xs) or not all(finite(y) for y in ys):
        nan = float("nan")
        return nan, nan, nan
    mx = sum(xs) / n
    my = sum(ys) / n
    sxx = sum((x - mx) ** 2 for x in xs)
    if sxx == 0:
        nan = float("nan")
        return nan, nan, nan
    b = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / sxx
    a = my - b * mx
    rms = math.sqrt(sum((y - a - b * x) ** 2 for x, y in zip(xs, ys)) / n)
    return a, b, rms


def fmt(x, digits=4):
    """Format a number for messages; shows nan/inf explicitly."""
    if x is None:
        return "None"
    if isinstance(x, (int,)) and not isinstance(x, bool):
        return str(x)
    try:
        if not math.isfinite(x):
            return str(x)
    except TypeError:
        return str(x)
    return f"{x:.{digits}g}"
