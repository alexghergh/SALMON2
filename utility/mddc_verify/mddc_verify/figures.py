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
"""Figures (matplotlib, optional) with a numbers table next to every PNG.

Style: light surface, dark ink, categorical colours in a fixed order, grey for "other", and
status colours only for states (fail/unknown/warn), each paired with a text glyph.
"""
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from .numeric import finite, fmt  # noqa: E402
from .physics_md import energy_series, field_work, tion_row0_unwritten, total_energy  # noqa: E402

SURFACE, INK, INK2, GRID, OTHER = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df", "#b9b7b0"
CATS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6"]
STATUS = {"fail": "#d03b3b", "unknown": "#fab219", "warn": "#ec835a", "pass": "#0ca30c"}


def _style():
    plt.rcParams.update({"font.size": 9, "font.family": "DejaVu Sans", "text.color": INK,
                         "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
                         "axes.edgecolor": GRID, "lines.linewidth": 1.6})


def _ax(ax):
    ax.set_facecolor(SURFACE)
    ax.grid(color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)


def _save(fig, outdir, name, table_lines):
    os.makedirs(outdir, exist_ok=True)
    png = os.path.join(outdir, name + ".png")
    fig.savefig(png, dpi=150, bbox_inches="tight", facecolor=SURFACE)
    plt.close(fig)
    md = os.path.join(outdir, name + ".md")
    with open(md, "w") as fh:
        fh.write("\n".join(table_lines) + "\n")
    return [png, md]


def run_figures(run, rep, outdir):
    _style()
    mode = rep.mode
    out = []
    es = energy_series(run)
    if mode in ("ehrenfest", "rt") and es is not None:
        H = total_energy(run, es)
        fw = field_work(run)
        n = run.natom or 1
        t = es["t_fs"]
        fig, axes = plt.subplots(1, 2, figsize=(10.5, 3.4), facecolor=SURFACE)
        ax = axes[0]
        _ax(ax)
        k_ref = 1 if tion_row0_unwritten(es) else 0  # t=0 Tion is a placeholder in SALMON output
        dH = [(h - H[k_ref]) / n * 1e3 if k >= k_ref else float("nan") for k, h in enumerate(H)]
        ax.plot(t, dH, color=CATS[0], label="H(t) − H(0)")
        resid = None
        if fw is not None and fw["max_E"] > 0:
            idx = {round(x, 6): i for i, x in enumerate(fw["t_fs"])}
            W = [fw["W_left"][idx.get(round(x, 6), 0)] / n * 1e3 for x in t]
            ax.plot(t, W, color=CATS[1], linestyle="--", label="field work W(t)")
            resid = [a - b for a, b in zip(dH, W)]
        ax.set_xlabel("time (fs)")
        ax.set_ylabel("meV per atom")
        ax.legend(frameon=False, fontsize=8)
        ax.set_title("Energy: H = Eall" + (" + Tion" if es.get("Tion") else "") + " vs field work",
                     loc="left", fontsize=9.5)
        ax = axes[1]
        _ax(ax)
        if resid is not None:
            ax.plot(t, resid, color=CATS[2], label="H − H(0) − W")
            ax.set_title("Energy-balance residual", loc="left", fontsize=9.5)
        else:
            ax.plot(t, dH, color=CATS[2], label="H − H(0)")
            ax.set_title("Energy conservation (field-free)", loc="left", fontsize=9.5)
        ax.set_xlabel("time (fs)")
        ax.set_ylabel("meV per atom")
        fig.tight_layout()
        tab = ["| quantity | value |", "|---|---|"]
        for c in rep.checks:
            if c["id"] in ("energy_conservation", "energy_balance", "kinematic_identity", "electron_number"):
                for k, v in c["metrics"].items():
                    if not isinstance(v, (dict, list)):
                        tab.append(f"| {c['id']}.{k} | {fmt(v) if isinstance(v, float) else v} |")
        tab.append(f"| final H−H(0) (meV/atom) | {fmt(dH[-1])} |")
        if resid is not None:
            tab.append(f"| max abs residual (meV/atom) | {fmt(max(abs(x) for x in resid))} |")
        out += _save(fig, outdir, "energy", tab)
    if mode == "dft_md" and run.dft_md() is not None:
        t = run.dft_md()
        tf = run.time_factor(t.unit("Time"))
        tt = [x * tf for x in t.col("Time")][1:]
        E = t.col("E_tot")[1:]
        n = run.natom or 1
        fig, ax = plt.subplots(figsize=(6, 3.2), facecolor=SURFACE)
        _ax(ax)
        ax.plot(tt, [(e - E[0]) / n * 1e3 for e in E], color=CATS[0])
        ax.set_xlabel("time (fs)")
        ax.set_ylabel("E_tot − E_tot(t1) (meV/atom)")
        ax.set_title("dft_md: Born–Oppenheimer energy conservation", loc="left", fontsize=9.5)
        out += _save(fig, outdir, "dft_md_energy",
                     ["| t (fs) | E_tot − E_tot(t1) (meV/atom) |", "|---:|---:|"] +
                     [f"| {fmt(a)} | {fmt((e - E[0]) / n * 1e3)} |" for a, e in
                      list(zip(tt, E))[::max(1, len(tt) // 20)]])
    if mode == "dc":
        so = run.stdout()
        if so and so.dc_scf:
            it = [r[0] for r in so.dc_scf]
            d = [r[2] for r in so.dc_scf]
            fig, ax = plt.subplots(figsize=(6, 3.2), facecolor=SURFACE)
            _ax(ax)
            ax.semilogy(it, [x if finite(x) and x > 0 else float("nan") for x in d], color=CATS[0])
            thr = next((c["metrics"].get("threshold") for c in rep.checks if c["id"] == "gs_convergence"), None)
            if finite(thr):
                ax.axhline(thr, color=OTHER, linestyle=":", linewidth=1.2)
                ax.text(it[0], thr, "  threshold", va="bottom", color=INK2, fontsize=8)
            ax.set_xlabel("DC SCF iteration")
            ax.set_ylabel("density residual")
            ax.set_title("DC SCF convergence ('DC #SCF' lines)", loc="left", fontsize=9.5)
            out += _save(fig, outdir, "dc_scf", ["| iteration | E (eV) | residual |", "|---:|---:|---:|"] +
                         [f"| {r[0]} | {fmt(r[1], 12)} | {fmt(r[2])} |" for r in so.dc_scf])
    return out


def compare_figures(series, rep, outdir):
    """|A − B| over time per column (log scale), threshold lines, numbers table."""
    _style()
    out = []
    for fname, cols in series.items():
        labs = [l for l in cols if any(finite(x) and x > 0 for x in cols[l]["diff"])]
        if not labs:
            continue
        def _frac(l):
            th = cols[l]["threshold"]
            m = max((x for x in cols[l]["diff"] if finite(x)), default=0)
            return m / th if finite(th) and th > 0 else 0
        labs = sorted(labs, key=lambda l: -_frac(l))[:7]
        fig, ax = plt.subplots(figsize=(7.5, 3.6), facecolor=SURFACE)
        _ax(ax)
        tab = ["| column | class | max abs diff | threshold | first nonzero (fs) | first over tol (fs) |",
               "|---|---|---:|---:|---:|---:|"]
        for i, label in enumerate(labs):
            s = cols[label]
            thr = s["threshold"] if finite(s["threshold"]) and s["threshold"] > 0 else None
            ys = [(x / thr if thr else float("nan")) if finite(x) and x > 0 else float("nan") for x in s["diff"]]
            ax.semilogy(s["t"], ys, color=CATS[i % len(CATS)], label=label, linewidth=1.4)
            fz = next((s["t"][k] for k, x in enumerate(s["diff"]) if not finite(x) or x > 0), None)
            fo = next((s["t"][k] for k, x in enumerate(s["diff"]) if not finite(x) or x > s["threshold"]), None)
            mx = max((x for x in s["diff"] if finite(x)), default=float("nan"))
            tab.append(f"| {label} | {s['class']} | {fmt(mx)} | {fmt(s['threshold'])} | {fmt(fz)} | {fmt(fo)} |")
        ax.axhline(1.0, color=INK2, linestyle=":", linewidth=1.2)
        ax.text(ax.get_xlim()[0], 1.0, " tolerance", va="bottom", color=INK2, fontsize=8)
        ax.set_xlabel("time (fs)")
        ax.set_ylabel("|candidate − reference| / tolerance")
        ax.set_title(f"Cross-run differences: {fname} (largest {len(labs)} columns)", loc="left", fontsize=9.5)
        ax.legend(frameon=False, fontsize=7.5, loc="lower right")
        out += _save(fig, outdir, "compare-" + fname.replace(".", "_"), tab)
    return out


def fault_matrix(results, outdir, name="fault-matrix"):
    """Rows: injected faults; columns: checks that can fire, plus overall/cross-run/stock verdicts."""
    _style()
    faults = [r for r in results if r["fault"] != "clean_control"]
    controls = [r for r in results if r["fault"] == "clean_control"]
    order = ["exit_status", "normal_termination", "runtime_errors", "gs_convergence", "files_present",
             "all_values_finite", "table_integrity", "timer_table",
             "energy_conservation", "energy_balance", "electron_number", "kinematic_identity",
             "tion_vs_velocities", "trajectory_bounds", "md_step_scf_converged", "dft_md_columns",
             "bo_energy_conservation", "dc_scf_history", "dc_fragment_energy_consistency", "lcfo_output",
             "dc_vs_conventional_energy", "dc_spectrum_vs_conventional"]
    seen = {c for r in results for c in r["fired"] + r["warned"]}
    checks = [c for c in order if c in seen] + sorted(seen - set(order))
    extra = ["OVERALL", "cross-run exact", "cross-run tolerance", "stock 112"]
    cols = checks + extra
    rows = controls + faults
    fig, ax = plt.subplots(figsize=(0.42 * len(cols) + 4.5, 0.24 * len(rows) + 1.8), facecolor=SURFACE)
    ax.set_facecolor(SURFACE)
    glyph = {"fail": "✕", "unknown": "?", "warn": "!", "pass": "✓", "reject": "✕", "accept": "✓"}
    for i, r in enumerate(rows):
        vals = {c: ("fail" if c in r["fired"] and c != "exit_status" else
                    ("unknown" if c in r["fired"] else ("warn" if c in r["warned"] else None))) for c in checks}
        if "exit_status" in r["fired"]:
            vals["exit_status"] = "unknown" if r["verdict"] == "unknown" else "fail"
        vals["OVERALL"] = r["verdict"]
        vals["cross-run exact"] = r["exact"] if r["exact"] in STATUS else None
        vals["cross-run tolerance"] = r["tolerance"] if r["tolerance"] in STATUS else None
        vals["stock 112"] = {"accept": "pass", "reject": "fail"}.get(r["stock"]) if r["stock"] else None
        for j, c in enumerate(cols):
            v = vals.get(c)
            if v is None:
                continue
            color = STATUS[v]
            if c in extra and v == "pass" and r["fault"] != "clean_control":
                color = "#efe9e9"  # a pass on a faulty copy = missed: pale, with a ✓ glyph
            ax.add_patch(plt.Rectangle((j + 0.06, i + 0.08), 0.88, 0.84, color=color, linewidth=0))
            g = glyph.get(v, "")
            ax.text(j + 0.5, i + 0.52, g, ha="center", va="center", fontsize=7,
                    color=("#0b0b0b" if v in ("unknown", "warn") or color == "#efe9e9" else "#ffffff"))
    ax.set_xlim(0, len(cols))
    ax.set_ylim(len(rows), 0)
    ax.set_xticks([j + 0.5 for j in range(len(cols))])
    ax.set_xticklabels(cols, rotation=60, ha="right", fontsize=7.5)
    ax.set_yticks([i + 0.5 for i in range(len(rows))])
    ax.set_yticklabels([f"{r['fault']}  [{r['fixture']}]" for r in rows], fontsize=7.5)
    ax.axvline(len(checks), color=INK2, linewidth=1.0)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.tick_params(length=0)
    ax.set_title("Fault injection: which check fires (✕ fail, ? unknown, ! warn; pale ✓ = fault missed)",
                 loc="left", fontsize=9.5)
    fig.tight_layout()
    tab = ["| fault | fixture | overall | checks that fired | exact | tolerance | stock 112 |",
           "|---|---|---|---|---|---|---|"]
    for r in rows:
        tab.append(f"| {r['fault']} | {r['fixture']} | {r['verdict']} | {', '.join(r['fired']) or '–'} | "
                   f"{r['exact'] or '–'} | {r['tolerance'] or '–'} | {r['stock'] or '–'} |")
    return _save(fig, outdir, name, tab)
