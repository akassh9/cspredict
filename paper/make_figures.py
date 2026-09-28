"""Figures for cspredict.tex, drawn from the fitted models and the saved final evaluation runs.

Run from the repository root once the pooled models are built and calibrated and the final test
run is saved (docs/results.md):

    .venv/bin/python paper/make_figures.py

Reads data/maps/de_mirage/hltv+xego/ and outputs/final/test_pooled/, writes paper/figures/*.pdf,
and prints the paired comparisons quoted in the paper that are not part of the evaluate report.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.patheffects as pe  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import polars as pl  # noqa: E402
from matplotlib.collections import LineCollection  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap, ListedColormap, LogNorm, to_rgb  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Circle, Patch  # noqa: E402
from matplotlib.ticker import PercentFormatter  # noqa: E402

from cspredict import calibrate, evaluate  # noqa: E402
from cspredict.build import Models, load_models, model_dir  # noqa: E402
from cspredict.config import CELL, OUTPUT_DIR  # noqa: E402
from cspredict.dataset import list_demos  # noqa: E402
from cspredict.infostate import episodes  # noqa: E402
from cspredict.review import HDR_MASS, RoundRenderer, key_steps, round_beliefs  # noqa: E402
from cspredict.visibility import DIST_EDGES, MAX_RANGE, P_MAX, SMOKE_RADIUS  # noqa: E402

OUT = Path(__file__).resolve().parent / "figures"
RUN_TEST = OUTPUT_DIR / "final" / "test_pooled"
EXAMPLE = ("hotspawn-showdown-2026-eac-vs-lilmix", 9, "ct")  # the held-out round shown in the README

# Reference palette of the dataviz method, light mode on a white page.
INK, INK2, MUTED = "#0b0b0b", "#52514e", "#898781"
GRID, AXIS = "#e1e0d9", "#c3c2b7"
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
BLUE_RAMP = ["#cde2fb", "#b7d3f6", "#9ec5f4", "#86b6ef", "#6da7ec", "#5598e7", "#3987e5",
             "#2a78d6", "#256abf", "#1c5cab", "#184f95", "#104281", "#0d366b"]  # fmt: skip
HEAT = LinearSegmentedColormap.from_list("heat", BLUE_RAMP[3:])  # the lightest shown step stays off the floor gray
SEQ = LinearSegmentedColormap.from_list("seq", BLUE_RAMP)
FULL, HALF = 6.5, 3.2  # inches: the paper's text width and about half of it
HALO = [pe.withStroke(linewidth=1.6, foreground="white")]

CALLOUT = {
    "BackAlley": "Back Alley", "BombsiteA": "Bombsite A", "BombsiteB": "Bombsite B", "CTSpawn": "CT Spawn",
    "PalaceAlley": "Palace Alley", "PalaceInterior": "Palace", "SideAlley": "Side Alley", "SnipersNest": "Snipers Nest",
    "TRamp": "T Ramp", "TSpawn": "T Spawn", "TopofMid": "Top of Mid",
}  # fmt: skip

plt.rcParams.update({
    "font.family": "Arial", "font.size": 8, "axes.labelsize": 8, "axes.titlesize": 8, "axes.titlelocation": "left",
    "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 7,
    "axes.edgecolor": AXIS, "axes.linewidth": 0.6, "axes.labelcolor": INK2, "axes.titlecolor": INK,
    "xtick.color": AXIS, "ytick.color": AXIS, "xtick.labelcolor": INK2, "ytick.labelcolor": INK2,
    "xtick.major.width": 0.6, "ytick.major.width": 0.6, "xtick.major.size": 2.5, "ytick.major.size": 2.5,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.5, "grid.linestyle": "-", "axes.axisbelow": True,
    "lines.linewidth": 1.5, "lines.solid_capstyle": "round", "lines.solid_joinstyle": "round",
    "legend.frameon": False, "legend.handlelength": 1.6, "hatch.linewidth": 0.4,
    "pdf.fonttype": 42, "savefig.bbox": "tight", "savefig.pad_inches": 0.02,
})  # fmt: skip


def save(fig: plt.Figure, name: str) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / f"{name}.pdf")
    plt.close(fig)
    print(f"wrote {OUT / name}.pdf")


def map_axes(ax: plt.Axes, extent: tuple[float, float, float, float]) -> None:
    ax.set_xlim(extent[0] + 200, extent[1] - 200)
    ax.set_ylim(extent[2] + 200, extent[3] - 200)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.grid(False)
    for spine in ax.spines.values():
        spine.set_visible(False)


def callout_labels(ax: plt.Axes, labels, size: float, color: str = INK2) -> None:
    for name, x, y in labels:
        ax.text(x, y, CALLOUT.get(name, name), fontsize=size, color=color, ha="center", va="center",
                path_effects=HALO, zorder=8)  # fmt: skip


# ---------------------------------------------------------------------- figure 1: a held-out round
def draw_round(ax: plt.Axes, r: RoundRenderer, s: int) -> None:
    ep, g = r.ep, r.m.grid
    floor = np.zeros((g.fine_ny, g.fine_nx, 4))
    floor[..., :3] = to_rgb(GRID)
    floor[..., 3] = (g.fine_count > 0) * 1.0
    ax.imshow(floor, extent=r.fine_extent, origin="lower", interpolation="nearest", zorder=0)
    h = r.heat(s)
    if h.max() > 0:
        flat = np.sort(h.ravel())[::-1]
        k = min(int(np.searchsorted(np.cumsum(flat), HDR_MASS * flat.sum())), len(flat) - 1)
        rgba = HEAT(np.sqrt(h / h.max()))
        rgba[..., 3] = h >= max(flat[k], 1e-9)
        ax.imshow(rgba, extent=r.extent, origin="lower", interpolation="nearest", zorder=1)
    for x, y in ep.smokes[s]:
        ax.add_patch(Circle((x, y), SMOKE_RADIUS, facecolor=(*to_rgb(MUTED), 0.55), lw=0, zorder=3))
    fire_r = r.m.fires.radius if r.m.fires is not None else 150.0
    for x, y, _ in ep.fires[s]:
        ax.add_patch(Circle((x, y), fire_r, facecolor=(*to_rgb(AXIS), 0.45), edgecolor=(*to_rgb(INK2), 0.8),
                            hatch="//////", lw=0.4, zorder=3))  # fmt: skip
    callout_labels(ax, r.labels, 4.0, MUTED)
    for f in range(len(ep.obs_ids)):
        x, y, _ = ep.obs_xyz[s, f]
        if np.isfinite(x) and ep.obs_alive[s, f]:
            yaw = np.radians(ep.obs_yaw[s, f])
            ax.plot([x, x + 240 * np.cos(yaw)], [y, y + 240 * np.sin(yaw)], color=INK2, lw=0.8, zorder=5)
            ax.scatter(x, y, s=16, color=INK, edgecolor="white", linewidth=0.6, zorder=6)
    for e in range(len(ep.enemy_ids)):
        if not ep.enemy_alive[s, e]:
            continue
        if ep.enemy_seen[s, e]:
            x, y = g.node_xy[ep.enemy_seen_node[s, e]]
            ax.scatter(x, y, s=20, color=ORANGE, edgecolor="white", linewidth=0.6, zorder=7)
            continue
        if r.last_seen[s][e] is not None:
            s0, node = r.last_seen[s][e]
            x, y = g.node_xy[node]
            ax.scatter(x, y, s=46, facecolor="none", edgecolor="white", linewidth=2.2, zorder=7)
            ax.scatter(x, y, s=46, facecolor="none", edgecolor=ORANGE, linewidth=1.1, zorder=7)
            ax.text(x + 80, y + 80, f"{ep.t_rel[s] - ep.t_rel[s0]:.0f} s", fontsize=5, color=INK, path_effects=HALO, zorder=8)
        x, y, _ = ep.enemy_xyz[s, e]
        ax.scatter(x, y, s=18, color=ORANGE, marker="x", linewidth=1.2, path_effects=HALO, zorder=9)
    n_f, n_e = int(ep.obs_alive[s].sum()), int(ep.enemy_alive[s].sum())
    bomb = f", bomb on {'AB'[ep.phase[s] - 1]}" if ep.phase[s] else ""
    ax.set_title(f"{ep.t_rel[s]:.1f} s   CT {n_f}v{n_e}{bomb}", fontsize=7, loc="center", pad=2)
    map_axes(ax, r.extent)


def fig_review(models: Models) -> None:
    ref = next(r for r in list_demos(["hltv"]) if r.demo_id.startswith(EXAMPLE[0]))
    assert ref.split == "test", f"the example round must be held out, not {ref.split}"
    ep = next(e for e in episodes(models.grid, ref, EXAMPLE[2]) if e.round_num == EXAMPLE[1])
    r = RoundRenderer(ep, models, round_beliefs(ep, models, "ens"), show_truth=True)
    fig, axes = plt.subplots(2, 3, figsize=(FULL, 4.6))
    for ax, s in zip(axes.ravel(), key_steps(ep, 6)):
        draw_round(ax, r, s)
    handles = [
        Line2D([], [], color=INK2, lw=0.8, marker="o", markerfacecolor=INK, markeredgecolor="white", markersize=5,
               label="CT player and view direction"),
        Line2D([], [], ls="", marker="o", markerfacecolor=ORANGE, markeredgecolor="white", markersize=5.5, label="T on the radar"),
        Line2D([], [], ls="", marker="o", markerfacecolor="none", markeredgecolor=ORANGE, markeredgewidth=1.1, markersize=6.5,
               label="T last seen (seconds ago)"),
        Line2D([], [], ls="", marker="x", color=ORANGE, markeredgewidth=1.2, markersize=5, label="T true position (scoring only)"),
        Patch(facecolor=BLUE_RAMP[7], label=f"most likely {HDR_MASS:.0%} of hidden-T probability"),
        Patch(facecolor=(*to_rgb(MUTED), 0.55), label="smoke"),
        Patch(facecolor=(*to_rgb(AXIS), 0.45), edgecolor=INK2, hatch="//////", lw=0.4, label="burning molotov"),
    ]  # fmt: skip
    fig.legend(handles=handles, loc="lower center", ncol=4, bbox_to_anchor=(0.5, 0.0), columnspacing=1.4, handletextpad=0.5)
    fig.subplots_adjust(left=0.0, right=1.0, top=0.96, bottom=0.1, wspace=0.02, hspace=0.1)
    save(fig, "review")


# ---------------------------------------------------------------------- figure: the learned grid
def fig_grid(models: Models) -> None:
    g = models.grid
    extent = (g.x0, g.x0 + g.nx * CELL, g.y0, g.y0 + g.ny * CELL)
    lowest = g.col_nodes[:, 0]
    place = np.where(lowest >= 0, g.node_place_idx[np.maximum(lowest, 0)], -1).reshape(g.ny, g.nx)
    floors = (g.col_nodes >= 0).sum(axis=1).reshape(g.ny, g.nx)

    fig, (a, b) = plt.subplots(1, 2, figsize=(FULL, 2.95))
    cells = np.where(floors == 0, 0, np.where(floors > 1, 2, 1))
    a.imshow(cells, extent=extent, origin="lower", interpolation="nearest", cmap=ListedColormap(["white", GRID, BLUE]), vmin=0, vmax=2)
    segs = []
    for iy in range(g.ny):
        for ix in range(g.nx):
            p = place[iy, ix]
            if p < 0:
                continue
            x, y = g.x0 + ix * CELL, g.y0 + iy * CELL
            if ix + 1 < g.nx and place[iy, ix + 1] >= 0 and place[iy, ix + 1] != p:
                segs.append([(x + CELL, y), (x + CELL, y + CELL)])
            if iy + 1 < g.ny and place[iy + 1, ix] >= 0 and place[iy + 1, ix] != p:
                segs.append([(x, y + CELL), (x + CELL, y + CELL)])
    a.add_collection(LineCollection(segs, colors=INK2, linewidths=0.5, zorder=2))
    labels = []
    for i, name in enumerate(g.places):
        m = g.node_place_idx == i
        if m.sum() >= 3:
            w = g.node_count[m].astype(float)
            labels.append((name, float(np.average(g.node_xy[m, 0], weights=w)), float(np.average(g.node_xy[m, 1], weights=w))))
    callout_labels(a, labels, 5.2)
    x0, y0 = extent[0] + 380, extent[2] + 330
    a.plot([x0, x0 + 500], [y0, y0], color=INK, lw=1.2, solid_capstyle="butt")
    a.text(x0 + 250, y0 + 70, "500 units", fontsize=6, color=INK2, ha="center", va="bottom")
    map_axes(a, extent)
    a.set_title(f"(a) {g.n:,} learned cells, {len(g.places)} callouts", pad=3)

    occ = models.motion.occupancy["t_pre"][6]  # T side, bomb not planted, 30-35 s after freeze time
    col = (g.node_iy * g.nx + g.node_ix).astype(np.int64)
    img = np.bincount(col, weights=occ, minlength=g.nx * g.ny).reshape(g.ny, g.nx)
    img = np.where(floors > 0, img, np.nan)
    im = b.imshow(img, extent=extent, origin="lower", interpolation="nearest", cmap=SEQ, norm=LogNorm(3e-5, 8e-3))
    map_axes(b, extent)
    b.set_title("(b) Where Ts usually are 30–35 s into a round", pad=3)
    cb = fig.colorbar(im, ax=b, fraction=0.04, pad=0.01)
    cb.outline.set_visible(False)
    cb.ax.tick_params(labelsize=6, length=2, color=AXIS)
    cb.set_label("share of T players per cell", fontsize=6.5, color=INK2)
    fig.subplots_adjust(wspace=0.03)
    save(fig, "grid")


# ---------------------------------------------------------------------- figure: the spotting model
def fig_spot(models: Models) -> None:
    s = models.spot

    def p(cls: int, d: int | slice, h: int | slice, v: int | slice) -> np.ndarray:
        return np.minimum(1.0 / (1.0 + np.exp(-(s.a[cls, d] + s.b[h] + s.c[v]))), P_MAX)

    d_edges = np.r_[0.0, DIST_EDGES, MAX_RANGE]
    fig, axes = plt.subplots(1, 3, figsize=(FULL, 2.05), sharey=True)
    for cls, color, label in ((3, BLUE, "clear ray cast"), (1, ORANGE, "clear only through carved space"), (0, AQUA, "blocked")):
        axes[0].stairs(p(cls, slice(None), 2, 0), d_edges, color=color, lw=1.5, baseline=None, label=label)
    axes[0].set_xlabel("distance (units)")
    axes[0].set_ylabel("P(on the radar)")
    axes[0].set_title("(a) By distance and sight line", pad=3)
    axes[0].legend(loc="upper right", borderaxespad=0.2)
    axes[1].stairs(p(3, 2, slice(None), 0), np.arange(0, 181, 5), color=BLUE, lw=1.5, baseline=None)
    axes[1].set_xlabel("horizontal offset (°)")
    axes[1].set_xticks([0, 45, 90, 135, 180])
    axes[1].set_title("(b) By horizontal angle", pad=3)
    axes[2].stairs(p(3, 2, 2, slice(None)), np.arange(0, 51, 5), color=BLUE, lw=1.5, baseline=None)
    axes[2].set_xlabel("vertical offset (°)")
    axes[2].set_xticks([0, 15, 30, 45])
    axes[2].set_xticklabels(["0", "15", "30", "45+"])
    axes[2].set_title("(c) By vertical angle", pad=3)
    for ax in axes:
        ax.set_ylim(0, 1)
    fig.subplots_adjust(wspace=0.12)
    save(fig, "spot")


# ---------------------------------------------------------------------- figure: molotov profiles
def fig_fire(models: Models) -> None:
    f = models.fires
    fig, ax = plt.subplots(figsize=(HALF, 2.0))
    ax.stairs(f.occupancy, f.edges, color=BLUE, lw=1.5, baseline=None, label="occupancy $O(r)$")
    ax.stairs(f.step, f.edges, color=ORANGE, lw=1.5, baseline=None, label="step multiplier $m(r)$")
    ax.axvline(f.radius, color=MUTED, lw=0.6)
    ax.text(f.radius + 4, 0.05, f"fire radius {f.radius:.0f}", fontsize=6.5, color=INK2)
    ax.set_xlim(0, float(f.edges[-1]))
    ax.set_ylim(0, 1.05)
    ax.set_xlabel("distance from the fire's centre (units)")
    ax.set_ylabel("relative to no fire")
    ax.legend(loc="upper left", borderaxespad=0.1)
    save(fig, "fire")


# ---------------------------------------------------------------------- figures from the test run
SERIES = (("ens", BLUE, "cspredict"), ("last_seen", ORANGE, "last seen"), ("prior_neg", AQUA, "usual positions"))


def fig_horizon() -> None:
    cols = ["model", "demo", "round", "friendly", "seen_before", "since", "place_rank", "p_place", "p_true", "n_friend", "n_enemy"]
    df = pl.read_parquet(RUN_TEST / "samples_test.parquet", columns=cols)
    df = df.filter(pl.col("seen_before") & pl.col("model").is_in([k for k, _, _ in SERIES]))
    df = evaluate.with_slices(df.with_columns(pl.lit(-1).alias("last_weapon")))
    rows = []
    for label in evaluate.SINCE_LABELS:
        t = evaluate.intervals(df.filter(pl.col("since_bin") == label), metrics=("place_top1", "place_nll"), reps=2000)
        rows += [{"bin": label, **r} for r in t.iter_rows(named=True)]
    hz = pl.DataFrame(rows)
    x = np.arange(len(evaluate.SINCE_LABELS))
    fig, axes = plt.subplots(1, 2, figsize=(FULL, 2.3))
    for ax, metric, ylabel in ((axes[0], "place_top1", "right callout named first"), (axes[1], "place_nll", "callout log-loss (lower is better)")):
        for key, color, label in SERIES:
            d = {r["bin"]: r for r in hz.filter(pl.col("model") == key).iter_rows(named=True)}
            y = np.array([d[b][metric] for b in evaluate.SINCE_LABELS])
            lo = np.array([d[b][f"{metric}_lo"] for b in evaluate.SINCE_LABELS])
            hi = np.array([d[b][f"{metric}_hi"] for b in evaluate.SINCE_LABELS])
            ax.errorbar(x, y, yerr=[y - lo, hi - y], color=color, marker="o", markersize=4.5, markeredgecolor="white",
                        markeredgewidth=0.8, elinewidth=0.8, capsize=0, label=label)  # fmt: skip
        ax.set_xticks(x)
        ax.set_xticklabels(["0–2", "2–5", "5–10", "10–20", "20–40", "40+"])
        ax.set_xlabel("seconds since the enemy was last seen")
        ax.set_ylabel(ylabel)
    axes[0].set_ylim(0, 1)
    axes[0].yaxis.set_major_formatter(PercentFormatter(1.0))
    axes[1].set_ylim(0, 10.5)
    axes[0].legend(loc="upper right")
    axes[0].set_title("(a) Accuracy", pad=3)
    axes[1].set_title("(b) Log-loss", pad=3)
    fig.subplots_adjust(wspace=0.25)
    save(fig, "horizon")


def reliability(p: np.ndarray, truth: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """Every callout probability binned by its value: (count, mean said, share where the enemy was, ECE)."""
    edges = np.asarray(evaluate.CALIB_EDGES)
    y = np.zeros_like(p, dtype=bool)
    y[np.arange(len(p)), truth] = True
    pv, yv = p.ravel(), y.ravel()
    b = np.clip(np.searchsorted(edges, pv, side="right") - 1, 0, len(edges) - 2)
    n = np.bincount(b, minlength=len(edges) - 1)
    said = np.bincount(b, pv, len(edges) - 1) / np.maximum(n, 1)
    there = np.bincount(b, yv, len(edges) - 1) / np.maximum(n, 1)
    return n, said, there, float((n / n.sum() * np.abs(said - there)).sum())


def fig_calibration() -> None:
    groups = (("0–5 s since last seen", 0), ("5–20 s", 1), ("20 s and longer", 2))
    data = {}
    for model in ("ens_uncal", "ens"):
        p, truth, since, _ = calibrate.arrays(RUN_TEST, "test", model)
        g = calibrate._groups(since, list(calibrate.SINCE_GROUPS))
        data[model] = [reliability(p[g == k], truth[g == k]) for _, k in groups]
        n, said, there, ece = reliability(p, truth)
        print(model, "ECE overall", round(ece, 4), "by group", [round(d[3], 4) for d in data[model]])
        print("  observed frequency by stated-probability bin (all horizons):", " ".join(f"{v:.1%}" for v in there))
    fig, axes = plt.subplots(1, 3, figsize=(FULL, 2.45), sharey=True)
    for ax, (title, k) in zip(axes, groups):
        ax.plot([0, 1], [0, 1], color=AXIS, lw=0.8, zorder=1)
        for model, color, label in (("ens_uncal", MUTED, "before calibration"), ("ens", BLUE, "after calibration")):
            n, said, there, _ = data[model][k]
            keep = n >= 200
            ax.plot(said[keep], there[keep], color=color, marker="o", markersize=4, markeredgecolor="white",
                    markeredgewidth=0.8, label=label, zorder=3)  # fmt: skip
        ax.text(0.04, 0.95, f"ECE {data['ens_uncal'][k][3]:.4f} → {data['ens'][k][3]:.4f}", transform=ax.transAxes,
                fontsize=6.5, color=INK2, va="top")  # fmt: skip
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_aspect("equal")
        ax.set_title(title, pad=3)
        ax.set_xlabel("stated probability")
        ax.set_xticks([0, 0.25, 0.5, 0.75, 1])
        ax.set_xticklabels(["0", "", "0.5", "", "1"])
    axes[0].set_ylabel("how often the enemy was there")
    axes[2].legend(loc="lower right", borderaxespad=0.2)
    axes[0].set_yticks([0, 0.25, 0.5, 0.75, 1])
    axes[0].set_yticklabels(["0", "", "0.5", "", "1"])
    fig.subplots_adjust(wspace=0.14)
    save(fig, "calibration")


def paired_table() -> None:
    """Paired differences on the test run quoted in the text (round bootstrap, 2,000 replicates)."""
    cols = ["model", "demo", "round", "friendly", "seen_before", "place_rank", "p_place", "p_true"]
    mid = pl.read_parquet(RUN_TEST / "samples_test.parquet", columns=cols).filter(pl.col("seen_before"))
    pairs = [("pf", "pf_noneg"), ("hmm", "hmm_noneg"), ("ens", "ens_nokill"), ("pf", "diffuse"), ("hmm", "diffuse"),
             ("pf", "hmm"), ("ens_nofire", "pf"), ("ens", "pf"), ("ens_nofire", "ens_seed1"), ("ens", "last_seen"), ("ens", "prior")]  # fmt: skip
    for a, b in pairs:
        r = evaluate.intervals(mid.filter(pl.col("model").is_in([a, b])), ref=b).filter(pl.col("model") == a).row(0, named=True)
        cells = [f"{m} {r['d_' + m]:+.3f} ({r['d_' + m + '_lo']:+.3f}, {r['d_' + m + '_hi']:+.3f})" for m in evaluate.HEADLINE]
        print(f"{a} - {b}: " + "; ".join(cells))


def main() -> None:
    models = load_models(model_dir(["hltv", "xego"]))
    fig_review(models)
    fig_grid(models)
    fig_spot(models)
    fig_fire(models)
    fig_horizon()
    fig_calibration()
    paired_table()


if __name__ == "__main__":
    main()
