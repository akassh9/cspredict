"""Render a round for demo review: one team's view with a heatmap of where the enemies probably
are, optionally next to where they really were.

Usage:
    python -m cspredict.review --list <demo_id>                 # rounds with their situations
    python -m cspredict.review <demo_id> <round> [--side ct] [--model hmm] [--truth]

Writes <out>/<demo>_r<round>_<side>.gif (whole round) and a contact sheet .png of moments
after first contact.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib import colormaps  # noqa: E402
from PIL import Image  # noqa: E402

from cspredict.build import Models, load_models, model_dir  # noqa: E402
from cspredict.config import CELL, FINE_CELL, OUTPUT_DIR  # noqa: E402
from cspredict.dataset import list_demos  # noqa: E402
from cspredict.filters import ALL_CONFIGS, gather_evidence, run_filter  # noqa: E402
from cspredict.infostate import Episode, episodes  # noqa: E402
from cspredict.visibility import SMOKE_RADIUS  # noqa: E402

BG = "#15171c"
FLOOR_COLOR = np.array([0.30, 0.32, 0.36])
FRIEND = {"ct": "#5fa8ff", "t": "#f2b84b"}
ENEMY = "#ff4d4d"
SMOKE_COLOR = "#c8ced6"
FIRE_COLOR = "#ff8c1a"
HDR_MASS = 0.8  # heat shows the smallest set of cells holding this share of hidden-enemy probability


def round_beliefs(ep: Episode, models: Models, model: str) -> np.ndarray:
    """(S, E, N) beliefs for one filter configuration."""
    cfg = next(c for c in ALL_CONFIGS if c.name == model)
    ev = gather_evidence(ep, models.grid, models.spot, models.fires)
    beliefs = run_filter(ep, models.grid, models.motion, cfg, ev, models.library, models.teams, models.calibration)
    return np.stack([b.astype(np.float32) for _, b in beliefs])


def _place_labels(models: Models) -> list[tuple[str, float, float]]:
    g = models.grid
    labels = []
    for i, name in enumerate(g.places):
        m = (g.node_place_idx == i) & (g.node_count > 20)
        if name != "Unknown" and m.sum() >= 3:
            w = g.node_count[m].astype(float)
            labels.append((name, float(np.average(g.node_xy[m, 0], weights=w)), float(np.average(g.node_xy[m, 1], weights=w))))
    return labels


class RoundRenderer:
    def __init__(self, ep: Episode, models: Models, beliefs: np.ndarray, show_truth: bool):
        self.ep, self.m, self.b, self.show_truth = ep, models, beliefs, show_truth
        g = models.grid
        self.extent = (g.x0, g.x0 + g.nx * CELL, g.y0, g.y0 + g.ny * CELL)
        fine_extent = (g.x0, g.x0 + g.fine_nx * FINE_CELL, g.y0, g.y0 + g.fine_ny * FINE_CELL)
        floor = np.zeros((g.fine_ny, g.fine_nx, 4))
        floor[..., :3] = FLOOR_COLOR
        floor[..., 3] = (g.fine_count > 0) * 1.0
        self.floor, self.fine_extent = floor, fine_extent
        self.col = g.node_iy * g.nx + g.node_ix
        self.labels = _place_labels(models)
        self.last_seen = self._last_seen()
        self.cmap = colormaps["inferno"]

    def _last_seen(self) -> list[list[tuple[int, int] | None]]:
        """Per step and enemy: (step, node) of the most recent sighting, if any."""
        out, last = [], [None] * len(self.ep.enemy_ids)
        for s in range(self.ep.n_steps):
            for e, node in enumerate(self.ep.enemy_seen_node[s]):
                if node >= 0:
                    last[e] = (s, int(node))
            out.append(list(last))
        return out

    def heat(self, s: int) -> np.ndarray:
        """Expected number of off-radar enemies per map column at step s."""
        ep = self.ep
        hidden = ep.enemy_alive[s] & ~ep.enemy_seen[s]
        team = self.b[s, hidden].sum(axis=0) if hidden.any() else np.zeros(self.m.grid.n)
        g = self.m.grid
        return np.bincount(self.col, weights=team, minlength=g.nx * g.ny).reshape(g.ny, g.nx)

    def draw(self, ax: plt.Axes, s: int, compact: bool = False) -> None:
        ep, g = self.ep, self.m.grid
        ax.set_facecolor(BG)
        ax.imshow(self.floor, extent=self.fine_extent, origin="lower", interpolation="nearest")
        h = self.heat(s)
        if h.max() > 0:
            flat = np.sort(h.ravel())[::-1]
            k = min(int(np.searchsorted(np.cumsum(flat), HDR_MASS * flat.sum())), len(flat) - 1)
            shown = h >= max(flat[k], 1e-9)
            v = h / h.max()
            rgba = self.cmap(0.3 + 0.7 * v)
            rgba[..., 3] = np.where(shown, 0.35 + 0.6 * np.sqrt(v), 0.0)
            ax.imshow(rgba, extent=self.extent, origin="lower", interpolation="nearest")
        for x, y in ep.smokes[s]:
            ax.add_patch(plt.Circle((x, y), SMOKE_RADIUS, color=SMOKE_COLOR, alpha=0.35, lw=0, zorder=3))
        fire_r = self.m.fires.radius if self.m.fires is not None else 150.0
        for x, y, _ in ep.fires[s]:
            ax.add_patch(plt.Circle((x, y), fire_r, color=FIRE_COLOR, alpha=0.4, lw=0, zorder=3))
        for name, x, y in self.labels:
            ax.text(x, y, name, color="#c3c9d1", fontsize=4.5 if compact else 6, ha="center", va="center", alpha=0.75)

        friend = FRIEND[ep.friendly]
        for f in range(len(ep.obs_ids)):
            x, y, _ = ep.obs_xyz[s, f]
            if not np.isfinite(x):
                continue
            if ep.obs_alive[s, f]:
                yaw = np.radians(ep.obs_yaw[s, f])
                ax.plot([x, x + 220 * np.cos(yaw)], [y, y + 220 * np.sin(yaw)], color=friend, lw=1, alpha=0.8)
                ax.scatter(x, y, s=34, color=friend, edgecolor="white", linewidth=0.6, zorder=5)
            else:
                ax.scatter(x, y, s=24, color="#6b7280", marker="x", zorder=4)

        for e in range(len(ep.enemy_ids)):
            if not ep.enemy_alive[s, e]:
                continue
            if ep.enemy_seen[s, e]:
                x, y = g.node_xy[ep.enemy_seen_node[s, e]]
                ax.scatter(x, y, s=46, color=ENEMY, edgecolor="white", linewidth=0.7, zorder=6)
            elif self.last_seen[s][e] is not None:
                s0, node = self.last_seen[s][e]
                x, y = g.node_xy[node]
                ax.scatter(x, y, s=80, facecolor="none", edgecolor="white", linewidth=2.2, zorder=6)
                ax.scatter(x, y, s=80, facecolor="none", edgecolor=ENEMY, linewidth=1.2, zorder=6)
                ax.text(x + 70, y + 70, f"{ep.t_rel[s] - ep.t_rel[s0]:.0f}s", color="white", fontsize=6, zorder=6)
            if self.show_truth and not ep.enemy_seen[s, e]:
                x, y, _ = ep.enemy_xyz[s, e]
                ax.scatter(x, y, s=30, color="#7CFC00", marker="x", linewidth=1.3, zorder=7)

        n_f, n_e = int(ep.obs_alive[s].sum()), int(ep.enemy_alive[s].sum())
        bomb = f"  bomb on {'AB'[ep.phase[s] - 1]}" if ep.phase[s] else ""
        ax.set_title(f"{ep.t_rel[s]:5.1f}s   {ep.friendly.upper()} {n_f}v{n_e}{bomb}", color="white", fontsize=9)
        ax.set_xlim(self.extent[0] + 200, self.extent[1] - 200)
        ax.set_ylim(self.extent[2] + 200, self.extent[3] - 200)
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)

    def legend_text(self) -> str:
        truth = "   green x = true enemy position" if self.show_truth else ""
        return (
            f"{FRIEND_NAME[self.ep.friendly]} = your team + view direction   red dot = enemy on radar   "
            f"ring = last seen (age){truth}   grey = smoke   orange = molotov   "
            f"heat = most likely {HDR_MASS:.0%} of hidden-enemy probability"
        )


FRIEND_NAME = {"ct": "blue", "t": "yellow"}


def _figure_to_image(fig: plt.Figure) -> Image.Image:
    fig.canvas.draw()
    return Image.fromarray(np.asarray(fig.canvas.buffer_rgba())[..., :3])


def render_gif(r: RoundRenderer, path: Path, every: int = 2, fps: int = 8) -> None:
    frames = []
    for s in range(0, r.ep.n_steps, every):
        fig, ax = plt.subplots(figsize=(6.2, 5.8), dpi=100, facecolor=BG)
        r.draw(ax, s)
        fig.text(0.5, 0.015, r.legend_text(), color="#9aa3ad", fontsize=5.5, ha="center")
        fig.subplots_adjust(left=0.01, right=0.99, top=0.94, bottom=0.04)
        frames.append(_figure_to_image(fig).convert("P", palette=Image.ADAPTIVE, colors=128))
        plt.close(fig)
    frames[0].save(path, save_all=True, append_images=frames[1:], duration=int(1000 / fps), loop=0, optimize=True)


def key_steps(ep: Episode, n: int) -> list[int]:
    """Steps spread over the part of the round after the first enemy has gone off the radar."""
    seen_any = np.flatnonzero(ep.enemy_seen.any(axis=1))
    start = int(seen_any[0]) + 4 if len(seen_any) else 0
    return np.linspace(min(start, ep.n_steps - 1), ep.n_steps - 1, n).astype(int).tolist()


def render_sheet(r: RoundRenderer, path: Path, steps: list[int], title: str) -> None:
    cols = 3
    rows = int(np.ceil(len(steps) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(4.2 * cols, 4.0 * rows), dpi=110, facecolor=BG)
    for ax, s in zip(np.atleast_1d(axes).ravel(), steps):
        r.draw(ax, s, compact=True)
    for ax in np.atleast_1d(axes).ravel()[len(steps) :]:
        ax.axis("off")
    fig.suptitle(title, color="white", fontsize=11)
    fig.text(0.5, 0.01, r.legend_text(), color="#9aa3ad", fontsize=7, ha="center")
    fig.subplots_adjust(left=0.01, right=0.99, top=0.92, bottom=0.04, wspace=0.03, hspace=0.12)
    fig.savefig(path, facecolor=BG)
    plt.close(fig)


def list_rounds(models: Models, demo_id: str) -> None:
    ref = next(r for r in list_demos() if r.demo_id == demo_id)
    for side in ("ct", "t"):
        for ep in episodes(models.grid, ref, side):
            n_f = ep.obs_alive.sum(axis=1)
            clutch = np.flatnonzero(n_f == 1)
            note = f"1v{int(ep.enemy_alive[clutch[0]].sum())} from {ep.t_rel[clutch[0]]:.0f}s" if len(clutch) else ""
            print(f"round {ep.round_num:2d}  {side}  {ep.t_rel[-1]:5.0f}s  sightings={int(ep.enemy_seen.sum()):4d}  {note}")


def main() -> None:
    ap = argparse.ArgumentParser(description="Render a round with the enemy-position heatmap.")
    ap.add_argument("demo_id", nargs="?")
    ap.add_argument("round", nargs="?", type=int)
    ap.add_argument("--list", metavar="DEMO_ID", help="list rounds of a demo and exit")
    ap.add_argument("--side", default="ct", choices=["ct", "t"], help="whose information to use")
    ap.add_argument("--model", default="ens", choices=[c.name for c in ALL_CONFIGS])
    ap.add_argument("--truth", action="store_true", help="also mark where enemies really were")
    ap.add_argument("--train-sources", nargs="+", default=["xego"])
    ap.add_argument("--panels", type=int, default=6)
    ap.add_argument("--out", type=Path, default=OUTPUT_DIR / "review")
    args = ap.parse_args()

    models = load_models(model_dir(args.train_sources))
    if args.list:
        list_rounds(models, args.list)
        return
    if args.demo_id is None or args.round is None:
        ap.error("give a demo id and a round number (or --list DEMO_ID)")
    ref = next(r for r in list_demos() if r.demo_id == args.demo_id)
    ep = next(e for e in episodes(models.grid, ref, args.side) if e.round_num == args.round)
    renderer = RoundRenderer(ep, models, round_beliefs(ep, models, args.model), args.truth)

    args.out.mkdir(parents=True, exist_ok=True)
    stem = f"{args.demo_id[:18]}_r{args.round}_{args.side}_{args.model}"
    title = f"{args.demo_id[:18]}  round {args.round}  {args.side.upper()} view  ({args.model})"
    render_sheet(renderer, args.out / f"{stem}.png", key_steps(ep, args.panels), title)
    render_gif(renderer, args.out / f"{stem}.gif")
    print(f"Wrote {args.out / stem}.png and .gif")


if __name__ == "__main__":
    main()
