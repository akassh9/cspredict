"""What one team knows during a round, sampled at every filter step.

For a friendly side and a round this collects, per step: friendly positions and view angles
(the observers), which enemies are alive (public via the scoreboard), which enemies were on
the radar during the step and where, active smokes, and blinded observers. Enemy true
positions are kept for scoring only; the filters never read them.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass

import numpy as np
import polars as pl

from cspredict.config import SAMPLE_EVERY, STEP_TICKS, TICKRATE
from cspredict.dataset import DemoRef
from cspredict.grid import NavGrid

BUY_EDGES = (1500.0, 3800.0)  # team mean equipment value when freeze time ends


def buy_type(mean_equip: float) -> int:
    """0 eco (incl. pistol rounds), 1 force / half buy, 2 full buy."""
    return int(np.searchsorted(BUY_EDGES, mean_equip, side="right"))


@dataclass
class Episode:
    demo_id: str
    round_num: int
    friendly: str  # "ct" or "t"
    enemy: str
    ticks: np.ndarray  # (S,) tick of each step
    t_rel: np.ndarray  # (S,) seconds since freeze time ended
    phase: np.ndarray  # (S,) 0 bomb not planted, 1 planted on A, 2 planted on B
    obs_ids: np.ndarray  # (F,) friendly steamids
    obs_xyz: np.ndarray  # (S, F, 3)
    obs_yaw: np.ndarray  # (S, F)
    obs_pitch: np.ndarray  # (S, F)
    obs_alive: np.ndarray  # (S, F)
    obs_blind: np.ndarray  # (S, F) flashed hard enough to see nothing
    enemy_ids: np.ndarray  # (E,)
    enemy_names: list[str]
    enemy_buy: int  # 0 eco, 1 force, 2 full (see buy_type)
    enemy_alive: np.ndarray  # (S, E)
    enemy_seen: np.ndarray  # (S, E) on the friendly radar at some sample within the step
    enemy_seen_node: np.ndarray  # (S, E) node where last seen within the step, -1 if unseen
    enemy_seen_xyz: np.ndarray  # (S, E, 3) position where last seen within the step, NaN if unseen
    enemy_node: np.ndarray  # (S, E) true node (scoring only)
    enemy_xyz: np.ndarray  # (S, E, 3) true position (scoring and drawing only)
    smokes: list[np.ndarray]  # per step, (k, 2) centres of active smokes
    kills: list[list[tuple[int, np.ndarray, bool]]]  # per step: (enemy index, victim xyz, through smoke/wall)

    @property
    def n_steps(self) -> int:
        return len(self.ticks)

    @property
    def planted(self) -> np.ndarray:
        return self.phase > 0

    def observers(self, s: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Positions, yaw and pitch of friendly players able to spot at step s."""
        m = self.obs_alive[s] & ~self.obs_blind[s]
        return self.obs_xyz[s, m], self.obs_yaw[s, m], self.obs_pitch[s, m]


def _dense(rt: pl.DataFrame, ids: np.ndarray, tick_index: dict[int, int], cols: list[str]) -> dict[str, np.ndarray]:
    """(n_ticks, n_players) arrays for the given columns; missing samples are dead/NaN."""
    n_t, n_p = len(tick_index), len(ids)
    pid = {int(s): i for i, s in enumerate(ids)}
    sub = rt.filter(pl.col("steamid").is_in(ids.tolist()))
    ti = np.array([tick_index[t] for t in sub["tick"].to_list()])
    pi = np.array([pid[int(s)] for s in sub["steamid"].to_list()])
    out = {}
    for c in cols:
        v = sub[c].to_numpy()
        arr = np.zeros((n_t, n_p), dtype=bool) if v.dtype == bool else np.full((n_t, n_p), np.nan, dtype=np.float32)
        arr[ti, pi] = v
        out[c] = arr
    return out


def _phase(step_ticks: np.ndarray, plant: int | None, site: str | None) -> np.ndarray:
    """Per step: 0 before the plant, then 1 (bomb on A) or 2 (bomb on B). The site is known
    to both teams once the bomb is down."""
    if plant is None:
        return np.zeros(len(step_ticks), dtype=np.int8)
    return np.where(step_ticks >= plant, 2 if site == "bombsite_b" else 1, 0).astype(np.int8)


def episodes(grid: NavGrid, ref: DemoRef, friendly: str) -> Iterator[Episode]:
    """One Episode per round, seen from the friendly side ("ct" or "t")."""
    enemy = "t" if friendly == "ct" else "ct"
    ticks = ref.table("ticks")
    rounds = ref.table("rounds")
    smokes = ref.table("smokes")
    blinds = ref.table("blinds")
    kills = ref.table("kills")

    for r in rounds.iter_rows(named=True):
        rt = ticks.filter(pl.col("round_num") == r["round_num"])
        if rt.height == 0:
            continue
        all_ticks = np.sort(rt["tick"].unique().to_numpy())
        tick_index = {int(t): i for i, t in enumerate(all_ticks)}
        step_ticks = all_ticks[all_ticks % STEP_TICKS == 0]
        if len(step_ticks) < 2:
            continue
        # Only players alive at some point: demos also list disconnected players and substitutes.
        playing = rt.filter(pl.col("is_alive"))
        obs_ids = np.sort(playing.filter(pl.col("side") == friendly)["steamid"].unique().to_numpy())
        enemy_ids = np.sort(playing.filter(pl.col("side") == enemy)["steamid"].unique().to_numpy())
        if len(obs_ids) == 0 or len(enemy_ids) == 0:
            continue

        f = _dense(rt, obs_ids, tick_index, ["X", "Y", "Z", "yaw", "pitch", "is_alive"])
        e = _dense(rt, enemy_ids, tick_index, ["X", "Y", "Z", "is_alive", "spotted"])
        exyz = np.stack([e["X"], e["Y"], e["Z"]], axis=-1)
        enode = np.full(e["X"].shape, -1, dtype=np.int32)
        ok = np.isfinite(exyz).all(axis=-1)
        enode[ok] = grid.locate(exyz[ok, 0], exyz[ok, 1], exyz[ok, 2])

        si = np.array([tick_index[int(t)] for t in step_ticks])
        prev = np.maximum(si - (STEP_TICKS // SAMPLE_EVERY - 1), 0)  # the other sample inside the step
        seen_now, seen_prev = e["spotted"][si] & e["is_alive"][si], e["spotted"][prev] & e["is_alive"][prev]
        seen_node = np.where(seen_now, enode[si], np.where(seen_prev, enode[prev], -1))
        seen_xyz = np.where(seen_now[..., None], exyz[si], np.where(seen_prev[..., None], exyz[prev], np.nan))

        blind = np.zeros((len(si), len(obs_ids)), dtype=bool)
        for b in blinds.filter(pl.col("round_num") == r["round_num"]).iter_rows(named=True):
            if (b.get("blind_duration") or 0) > 0.3 and b.get("user_steamid"):
                hit = np.flatnonzero(obs_ids == np.uint64(int(b["user_steamid"])))
                if len(hit):
                    until = b["tick"] + b["blind_duration"] * TICKRATE
                    blind[(step_ticks >= b["tick"]) & (step_ticks <= until), hit[0]] = True

        rs = smokes.filter(pl.col("round_num") == r["round_num"])
        sx, sy = rs["X"].to_numpy(), rs["Y"].to_numpy()
        s0, s1 = rs["start_tick"].to_numpy(), rs["end_tick"].to_numpy()
        step_smokes = [np.column_stack([sx, sy])[(s0 <= t) & (t < s1)] for t in step_ticks]

        # The kill feed names every killer, and where the teammate died is known.
        step_kills: list[list[tuple[int, np.ndarray, bool]]] = [[] for _ in step_ticks]
        enemy_index = {int(s): i for i, s in enumerate(enemy_ids)}
        for k in kills.filter(
            (pl.col("round_num") == r["round_num"]) & (pl.col("attacker_side") == enemy) & (pl.col("victim_side") == friendly)
        ).iter_rows(named=True):
            killer = enemy_index.get(int(k["attacker_steamid"])) if k["attacker_steamid"] else None
            step = int(np.searchsorted(step_ticks, k["tick"]))  # first step at or after the kill
            if killer is not None and step < len(step_ticks):
                victim = np.array([k["victim_X"], k["victim_Y"], k["victim_Z"]], dtype=np.float64)
                step_kills[step].append((killer, victim, bool(k["thrusmoke"]) or bool(k["penetrated"])))

        first = rt.filter((pl.col("tick") == all_ticks[0]) & (pl.col("side") == enemy))
        enemy_buy = buy_type(float(first["equip"].mean() or 0.0))

        plant = r["bomb_plant"]
        yield Episode(
            demo_id=ref.demo_id,
            round_num=int(r["round_num"]),
            friendly=friendly,
            enemy=enemy,
            ticks=step_ticks,
            t_rel=((step_ticks - r["freeze_end"]) / TICKRATE).astype(np.float32),
            phase=_phase(step_ticks, plant, r["bomb_site"]),
            obs_ids=obs_ids,
            obs_xyz=np.stack([f["X"], f["Y"], f["Z"]], axis=-1)[si],
            obs_yaw=f["yaw"][si],
            obs_pitch=f["pitch"][si],
            obs_alive=f["is_alive"][si],
            obs_blind=blind,
            enemy_ids=enemy_ids,
            enemy_names=[rt.filter(pl.col("steamid") == s)["name"][0] for s in enemy_ids.tolist()],
            enemy_buy=enemy_buy,
            enemy_alive=e["is_alive"][si],
            enemy_seen=seen_node >= 0,
            enemy_seen_node=seen_node.astype(np.int32),
            enemy_seen_xyz=seen_xyz.astype(np.float32),
            enemy_node=enode[si],
            enemy_xyz=exyz[si],
            smokes=step_smokes,
            kills=step_kills,
        )
