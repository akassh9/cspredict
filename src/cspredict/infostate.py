"""What one team knows during a round, sampled at every filter step.

For a friendly side and a round this collects, per step: friendly positions and view angles
(the observers), which enemies are alive (public via the scoreboard), which enemies were on
the radar during the step and where, active smokes, and blinded observers; and per round the
enemy team's round history (the scoreboard: which rounds it won and lost in this half). Enemy
true positions and the enemies' true buy (from equipment values nobody on the other team can
see) are kept for scoring and ablations only; the filters never read them.
"""

from __future__ import annotations

import re
from collections.abc import Iterator
from dataclasses import dataclass

import numpy as np
import polars as pl

from cspredict.config import SAMPLE_EVERY, STEP_TICKS, TICKRATE
from cspredict.dataset import DemoRef
from cspredict.fires import active_fires
from cspredict.grid import NavGrid

BUY_EDGES = (1500.0, 3800.0)  # team mean equipment value when freeze time ends
# Spotting rates through smokes in the training demos: a smoke blocks sight about a second after
# it bursts and is see-through again about 1.5 s before its expiry event (it thins out first).
SMOKE_BLOOM_S = 1.0
SMOKE_FADE_S = 1.5
# Match format: sides swap after 12 rounds, and overtime is played in halves of 3 rounds.
HALF_ROUNDS = 12
OT_HALF_ROUNDS = 3
MAX_LOSSES = 4  # cap on the loss-bonus count (BuyHistory.losses)
KNIFE = re.compile(r"knife|bayonet")

# The weapon in an enemy's hands is visible whenever they are seen (and called out: "AWP mid").
# Shotguns and machine guns are grouped with SMGs: together they are 0.1% of sightings.
WEAPON_CLASSES = ("sniper", "rifle", "smg", "pistol", "other")  # other: knife, grenade or bomb in hand
_WEAPONS = (
    ("AWP", "SSG 08", "SCAR-20", "G3SG1"),
    ("AK-47", "M4A1-S", "M4A4", "Galil AR", "FAMAS", "AUG", "SG 553"),
    ("MP9", "MAC-10", "MP7", "MP5-SD", "UMP-45", "P90", "PP-Bizon", "MAG-7", "XM1014", "Nova", "Sawed-Off", "M249", "Negev"),
    ("Glock-18", "USP-S", "P2000", "P250", "Five-SeveN", "Tec-9", "CZ75-Auto", "Desert Eagle", "R8 Revolver", "Dual Berettas"),
)
WEAPON_CLASS = {name: k for k, names in enumerate(_WEAPONS) for name in names}
OTHER_WEAPON = len(_WEAPONS)
# The same classes under the names the kill feed uses (the kill feed shows every killer's weapon).
_KILL_FEED_WEAPONS = (
    ("awp", "ssg08", "scar20", "g3sg1"),
    ("ak47", "m4a1_silencer", "m4a1", "galilar", "famas", "aug", "sg556"),
    ("mp9", "mac10", "mp7", "mp5sd", "ump45", "p90", "bizon", "mag7", "xm1014", "nova", "sawedoff", "m249", "negev"),
    ("glock", "usp_silencer", "hkp2000", "p250", "fiveseven", "tec9", "cz75a", "deagle", "revolver", "elite"),
)
KILL_WEAPON_CLASS = {name: k for k, names in enumerate(_KILL_FEED_WEAPONS) for name in names}


def buy_type(mean_equip: float) -> int:
    """0 eco (incl. pistol rounds), 1 force / half buy, 2 full buy."""
    return int(np.searchsorted(BUY_EDGES, mean_equip, side="right"))


def weapon_class_expr(col: str = "weapon") -> pl.Expr:
    """Index into WEAPON_CLASSES of the active weapon's name."""
    return pl.col(col).replace_strict(WEAPON_CLASS, default=OTHER_WEAPON, return_dtype=pl.Int8)


@dataclass(frozen=True)
class BuyHistory:
    """What the scoreboard tells both teams about one team's economy before a round."""

    half_round: int  # round of the half: 1 is the pistol round (or the first round of an overtime half)
    overtime: bool
    won_last: bool | None  # won the previous round of this half; None in its first round
    won_pistol: bool | None  # won this half's pistol round; None in that round and in overtime
    losses: int  # the loss-bonus count: 0 when the half starts, +1 per round lost (at most MAX_LOSSES), -1 per round won


def half_start(match_round: int) -> int:
    """First round of the half (regulation or overtime) that holds this round."""
    regulation = 2 * HALF_ROUNDS
    if match_round <= regulation:
        return 1 + HALF_ROUNDS * ((match_round - 1) // HALF_ROUNDS)
    return regulation + 1 + OT_HALF_ROUNDS * ((match_round - regulation - 1) // OT_HALF_ROUNDS)


def match_rounds(rounds: pl.DataFrame, kills: pl.DataFrame) -> dict[int, int]:
    """round_num -> the round's number in the match (the scoreboard's). A knife round deciding sides
    (no freeze time recorded, only knife kills) comes before the match and gets no number. A first
    round that is missing its freeze time but has gun kills is a pistol round the demo started
    recording late."""
    rounds = rounds.sort("round_num")
    first = rounds.row(0, named=True) if rounds.height else None
    weapons = kills.filter(pl.col("round_num") == first["round_num"])["weapon"].to_list() if first else []
    knife = first is not None and first["freeze_end"] is None and bool(weapons)
    knife = knife and all(w == "world" or KNIFE.search(w or "") for w in weapons)
    nums = rounds["round_num"].to_list()[1 if knife else 0 :]
    return {int(r): m for m, r in enumerate(nums, start=1)}


def buy_histories(rounds: pl.DataFrame, kills: pl.DataFrame) -> dict[tuple[int, str], BuyHistory]:
    """(round_num, side) -> that side's round history within the half. Money resets at every half,
    and a team keeps its side for the whole half, so the side's earlier wins in the half are the
    team's."""
    number = match_rounds(rounds, kills)
    winner = {number[r]: w for r, w in rounds.select("round_num", "winner").iter_rows() if r in number}
    out = {}
    for rn, m in number.items():
        start = half_start(m)
        for side in ("t", "ct"):
            won = [winner.get(j) == side for j in range(start, m)]
            losses = 0
            for w in won:
                losses = max(losses - 1, 0) if w else min(losses + 1, MAX_LOSSES)
            out[(rn, side)] = BuyHistory(
                half_round=m - start + 1, overtime=m > 2 * HALF_ROUNDS, won_last=won[-1] if won else None,
                won_pistol=won[0] if won and m <= 2 * HALF_ROUNDS else None, losses=losses,
            )  # fmt: skip
    return out


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
    enemy_history: BuyHistory | None  # the enemy team's round history (economy.py estimates its buy from it)
    enemy_buy_true: int  # 0 eco, 1 force, 2 full from the enemies' equipment values, which the team cannot see (training and ablations only)
    enemy_alive: np.ndarray  # (S, E)
    enemy_seen: np.ndarray  # (S, E) on the friendly radar at some sample within the step
    enemy_seen_node: np.ndarray  # (S, E) node where last seen within the step, -1 if unseen
    enemy_seen_xyz: np.ndarray  # (S, E, 3) position where last seen within the step, NaN if unseen
    enemy_seen_weapon: np.ndarray  # (S, E) WEAPON_CLASSES index of the weapon in hand when seen, -1 if unseen
    enemy_kill_weapon: np.ndarray  # (S, E) WEAPON_CLASSES index of the weapon shown in the kill feed for a kill, -1 if none
    enemy_node: np.ndarray  # (S, E) true node (scoring only)
    enemy_xyz: np.ndarray  # (S, E, 3) true position (scoring and drawing only)
    smokes: list[np.ndarray]  # per step, (k, 2) centres of active smokes
    fires: list[np.ndarray]  # per step, (k, 3) centres of burning molotovs / incendiaries (known to both teams)
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
    infernos = ref.table("infernos")
    blinds = ref.table("blinds")
    kills = ref.table("kills")
    histories = buy_histories(rounds, kills)

    ticks = ticks.with_columns(weapon_class_expr().alias("wclass"))
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
        e = _dense(rt, enemy_ids, tick_index, ["X", "Y", "Z", "is_alive", "spotted", "wclass"])
        exyz = np.stack([e["X"], e["Y"], e["Z"]], axis=-1)
        enode = np.full(e["X"].shape, -1, dtype=np.int32)
        ok = np.isfinite(exyz).all(axis=-1)
        enode[ok] = grid.locate(exyz[ok, 0], exyz[ok, 1], exyz[ok, 2])

        si = np.array([tick_index[int(t)] for t in step_ticks])
        prev = np.maximum(si - (STEP_TICKS // SAMPLE_EVERY - 1), 0)  # the other sample inside the step
        seen_now, seen_prev = e["spotted"][si] & e["is_alive"][si], e["spotted"][prev] & e["is_alive"][prev]
        seen_node = np.where(seen_now, enode[si], np.where(seen_prev, enode[prev], -1))
        seen_xyz = np.where(seen_now[..., None], exyz[si], np.where(seen_prev[..., None], exyz[prev], np.nan))
        wclass = np.nan_to_num(e["wclass"], nan=OTHER_WEAPON).astype(np.int8)
        seen_weapon = np.where(seen_now, wclass[si], np.where(seen_prev, wclass[prev], -1))

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
        s0 = s0 + SMOKE_BLOOM_S * TICKRATE
        s1 = s1 - SMOKE_FADE_S * TICKRATE
        step_smokes = [np.column_stack([sx, sy])[(s0 <= t) & (t < s1)] for t in step_ticks]
        step_fires = active_fires(infernos.filter(pl.col("round_num") == r["round_num"]), step_ticks)

        # The kill feed names every killer and the weapon, and where the teammate died is known.
        step_kills: list[list[tuple[int, np.ndarray, bool]]] = [[] for _ in step_ticks]
        kill_weapon = np.full((len(step_ticks), len(enemy_ids)), -1, dtype=np.int8)
        enemy_index = {int(s): i for i, s in enumerate(enemy_ids)}
        for k in kills.filter(
            (pl.col("round_num") == r["round_num"]) & (pl.col("attacker_side") == enemy) & (pl.col("victim_side") == friendly)
        ).iter_rows(named=True):
            killer = enemy_index.get(int(k["attacker_steamid"])) if k["attacker_steamid"] else None
            step = int(np.searchsorted(step_ticks, k["tick"]))  # first step at or after the kill
            if killer is not None and step < len(step_ticks):
                victim = np.array([k["victim_X"], k["victim_Y"], k["victim_Z"]], dtype=np.float64)
                step_kills[step].append((killer, victim, bool(k["thrusmoke"]) or bool(k["penetrated"])))
                kill_weapon[step, killer] = KILL_WEAPON_CLASS.get(k["weapon"], OTHER_WEAPON)

        first = rt.filter((pl.col("tick") == all_ticks[0]) & (pl.col("side") == enemy))
        enemy_buy_true = buy_type(float(first["equip"].mean() or 0.0))

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
            enemy_history=histories.get((int(r["round_num"]), enemy)),
            enemy_buy_true=enemy_buy_true,
            enemy_alive=e["is_alive"][si],
            enemy_seen=seen_node >= 0,
            enemy_seen_node=seen_node.astype(np.int32),
            enemy_seen_xyz=seen_xyz.astype(np.float32),
            enemy_seen_weapon=seen_weapon.astype(np.int8),
            enemy_kill_weapon=kill_weapon,
            enemy_node=enode[si],
            enemy_xyz=exyz[si],
            smokes=step_smokes,
            fires=step_fires,
            kills=step_kills,
        )
