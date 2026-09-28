"""Describe one moment of a round in words, using only what the friendly team knew.

Used to benchmark text-based models (typesafe_bench.py) against the belief filters. Places
are named by Mirage's callouts and times are rounded to whole seconds. Enemy positions only
appear where the team actually saw them: radar sightings and the kill feed.
"""

from __future__ import annotations

import re

import numpy as np

from cspredict.build import Models
from cspredict.config import STEP_TICKS, TICKRATE
from cspredict.grid import NavGrid
from cspredict.infostate import Episode
from cspredict.motion import N_TIME_BINS, TIME_BIN_S, context_key

BUY_NAMES = ("eco (little or no equipment)", "force buy (partial equipment)", "full buy")
C4_TIMER_S = 40
ROUND_TIME_S = 115
SEEN_CALLOUT = 0.25  # share of a callout the team would spot an enemy in, to list it as "watched"
LOOK_TARGET = 0.15  # minimum spot probability for a callout to count as what a player looks at
KILL_FEED_S = 30  # how far back the kill feed is listed
MOVING = 60.0  # units per second; slower sightings are described as holding still
STEP_S = STEP_TICKS / TICKRATE

_READABLE = {
    "BackAlley": "Back Alley", "BombsiteA": "Bombsite A", "BombsiteB": "Bombsite B", "CTSpawn": "CT Spawn",
    "PalaceAlley": "Palace Alley", "PalaceInterior": "Palace Interior", "SideAlley": "Side Alley",
    "SnipersNest": "Snipers Nest", "TopofMid": "Top of Mid", "TRamp": "T Ramp", "TSpawn": "T Spawn",
}  # fmt: skip


def callout_name(place: str) -> str:
    """Game place name -> readable callout ('TopofMid' -> 'Top of Mid')."""
    return _READABLE.get(place, re.sub(r"(?<=[a-z])(?=[A-Z])", " ", place))


def callout_neighbours(grid: NavGrid) -> dict[str, list[str]]:
    """Readable callout -> callouts it directly connects to (walkable edges between them)."""
    adj = grid.adjacency.tocoo()
    a, b = grid.node_place[adj.row], grid.node_place[adj.col]
    links: dict[str, dict[str, int]] = {}
    for pa, pb in zip(a.tolist(), b.tolist()):
        if pa != pb and "Unknown" not in (pa, pb):
            links.setdefault(pa, {}).setdefault(pb, 0)
            links[pa][pb] += 1
    return {
        callout_name(p): sorted(callout_name(q) for q, n in nbrs.items() if n >= 3)
        for p, nbrs in links.items()
    }


def callouts(grid: NavGrid) -> list[str]:
    """The readable callouts a model may answer with, in grid.places order."""
    return [callout_name(p) for p in grid.places]


RUN_SPEED = 250.0  # units per second with a knife out; rifles are about 10% slower
WALK_SPEED = 120.0  # shift-walk; the median moving speed of the last player alive in pro clutches is ~117


def callout_run_times(grid: NavGrid) -> tuple[list[str], np.ndarray]:
    """Readable callouts and a (K, K) matrix of seconds to run from each callout's busiest spot
    to the nearest point of another callout."""
    names, rows = [], []
    for k, place in enumerate(grid.places):
        nodes = np.flatnonzero(grid.node_place_idx == k)
        if len(nodes) == 0 or place == "Unknown":
            continue
        names.append(callout_name(place))
        rows.append(nodes)
    centres = [nodes[np.argmax(grid.node_count[nodes])] for nodes in rows]
    d = grid.dist[centres]
    seconds = np.stack([d[:, nodes].min(axis=1) for nodes in rows], axis=1) / RUN_SPEED
    return names, seconds


def add_reachable(state: dict, names: list[str], seconds: np.ndarray, limit: int = 12, walking: bool = False) -> dict:
    """Add, for each off-radar enemy, the callouts they could have run to since last seen.
    With walking=True each entry also gives the time at a silent shift-walk, and the list is
    ordered by it."""
    index = {n: i for i, n in enumerate(names)}
    slow = RUN_SPEED / WALK_SPEED
    for enemy in state["enemies"]:
        if enemy.get("last_seen_at") not in index:
            continue
        row = seconds[index[enemy["last_seen_at"]]]
        reach = sorted((t, n) for n, t in zip(names, row) if t <= enemy["seconds_since_seen"] + 1)
        if walking:
            listed = [{"callout": n, "walking_s": round(t * slow), "running_s": round(t)} for t, n in reach[:limit]]
        else:
            listed = [{"callout": n, "run_time_s": round(t)} for t, n in reach[:limit]]
        if len(reach) > limit:
            listed.append(f"and {len(reach) - limit} more callouts")
        enemy["could_have_reached"] = listed
    return state


def _place_of(grid: NavGrid, x: float, y: float, z: float) -> str:
    return callout_name(grid.node_place[grid.locate(np.array([x]), np.array([y]), np.array([z]))[0]])


def _place_means(grid: NavGrid, values: np.ndarray) -> dict[str, float]:
    """Traffic-weighted mean of a per-node value within each callout."""
    w = grid.node_count.astype(np.float64)
    idx = grid.node_place_idx
    totals = np.bincount(idx, weights=values * w, minlength=len(grid.places))
    norm = np.maximum(np.bincount(idx, weights=w, minlength=len(grid.places)), 1e-9)
    return {callout_name(p): float(v) for p, v in zip(grid.places, totals / norm) if p != "Unknown"}


def watched_by_step(grid: NavGrid, unseen: np.ndarray) -> np.ndarray:
    """(S, C) per step, the traffic-weighted chance that an enemy standing in each callout (in
    grid.places order) would have shown up on the team's radar."""
    w = grid.node_count.astype(np.float64)
    onehot = np.zeros((grid.n, len(grid.places)))
    onehot[np.arange(grid.n), grid.node_place_idx] = w
    return ((1.0 - unseen) @ onehot) / np.maximum(onehot.sum(axis=0), 1e-9)


def enemy_evidence(ep: Episode, s: int, e: int, models: Models, watched: np.ndarray) -> dict:
    """Facts about one off-radar enemy that the team could have worked out, computed in code and
    keyed by readable callout: how likely the team was to spot someone there now and on average
    since the enemy was last seen, and where that side's players usually are at this point of a
    round (share of players per callout in the training demos). Plus the enemy's recent sightings.
    `watched` is watched_by_step for the episode."""
    grid = models.grid
    names = [callout_name(p) for p in grid.places]
    keep = [i for i, p in enumerate(grid.places) if p != "Unknown"]
    seen_steps = np.flatnonzero(ep.enemy_seen[: s + 1, e])
    s0 = int(seen_steps[-1])
    since = watched[s0 + 1 : s + 1]
    mean_since = since.mean(axis=0) if len(since) else watched[s]
    tbin = min(int(ep.t_rel[s] // TIME_BIN_S), N_TIME_BINS - 1)
    occ = models.motion.occupancy[context_key(ep.enemy, int(ep.phase[s]))][tbin].astype(np.float64)
    usual = np.bincount(grid.node_place_idx, occ, len(grid.places))

    # Recent sightings: each unbroken stretch on the radar, newest first.
    runs = np.split(seen_steps, np.flatnonzero(np.diff(seen_steps) > 1) + 1)
    history = []
    for run in runs[::-1][:3]:
        end = int(run[-1])
        history.append({
            "at": _place_of(grid, *ep.enemy_seen_xyz[end, e]),
            "seconds_ago": round(float(ep.t_rel[s] - ep.t_rel[end])),
            "seen_for_s": round(float(ep.t_rel[end] - ep.t_rel[int(run[0])]) + STEP_S, 1),
        })  # fmt: skip
    return {
        "last_seen_at": history[0]["at"],
        "seconds_since_seen": round(float(ep.t_rel[s] - ep.t_rel[s0])),
        "spot_chance_now": {names[i]: float(watched[s, i]) for i in keep},
        "spot_chance_since_seen": {names[i]: float(mean_since[i]) for i in keep},
        "usual_share": {names[i]: float(usual[i]) for i in keep},
        "history": history,
    }


def enemy_labels(ep: Episode) -> list[str]:
    side = ep.enemy.upper()
    return [f"{side}{i + 1}" for i in range(len(ep.enemy_ids))]


def describe(ep: Episode, s: int, models: Models, unseen_s: np.ndarray) -> dict:
    """JSON-ready description of step s from the friendly side's point of view. `unseen_s` is
    the step's per-node "enemy would stay off our radar" likelihood (filters.gather_evidence)."""
    grid = models.grid
    t = float(ep.t_rel[s])
    friend = ep.friendly.upper()
    labels = enemy_labels(ep)

    if ep.phase[s]:
        plant_s = int(np.flatnonzero(ep.phase > 0)[0])
        since_plant = t - float(ep.t_rel[plant_s])
        bomb = f"planted on {'AB'[ep.phase[s] - 1]} site {since_plant:.0f} s ago (explodes in {C4_TIMER_S - since_plant:.0f} s)"
    else:
        bomb = f"not planted ({max(ROUND_TIME_S - t, 0):.0f} s of round time left)"

    teammates = []
    for i, f in enumerate(np.flatnonzero(ep.obs_alive[s])):
        x, y, z = ep.obs_xyz[s, f]
        at = _place_of(grid, x, y, z)
        entry = {"id": f"{friend}{i + 1}", "at": at}
        if ep.obs_blind[s, f]:
            entry["status"] = "flashed (blind)"
        else:
            sees = models.spot.spot_prob(
                grid, ep.obs_xyz[s, f][None], ep.obs_yaw[s, f : f + 1], ep.obs_pitch[s, f : f + 1], ep.smokes[s]
            )[0]
            means = _place_means(grid, sees)
            means.pop(at, None)
            target = max(means, key=means.get) if means else None
            entry["looking_toward"] = target if target and means[target] >= LOOK_TARGET else "a nearby wall or corner"
        teammates.append(entry)
    watched = _place_means(grid, 1.0 - unseen_s)
    watched_now = sorted(p for p, v in watched.items() if v >= SEEN_CALLOUT)

    enemies = []
    for e, label in enumerate(labels):
        if not ep.enemy_alive[s, e]:
            enemies.append({"id": label, "status": "dead"})
            continue
        if ep.enemy_seen[s, e]:
            x, y, z = ep.enemy_seen_xyz[s, e]
            enemies.append({"id": label, "status": "on your radar now", "at": _place_of(grid, x, y, z)})
            continue
        seen_steps = np.flatnonzero(ep.enemy_seen[: s + 1, e])
        if len(seen_steps) == 0:
            enemies.append({"id": label, "status": "not seen yet this round"})
            continue
        s0 = int(seen_steps[-1])
        x, y, z = ep.enemy_seen_xyz[s0, e]
        entry = {"id": label, "status": "off your radar", "last_seen_at": _place_of(grid, x, y, z),
                 "seconds_since_seen": round(t - float(ep.t_rel[s0]))}  # fmt: skip
        if s0 >= 1 and ep.enemy_seen[s0 - 1, e]:
            v = ep.enemy_seen_xyz[s0, e, :2] - ep.enemy_seen_xyz[s0 - 1, e, :2]
            speed = float(np.hypot(*v)) / STEP_S
            if speed < MOVING:
                entry["when_last_seen"] = "holding still"
            else:
                ahead = ep.enemy_seen_xyz[s0, e, :2] + v / np.hypot(*v) * 400.0
                toward = _place_of(grid, ahead[0], ahead[1], z)
                last = entry["last_seen_at"]
                entry["when_last_seen"] = f"moving toward {toward}" if toward != last else f"moving within {last}"
        enemies.append(entry)

    kill_feed = []
    for k in range(s + 1):
        ago = t - float(ep.t_rel[k])
        if ago > KILL_FEED_S:
            continue
        for e, victim, obstructed in ep.kills[k]:
            item = {"seconds_ago": round(ago), "killer": labels[e], "victim_was_at": _place_of(grid, *victim)}
            if obstructed:
                item["note"] = "through smoke or a wall"
            kill_feed.append(item)

    return {
        "game": "Counter-Strike 2 on Mirage. One team's view of a round; only what that team knew.",
        "your_team": friend,
        "seconds_into_round": round(t),
        "bomb": bomb,
        "players_alive": {"your_team": int(ep.obs_alive[s].sum()), "enemies": int(ep.enemy_alive[s].sum())},
        "enemy_team_buy": BUY_NAMES[ep.enemy_buy_true],  # true equipment values, as in the benchmark runs (the filters estimate it)
        "your_teammates": teammates,
        "callouts_your_team_is_watching": watched_now or ["none"],
        "enemies": enemies,
        "recent_kill_feed": kill_feed or ["no recent kills by enemies"],
        "smokes_active_at": sorted({_place_of(grid, x, y, 100.0) for x, y in ep.smokes[s]}) or ["none"],
        "molotovs_burning_at": sorted({_place_of(grid, x, y, z) for x, y, z in ep.fires[s]}) or ["none"],
    }
