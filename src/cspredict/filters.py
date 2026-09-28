"""Belief filters: a probability distribution over grid nodes for every enemy, updated every
filter step (0.25 s) from what the friendly team knows.

    predict:  b <- b @ T                          motion model for the enemy's side, bomb phase
                                                  and time since the enemy was last seen
    update:   on the radar  -> b = delta(node where seen)
              off the radar -> b <- b * L^gamma   L = P(not spotted | enemy in node), from visibility.py
              got a kill    -> b <- b * K         K = P(could hit the victim | killer in node)
    fires:    moves into burning cells are downweighted (the blocked probability stays with the other
              moves from the same cell), and the output belief is weighted by how rarely players
              stand that close to a burning fire (fires.py)

Configurations cover the learned filters and the baselines they are compared against:

    last_seen   stay at the last sighting (no motion, no other evidence)
    prior       where that side usually is at this round time; ignores all evidence
    prior_neg   prior + negative information + kills
    diffuse     lazy random walk + negative information + kills (classic occupancy map, no learning)
    hmm_noneg   learned grid motion + kills, no negative information (ablation)
    hmm         learned grid motion + negative information + kills
    pf_noneg    trajectory-library particles + kills, no negative information (ablation)
    pf          trajectory-library particles + negative information + kills (see particles.py)
    ens         75% pf + 15% hmm + 10% prior_neg: particles give sharp routes, the other two
                keep some mass on every plausible spot so an unexpected route is never ruled out.
                Particles and team snapshots softly prefer recorded rounds with the enemy's buy,
                as the friendly team can estimate it from the round history and the guns it has
                seen (economy.py).
                The grid motion avoids burning fires and the output is weighted by the fire
                occupancy profile (fires.py). Once some enemy has been seen, each enemy's callout
                probabilities are corrected by recorded whole-team snapshots (team.py). Finally
                the callout probabilities are calibrated, by time since the enemy was last seen
                (calibrate.py, fitted on validation).
    ens_nokill  ens without the kill-feed evidence (ablation)

Named ablations (ABLATIONS, scored with evaluate --models) keep earlier versions and rejected
variants runnable: ens_uncal (ens before calibration), ens_indep (before team coordination),
ens_nofire (before fires: the original ensemble), the weapon-matching variants, the team and
fire variants tried on validation, and the buy variants: ens_truebuy (the buy from the enemies'
true equipment values, which the team cannot see: ens_uncal as it was before economy.py),
ens_histbuy (round history only) and ens_nobuy (no buy preference).
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass, field, replace
from functools import lru_cache

import numpy as np
from loguru import logger

from cspredict.calibrate import Calibration
from cspredict.economy import BuyModel
from cspredict.fires import FireModel
from cspredict.grid import NavGrid
from cspredict.infostate import Episode
from cspredict.motion import N_TIME_BINS, TIME_BIN_S, MotionModel, context_key, since_bin
from cspredict.particles import SIDE_ID, TrajectoryLibrary, run_particles
from cspredict.team import TeamLibrary, TeamParams, team_callouts
from cspredict.visibility import SpotModel

FLOOR = 1e-6  # probability mass spread uniformly each step so no node is ever ruled out


@dataclass(frozen=True)
class FilterConfig:
    name: str
    motion: str  # "learned", "random_walk", "none", "prior" or "particles"
    negative: bool  # use negative information (enemy is not on the radar)
    kills: bool = True  # use the kill feed (the killer could see the victim)
    gamma: float = 1.0  # tempering of negative information; 1.0 was best on validation
    mix_hmm: float = 0.0  # weight of the hmm belief mixed into a particle belief
    mix_prior: float = 0.0  # weight of the prior_neg belief mixed into a particle belief
    weapon: float = 1.0  # particles: weight for recorded states with another last-seen weapon class (1 = off)
    weapon_mode: str = "all"  # which weapon classes to compare: "all", "guns" or "awp" (see particles.py)
    hmm_weapon: bool = False  # grid motion also conditioned on the weapon class when last seen
    fire_mask: bool = False  # grid motion: moves into burning cells are downweighted (fires.step)
    fire_wait: bool = False  # particles whose recording walks into fire wait at the edge (ablation)
    fire_weight: bool = False  # particles re-weighted every step by fires.occupancy (ablation)
    fire_occ: bool = False  # output belief weighted by the occupancy profile around burning fires
    team: float = 0.0  # weight of the team-coordination correction of the output (team.py; 0 = off)
    team_params: TeamParams = TeamParams()
    calibrated: bool = False  # apply the stored callout calibration (calibration.json) to the output
    seed: int = 0  # particle random seed (a second seed measures Monte Carlo noise)
    # The enemy buy that particle and team matching prefer: "estimate" (round history and guns seen,
    # economy.py), "history" (round history only), "true" (the enemies' equipment values, which the
    # team cannot see; ablation) or "none".
    buy: str = "estimate"


# The ensemble, one enhancement at a time; earlier stages stay runnable as named ablations.
ENS_NOFIRE = FilterConfig("ens_nofire", "particles", negative=True, mix_hmm=0.15, mix_prior=0.10)
ENS_INDEP = replace(ENS_NOFIRE, name="ens_indep", fire_mask=True, fire_occ=True)  # + fires; enemies independent
ENS_UNCAL = replace(ENS_INDEP, name="ens_uncal", team=1.0, team_params=TeamParams(pseudo=100.0, after_contact=True))
ENS = replace(ENS_UNCAL, name="ens", calibrated=True)  # + calibration.json, fitted on validation

DEFAULT_CONFIGS = (
    FilterConfig("last_seen", "none", negative=False, kills=False),
    FilterConfig("prior", "prior", negative=False, kills=False),
    FilterConfig("prior_neg", "prior", negative=True),
    FilterConfig("diffuse", "random_walk", negative=True),
    FilterConfig("hmm_noneg", "learned", negative=False),
    FilterConfig("hmm", "learned", negative=True),
    FilterConfig("pf_noneg", "particles", negative=False),
    FilterConfig("pf", "particles", negative=True),
    ENS,
    replace(ENS, name="ens_nokill", kills=False),
)
_ENS = ENS_NOFIRE  # the weapon and fire variants were tried on the ensemble before fires
# Variants scored only when named (evaluate --models), for validation ablations.
ABLATIONS: tuple[FilterConfig, ...] = (
    ENS_UNCAL,  # ens before calibration
    ENS_INDEP,  # ens before team coordination: enemies tracked independently
    ENS_NOFIRE,  # ens before fires (the ensemble as it was before these enhancements)
    *(replace(_ENS, name=f"ens_w{w}", weapon=w) for w in (0.7, 0.5, 0.3)),
    FilterConfig("hmm_weapon", "learned", negative=True, hmm_weapon=True),
    replace(_ENS, name="ens_hmmw", hmm_weapon=True),
    replace(_ENS, name="ens_w0.5_hmmw", weapon=0.5, hmm_weapon=True),
    replace(_ENS, name="ens_seed1", seed=1),
    *(replace(_ENS, name=f"ens_awp{w}", weapon=w, weapon_mode="awp") for w in (0.5, 0.3)),
    replace(_ENS, name="ens_guns0.5", weapon=0.5, weapon_mode="guns"),
    replace(_ENS, name="ens_fire_occ", fire_occ=True),
    replace(_ENS, name="ens_fire_wait", fire_mask=True, fire_wait=True, fire_occ=True),
    replace(_ENS, name="ens_fire_weight", fire_mask=True, fire_weight=True, fire_occ=True),
    FilterConfig("hmm_fire", "learned", negative=True, fire_mask=True, fire_occ=True),
    replace(ENS_INDEP, name="ens_seed1_fire", seed=1),
    replace(ENS_INDEP, name="ens_team", team=1.0),
    replace(ENS_INDEP, name="ens_team0.5", team=0.5),
    replace(ENS_INDEP, name="ens_team_b0.5", team=1.0, team_params=TeamParams(beta=0.5)),
    replace(ENS_INDEP, name="ens_team_p100", team=1.0, team_params=TeamParams(pseudo=100.0)),
    replace(ENS_INDEP, name="ens_team_p5", team=1.0, team_params=TeamParams(pseudo=5.0)),
    replace(ENS_INDEP, name="ens_team_alive1", team=1.0, team_params=TeamParams(extra_alive=1.0)),
    replace(ENS_INDEP, name="ens_team_c4", team=1.0, team_params=TeamParams(clock_sigma=4.0)),
    replace(ENS_INDEP, name="ens_team_contact", team=1.0, team_params=TeamParams(after_contact=True)),
    replace(ENS_INDEP, name="ens_team_b0.5_contact", team=1.0, team_params=TeamParams(beta=0.5, after_contact=True)),
    replace(ENS_INDEP, name="ens_team_best", team=1.0, team_params=TeamParams(best_pairing=True)),
    *(replace(ENS_INDEP, name=f"ens_team_p{p}c", team=1.0, team_params=TeamParams(pseudo=float(p), after_contact=True))
      for p in (100, 300, 1000)),
    replace(ENS_INDEP, name="ens_team_p100c_l0.7", team=0.7, team_params=TeamParams(pseudo=100.0, after_contact=True)),
    replace(ENS_INDEP, name="ens_team_p100c_b0.7", team=1.0, team_params=TeamParams(pseudo=100.0, beta=0.7, after_contact=True)),
    replace(ENS_UNCAL, name="ens_truebuy", buy="true"),  # ens_uncal before economy.py
    replace(ENS_UNCAL, name="ens_histbuy", buy="history"),
    replace(ENS_UNCAL, name="ens_nobuy", buy="none"),
    replace(ENS_UNCAL, name="ens_uncal_seed1", seed=1),
    replace(ENS_UNCAL, name="ens_truebuy_seed1", buy="true", seed=1),
)
ALL_CONFIGS = DEFAULT_CONFIGS + ABLATIONS


@dataclass
class Evidence:
    """Everything the friendly team observes besides direct sightings, precomputed per step."""

    unseen: np.ndarray  # (S, N) likelihood of staying off the radar
    kills: dict[int, list[tuple[int, np.ndarray]]] = field(default_factory=dict)  # step -> [(enemy, K)]
    fire_step: dict[int, np.ndarray] = field(default_factory=dict)  # step -> (N,) move multiplier near fires
    fire_occ: dict[int, np.ndarray] = field(default_factory=dict)  # step -> (N,) occupancy likelihood near fires
    buy: dict[str, np.ndarray] = field(default_factory=dict)  # "estimate" / "history" -> (S, 3) P(eco, force, full)


def gather_evidence(
    ep: Episode, grid: NavGrid, spot: SpotModel, fires: FireModel | None = None, buy: BuyModel | None = None
) -> Evidence:
    unseen = np.empty((ep.n_steps, grid.n))
    for s in range(ep.n_steps):
        xyz, yaw, pitch = ep.observers(s)
        unseen[s] = spot.unseen_likelihood(grid, xyz, yaw, pitch, ep.smokes[s])
    kills = {
        s: [(e, spot.shooter_likelihood(grid, victim, obstructed)) for e, victim, obstructed in step_kills]
        for s, step_kills in enumerate(ep.kills)
        if step_kills
    }
    ev = Evidence(unseen=unseen, kills=kills)
    if fires is not None:
        for s, burning in enumerate(ep.fires):
            if len(burning):
                ev.fire_step[s] = fires.node_factors(grid, burning, "step")
                ev.fire_occ[s] = fires.node_factors(grid, burning, "occupancy")
    if buy is not None:
        ev.buy = {"estimate": buy.posterior(ep), "history": buy.posterior(ep, weapons=False)}
    return ev


def predict(b: np.ndarray, t, mask: np.ndarray | None = None) -> np.ndarray:
    """One motion step b @ T. With a per-node mask m (fires) the transitions become
    T_ij m_j / sum_k T_ik m_k: a blocked move's probability goes to the other moves from the same
    cell (wait at the edge, go around) instead of vanishing."""
    if mask is None:
        return np.asarray(b @ t)
    reach = np.asarray(t @ mask).ravel()
    return np.asarray((b / np.maximum(reach, 1e-300)) @ t) * mask


def _fire_output(b: np.ndarray, cfg: FilterConfig, ev: Evidence, s: int) -> np.ndarray:
    """The yielded belief weighted by the fire occupancy profile. Only the output is weighted: the
    fire is one standing condition, not a fresh observation every step."""
    occ = ev.fire_occ.get(s) if cfg.fire_occ else None
    if occ is None:
        return b
    post = b * occ
    return post / post.sum(axis=1, keepdims=True)


def _buy_belief(cfg: FilterConfig, ep: Episode, ev: Evidence) -> np.ndarray | None:
    """(S, 3) P(eco, force, full) of the enemy team that particle and team matching assume at each
    step, or None for no buy preference."""
    if cfg.buy == "none":
        return None
    if cfg.buy == "true":
        return np.tile(np.eye(3)[ep.enemy_buy_true], (ep.n_steps, 1))
    if cfg.buy not in ev.buy:
        _warn_once(f"no buy.json next to the models: {cfg.name} matches without the enemy buy (run `python -m cspredict.build --only buy`)")
        return None
    return ev.buy[cfg.buy]


def _apply_kills(b: np.ndarray, cfg: FilterConfig, ev: Evidence, s: int) -> np.ndarray:
    if cfg.kills:
        for e, like in ev.kills.get(s, []):
            post = b[e] * like
            b[e] = post / max(post.sum(), 1e-300)
    return b


OUTPUT_OPTIONS = ("fire_occ", "team", "team_params", "calibrated")  # options that only reweight the output belief


def _core(cfg: FilterConfig) -> FilterConfig:
    """The config with its output-only options reset: configs sharing a core share one filter run."""
    return replace(cfg, name="", **{k: FilterConfig.__dataclass_fields__[k].default for k in OUTPUT_OPTIONS})


def run_filters(
    ep: Episode,
    grid: NavGrid,
    motion: MotionModel,
    cfgs: tuple[FilterConfig, ...],
    ev: Evidence,
    library: TrajectoryLibrary | None = None,
    teams: TeamLibrary | None = None,
    calibration: Calibration | None = None,
) -> Iterator[tuple[int, dict[str, np.ndarray]]]:
    """Yield (step, {config name: beliefs (E, N)}) for several configs at once. Configs that differ
    only in output options (OUTPUT_OPTIONS) share one run of the filter."""
    groups: dict[FilterConfig, list[FilterConfig]] = {}
    for cfg in cfgs:
        groups.setdefault(_core(cfg), []).append(cfg)
    runs = [(_run_core(ep, grid, motion, core, ev, library), members) for core, members in groups.items()]
    planted = np.flatnonzero(ep.phase > 0)
    post = _Post(ep, grid, ev, teams, float(ep.t_rel[planted[0]]) if len(planted) else None, calibration, _since_seen(ep))
    for steps in zip(*(run for run, _ in runs)):
        s = steps[0][0]
        yield s, {cfg.name: _output(b, cfg, post, s) for (_, b), (_, members) in zip(steps, runs) for cfg in members}


def run_filter(
    ep: Episode,
    grid: NavGrid,
    motion: MotionModel,
    cfg: FilterConfig,
    ev: Evidence,
    library: TrajectoryLibrary | None = None,
    teams: TeamLibrary | None = None,
    calibration: Calibration | None = None,
) -> Iterator[tuple[int, np.ndarray]]:
    """Yield (step, beliefs) with beliefs of shape (E, N) after each step's update."""
    for s, beliefs in run_filters(ep, grid, motion, (cfg,), ev, library, teams, calibration):
        yield s, beliefs[cfg.name]


@dataclass
class _Post:
    """What output processing needs besides the beliefs."""

    ep: Episode
    grid: NavGrid
    ev: Evidence
    teams: TeamLibrary | None
    plant_t: float | None  # round time of the plant, if any
    calibration: Calibration | None = None
    since: np.ndarray | None = None  # (S, E) seconds since each enemy was last on the radar, inf if never


def _since_seen(ep: Episode) -> np.ndarray:
    """(S, E) seconds since each enemy was last on the radar, at or before each step (inf if never)."""
    out = np.full(ep.enemy_seen.shape, np.inf)
    last = np.full(ep.enemy_seen.shape[1], np.nan)
    for s in range(ep.n_steps):
        last = np.where(ep.enemy_seen[s], ep.t_rel[s], last)
        out[s] = np.where(np.isfinite(last), ep.t_rel[s] - last, np.inf)
    return out


def _output(b: np.ndarray, cfg: FilterConfig, post: _Post, s: int) -> np.ndarray:
    """Output-only processing of a step's beliefs: team coordination, fires, then calibration."""
    return _calibrated_output(_fire_output(_team_output(b, cfg, post, s), cfg, post.ev, s), cfg, post, s)


@lru_cache(maxsize=None)
def _warn_once(message: str) -> None:
    logger.warning(message)


def _calibrated_output(b: np.ndarray, cfg: FilterConfig, post: _Post, s: int) -> np.ndarray:
    """Rescale each enemy's cells so its callout probabilities follow the stored calibration (which
    depends on how long ago the enemy was seen)."""
    if not cfg.calibrated:
        return b
    if post.calibration is None:
        _warn_once(f"no calibration.json next to the models: {cfg.name} is uncalibrated (run `python -m cspredict.calibrate`)")
        return b
    grid = post.grid
    P = np.stack([np.bincount(grid.node_place_idx, row, len(grid.places)) for row in b])
    Q = post.calibration.apply(P, post.since[s] if post.since is not None else None)
    out = b * (Q / np.maximum(P, 1e-300))[:, grid.node_place_idx]
    return out / out.sum(axis=1, keepdims=True)


def _team_output(b: np.ndarray, cfg: FilterConfig, post: _Post, s: int) -> np.ndarray:
    """Mix in each alive enemy's belief rescaled by the team-coordination correction of its
    callout probabilities (team.py). Enemies on the radar enter the matching with their callout known."""
    ep, grid = post.ep, post.grid
    if cfg.team <= 0 or post.teams is None:
        return b
    if cfg.team_params.after_contact and not post.ep.enemy_seen[: s + 1].any():
        return b
    alive = np.flatnonzero(ep.enemy_alive[s])
    if len(alive) == 0:
        return b
    n_places = len(grid.places)
    P = np.stack([np.bincount(grid.node_place_idx, b[e], n_places) for e in alive])
    for i, e in enumerate(alive):
        if ep.enemy_seen_node[s, e] >= 0:
            P[i] = 0.0
            P[i, grid.node_place_idx[ep.enemy_seen_node[s, e]]] = 1.0
    phase = int(ep.phase[s])
    clock = float(ep.t_rel[s]) - (post.plant_t if phase > 0 and post.plant_t is not None else 0.0)
    buy = _buy_belief(cfg, ep, post.ev)
    R = team_callouts(post.teams, SIDE_ID[ep.enemy], phase, clock, None if buy is None else buy[s], P, cfg.team_params)
    if R is None:
        return b
    corrected = b[alive] * R[:, grid.node_place_idx]
    corrected /= corrected.sum(axis=1, keepdims=True)
    out = b.copy()
    out[alive] = (1 - cfg.team) * b[alive] + cfg.team * corrected
    return out


def _run_core(
    ep: Episode,
    grid: NavGrid,
    motion: MotionModel,
    cfg: FilterConfig,
    ev: Evidence,
    library: TrajectoryLibrary | None = None,
) -> Iterator[tuple[int, np.ndarray]]:
    if cfg.motion == "particles":
        yield from _run_particle_filter(ep, grid, motion, cfg, ev, library)
        return
    n_e = len(ep.enemy_ids)
    b = np.tile(motion.spawn[ep.enemy].astype(np.float64), (n_e, 1))
    last_seen_t = np.full(n_e, np.inf)  # round time of each enemy's latest sighting
    last_weapon = np.full(n_e, -1)  # weapon class in hand at that sighting
    for s in range(ep.n_steps):
        phase = int(ep.phase[s])
        if cfg.motion == "prior":
            tbin = min(int(ep.t_rel[s] // TIME_BIN_S), N_TIME_BINS - 1)
            b = np.tile(motion.occupancy[context_key(ep.enemy, phase)][tbin].astype(np.float64), (n_e, 1))
            if cfg.negative:
                post = b * ev.unseen[s] ** cfg.gamma
                b = post / post.sum(axis=1, keepdims=True)
            yield s, _apply_kills(b, cfg, ev, s)
            continue
        if s > 0 and cfg.motion == "random_walk":
            b = np.asarray(b @ motion.random_walk)
        elif s > 0 and cfg.motion == "learned":
            # Each enemy moves by the matrix for how long ago they were last seen.
            since = ep.t_rel[s - 1] - last_seen_t
            keys = [
                motion.key(ep.enemy, phase, since_bin(v), int(w) if cfg.hmm_weapon else -1)
                for v, w in zip(since, last_weapon)
            ]
            mask = ev.fire_step.get(s) if cfg.fire_mask else None
            for key in set(keys):
                rows = [e for e, k in enumerate(keys) if k == key]
                b[rows] = predict(b[rows], motion.trans[key], mask)
        if cfg.negative:
            post = b * ev.unseen[s] ** cfg.gamma
            mass = post.sum(axis=1, keepdims=True)
            # If the observation rules out every node the models disagree with reality; keep the prior.
            b = np.where(mass > 1e-12, post / np.maximum(mass, 1e-300), b)
        b = _apply_kills(b, cfg, ev, s)
        seen = ep.enemy_seen_node[s]
        for e in np.flatnonzero(seen >= 0):
            b[e] = 0.0
            b[e, seen[e]] = 1.0
            last_seen_t[e] = ep.t_rel[s]
            last_weapon[e] = ep.enemy_seen_weapon[s, e]
        b = (1 - FLOOR) * b + FLOOR / grid.n
        yield s, b


def _run_particle_filter(ep, grid, motion, cfg, ev, library) -> Iterator[tuple[int, np.ndarray]]:
    if library is None:
        raise ValueError("particle filters need a TrajectoryLibrary")
    particles = run_particles(
        ep, grid, library, ev.unseen if cfg.negative else None, cfg.gamma,
        kills=ev.kills if cfg.kills else None, weapon_mismatch=cfg.weapon, weapon_mode=cfg.weapon_mode, seed=cfg.seed,
        fire_step=ev.fire_step if cfg.fire_wait else None, fire_weight=ev.fire_occ if cfg.fire_weight else None,
        buy=_buy_belief(cfg, ep, ev),
    )  # fmt: skip
    if cfg.mix_hmm <= 0 and cfg.mix_prior <= 0:
        for s, b in particles:
            yield s, (1 - FLOOR) * b + FLOOR / grid.n
        return
    plain = replace(cfg, mix_hmm=0.0, mix_prior=0.0)
    hmm = _run_core(ep, grid, motion, replace(plain, motion="learned"), ev)
    prior = _run_core(ep, grid, motion, replace(plain, motion="prior"), ev)
    w_pf = 1.0 - cfg.mix_hmm - cfg.mix_prior
    for (s, b), (_, h), (_, p) in zip(particles, hmm, prior):
        yield s, w_pf * b + cfg.mix_hmm * h + cfg.mix_prior * p
