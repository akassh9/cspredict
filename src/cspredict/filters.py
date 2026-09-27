"""Belief filters: a probability distribution over grid nodes for every enemy, updated every
filter step (0.25 s) from what the friendly team knows.

    predict:  b <- b @ T                          motion model for the enemy's side, bomb phase
                                                  and time since the enemy was last seen
    update:   on the radar  -> b = delta(node where seen)
              off the radar -> b <- b * L^gamma   L = P(not spotted | enemy in node), from visibility.py
              got a kill    -> b <- b * K         K = P(could hit the victim | killer in node)

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
                keep some mass on every plausible spot so an unexpected route is never ruled out
    ens_nokill  ens without the kill-feed evidence (ablation)
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass, field, replace

import numpy as np

from cspredict.grid import NavGrid
from cspredict.infostate import Episode
from cspredict.motion import N_TIME_BINS, TIME_BIN_S, MotionModel, context_key, since_bin, trans_key
from cspredict.particles import TrajectoryLibrary, run_particles
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


DEFAULT_CONFIGS = (
    FilterConfig("last_seen", "none", negative=False, kills=False),
    FilterConfig("prior", "prior", negative=False, kills=False),
    FilterConfig("prior_neg", "prior", negative=True),
    FilterConfig("diffuse", "random_walk", negative=True),
    FilterConfig("hmm_noneg", "learned", negative=False),
    FilterConfig("hmm", "learned", negative=True),
    FilterConfig("pf_noneg", "particles", negative=False),
    FilterConfig("pf", "particles", negative=True),
    FilterConfig("ens", "particles", negative=True, mix_hmm=0.15, mix_prior=0.10),
    FilterConfig("ens_nokill", "particles", negative=True, kills=False, mix_hmm=0.15, mix_prior=0.10),
)


@dataclass
class Evidence:
    """Everything the friendly team observes besides direct sightings, precomputed per step."""

    unseen: np.ndarray  # (S, N) likelihood of staying off the radar
    kills: dict[int, list[tuple[int, np.ndarray]]] = field(default_factory=dict)  # step -> [(enemy, K)]


def gather_evidence(ep: Episode, grid: NavGrid, spot: SpotModel) -> Evidence:
    unseen = np.empty((ep.n_steps, grid.n))
    for s in range(ep.n_steps):
        xyz, yaw, pitch = ep.observers(s)
        unseen[s] = spot.unseen_likelihood(grid, xyz, yaw, pitch, ep.smokes[s])
    kills = {
        s: [(e, spot.shooter_likelihood(grid, victim, obstructed)) for e, victim, obstructed in step_kills]
        for s, step_kills in enumerate(ep.kills)
        if step_kills
    }
    return Evidence(unseen=unseen, kills=kills)


def _apply_kills(b: np.ndarray, cfg: FilterConfig, ev: Evidence, s: int) -> np.ndarray:
    if cfg.kills:
        for e, like in ev.kills.get(s, []):
            post = b[e] * like
            b[e] = post / max(post.sum(), 1e-300)
    return b


def run_filter(
    ep: Episode,
    grid: NavGrid,
    motion: MotionModel,
    cfg: FilterConfig,
    ev: Evidence,
    library: TrajectoryLibrary | None = None,
) -> Iterator[tuple[int, np.ndarray]]:
    """Yield (step, beliefs) with beliefs of shape (E, N) after each step's update."""
    if cfg.motion == "particles":
        yield from _run_particle_filter(ep, grid, motion, cfg, ev, library)
        return
    n_e = len(ep.enemy_ids)
    b = np.tile(motion.spawn[ep.enemy].astype(np.float64), (n_e, 1))
    last_seen_t = np.full(n_e, np.inf)  # round time of each enemy's latest sighting
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
            keys = [trans_key(ep.enemy, phase, since_bin(v)) for v in since]
            for key in set(keys):
                rows = [e for e, k in enumerate(keys) if k == key]
                b[rows] = np.asarray(b[rows] @ motion.trans[key])
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
        b = (1 - FLOOR) * b + FLOOR / grid.n
        yield s, b


def _run_particle_filter(ep, grid, motion, cfg, ev, library) -> Iterator[tuple[int, np.ndarray]]:
    if library is None:
        raise ValueError("particle filters need a TrajectoryLibrary")
    particles = run_particles(
        ep, grid, library, ev.unseen if cfg.negative else None, cfg.gamma, kills=ev.kills if cfg.kills else None
    )
    if cfg.mix_hmm <= 0 and cfg.mix_prior <= 0:
        for s, b in particles:
            yield s, (1 - FLOOR) * b + FLOOR / grid.n
        return
    plain = replace(cfg, mix_hmm=0.0, mix_prior=0.0)
    hmm = run_filter(ep, grid, motion, replace(plain, motion="learned"), ev)
    prior = run_filter(ep, grid, motion, replace(plain, motion="prior"), ev)
    w_pf = 1.0 - cfg.mix_hmm - cfg.mix_prior
    for (s, b), (_, h), (_, p) in zip(particles, hmm, prior):
        yield s, w_pf * b + cfg.mix_hmm * h + cfg.mix_prior * p
