# Prior art: predicting enemy positions in Counter-Strike from partial information

Survey date: 2026-09-27. Scope: models that estimate where unseen enemies are, mid/late round, using only
information the friendly team legitimately has (radar spottings, deaths, what teammates can see).

## TL;DR

- **The core idea has been done once, at small scale:** Hladky & Bulitko (2008) and Hladky's 2009 MSc thesis
  built Bayesian filters (hidden semi-Markov models and particle filters) for **Counter-Strike: Source**.
  Their models matched or beat human experts at predicting Terrorist positions. But they used 190 LAN
  games on one map, made single-point guesses, tracked each enemy independently, and didn't use
  game-state features like economy or man-advantage.
- **Recent CS2 work comes at it from a different angle:** X-Ego (USC ICT, Oct 2025) predicts which
  callout zones contain opponents **from first-person video**, with the minimap masked out. Accuracy is
  low (about 15% exact-match over 23 zones on Mirage).
- **Nobody has published a modern CS2 model** that takes radar-level info plus game state
  (1vX, economy, bomb, time, players) and outputs a calibrated probability map. A few hobby repos try
  this, but none are finished or evaluated.

---

## 1. Direct prior art in Counter-Strike

### 1.1 Hladky & Bulitko — the closest match
- Paper: S. Hladky, V. Bulitko, *An Evaluation of Models for Predicting Opponent Positions in First-Person
  Shooter Video Games*, IEEE CIG 2008, pp. 39–46.
  [ResearchGate](https://www.researchgate.net/publication/224491321_An_Evaluation_of_Models_for_Predicting_Opponent_Positions_in_First-Person_Shooter_Video_Games)
  · [student slides (IIT Bombay)](https://www.cse.iitb.ac.in/~pb/cs621-2009/student-seminars-computer-games/sagar-counter-strike-AIPPT.pdf)
- Thesis (full details): S. Hladky, *Predicting Opponent Locations in First-Person Shooter Video Games*,
  MSc, University of Alberta, 2009. [PDF](https://www.collectionscanada.gc.ca/obj/thesescanada/vol2/002/MR53941.PDF)
- **Inputs (information a human could know):** number of opponents alive, round time, the region the
  friendly team can currently see, positions of opponents currently sighted, and where opponents died.
- **State space:** the map is split into grid cells. They use **one model per opponent** (a "factored"
  model). A joint model over all opponents was intractable: 100 regions and 5 opponents need about
  10^19 transition probabilities.
- **Motion model:** transition and duration probabilities learned from 140 training logs.
- **Observation model ("negative information"):** P(obs | enemy in cell g) = 1 − (fraction of g visible
  to the friendly team). Cells your team can see and that are empty lose their probability mass.
- **Data:** 190 CS:S logs from the Fragapalooza 2006/2007 LAN, de_dust2 only, 5v5, split 140 train / 50 test.
- **Evaluation:** in-game shortest-path distance between predicted and true positions, with the best
  matching between the two sets. A user study had human experts annotate the same rounds.
- **Results:** for Terrorist positions, the models were as accurate as human experts or better. Their
  mistakes looked more human than the true positions with Gaussian noise added. Each update took under 1 ms.
- **Limitations the author lists** (these map directly onto your project):
  1. Tiny corpus and only one map.
  2. Enemies modeled independently, so it can't learn coordination such as trades or stacks.
  3. Point predictions instead of probability distributions.
  4. No conditioning on game-state features. The author suggests conditioning on weapons as future work.
  5. The author suggests a live overlay to help novice players.

### 1.2 X-Ego (CS2, 2025): opponent location from first-person video
- Y. Wang, S. Hans, V. Ustun (USC ICT), *X-Ego: Acquiring Team-Level Tactical Situational Awareness via
  Cross-Egocentric Contrastive Video Representation Learning*, arXiv 2510.19150 (Oct 2025).
  [arXiv](https://arxiv.org/abs/2510.19150) · [code + data](https://github.com/HATS-ICT/x-ego)
- **Dataset (X-Ego-CS):** 45 FACEIT matches (top-100 players), **de_mirage only**. It has 124 h of
  synchronized first-person video for all 10 players plus 64-tick state trajectories.
- **Task, "opponent location nowcast":** given one player's first-person video, predict which of **23
  callout zones** contain opponents (multi-label classification).
- **Results:** exact-match ("subset") accuracy is about 13–15% (DINOv2 13.47%, 14.60% with their method).
- **Minimap masked out** to avoid leaking positions, so this is a different input than yours. Their
  23-zone Mirage callout scheme and the dataset are still reusable.

### 1.3 MLMove / CSKnow (Stanford/NVIDIA, 2024): learned pro movement
- D. Durst et al., *Learning to Move Like Professional Counter-Strike Players*, Computer Graphics Forum /
  SCA 2024. [arXiv](https://arxiv.org/abs/2408.13934) · [project page](https://davidbdurst.com/mlmove/) ·
  [code](https://github.com/David-Durst/csknow)
- Transformer movement model for bots, trained on **123 h / 17k+ rounds of pro de_dust2** (HLTV demos,
  Apr 2021–Nov 2022). The dataset is public (30 GB compressed).
- The model sees **all 10 players**, so it is not a belief model. But when no enemy is visible, the bot's
  aiming module "uses a probabilistic occupancy map to pick a target where enemies are likely to appear".
  Details are in the supplemental material.
- **Relevance:** a strong learned prior on how pros move. It could serve as the motion model inside a filter.

### 1.4 DECOY (2025): discretized CS:GO simulator
- Y. Wang, V. Ustun, C. McGroarty, *A Data-Driven Discretized CS:GO Simulation Environment...*, Winter
  Simulation Conference 2025. [arXiv](https://arxiv.org/abs/2509.06355)
- Waypoint-based simulator with neural models trained on tournament data. Could be used to generate
  rollouts ("where could they be in 10 s?").

### 1.5 Hobby / open-source attempts (none mature)
- [BarberAlec/CSGO_Map_Model](https://github.com/BarberAlec/CSGO_Map_Model) is **exactly this idea**:
  predict T positions from CT positions plus visible Ts. It's unfinished: the neural net was never built,
  and the author got stuck on demo parsing and lack of data.
- [yimingsu01/CSGO-Player-Position-Prediction-with-RNN](https://github.com/yimingsu01/CSGO-Player-Position-Prediction-with-RNN):
  an RNN/LSTM trajectory predictor using awpy. Early stage, no reported metrics.
- [Counter-Strike-coach-AI (Macena CS2 Analyzer)](https://github.com/renanaugustomacena-ux/Counter-Strike-coach-AI)
  has a `belief_model.py` ("Bayesian opponent mental state tracking"). The coach component that uses it
  is marked limited and off by default.

---

## 2. Adjacent Counter-Strike work you can reuse

- **Win probability / situation value:**
  - Xenopoulos et al., *Valuing Player Actions in CS:GO* (2020): XGBoost win-probability model over
    70M+ events, using remaining players and equipment. [arXiv](https://arxiv.org/abs/2011.01324)
  - Xenopoulos & Silva, *Graph Neural Networks to Predict Sports Outcomes* (2021). [arXiv](https://arxiv.org/abs/2207.14124)
  - [CS2_Win_Probability_Model](https://github.com/Henry1145141919810/CS2_Win_Probability_Model): 220
    tier-1 CS2 matches from HLTV, parsed with awpy. A realistic line-of-sight/FOV/smoke map-control
    feature only helped once they **added 15 s of memory**. That is evidence that the team's information
    state carries signal.
- **Economy:** Xenopoulos et al., *Optimal Team Economic Decisions in Counter-Strike*. [arXiv](https://arxiv.org/abs/2109.12990)
- **The inverse problem (anti-cheat):** AntiCheatPT (Loo, Lužkov, Burelli, IEEE CoG 2025) is a
  transformer cheat detector trained on CS2CD, 795 labeled CS2 matches.
  [arXiv](https://arxiv.org/abs/2508.06348) · [code](https://github.com/itubrainlab/AntiCheatPT).
  A good model of "what could this player know" also works as a wallhack detector: flag players who act
  on information the model says they shouldn't have.

---

## 3. The same problem in other games / domains

| Domain | Work | Method | Takeaway |
|---|---|---|---|
| StarCraft | Weber, Mateas, Jhala, *A Particle Model for State Estimation in RTS Games*, AIIDE 2011 ([paper](https://ojs.aaai.org/index.php/AIIDE/article/view/12424)) | Particle filter, parameters learned from expert replays | +10% bot win rate |
| StarCraft | Synnaeve et al., *Forward Modeling for Partial Observation Strategy Games – A StarCraft Defogger*, NeurIPS 2018 ([arXiv](https://arxiv.org/abs/1812.00054), [code](https://github.com/facebookresearch/starcraft_defogger)) | Conv + recurrent encoder-decoder over map grids | Beats rule-based baselines; closest neural analog |
| StarCraft | Jeong et al., *DefogGAN*, AAAI 2020 ([arXiv](https://arxiv.org/abs/2003.01927), [code](https://github.com/TeamSAIDA/DefogGAN)) | Conditional GAN, accumulated past observations | Claims pro-player-level accuracy |
| Dota 2 | OpenAI Five ([paper](https://arxiv.org/abs/1912.06680)) | Reuses the **last-seen** observation for heroes in fog | The "last seen" baseline to beat |
| RoboCup 2D | Sayareh et al., *Denoising Opponents Position in Partial Observation Environment*, 2023 ([arXiv](https://arxiv.org/abs/2310.14553)) | LSTM / DNN | Beats last-seen |
| Unreal Tournament | Tastan, Chang, Sukthankar, *Learning to Intercept Opponents in FPS Games*, IEEE CIG 2012; Tastan & Sukthankar, AIIDE 2011 ([S2](https://www.semanticscholar.org/paper/Learning-Policies-for-First-Person-Shooter-Games-Tastan-Sukthankar/4a4c28b55c65f0eb021029d8276fcf1358ce3d26)) | Max-ent inverse RL motion model + particle filter | Learned per-player motion model |
| Capture the flag | Becht & Bakkes, *meIRL-BC: Predicting Player Positions in Video Games* ([PDF](https://sander.landofsand.com/publications/meIRL-BC_-_Predicting_Player_Positions_in_Video_Games.pdf)) | Inverse RL conditioned on inferred **role** (attack/defend/ambush) | Analog of lurker vs. anchor roles in CS |
| Quake II | Laird, *It Knows What You're Going To Do: Adding Anticipation to a Quakebot*, Agents 2001 ([PDF](https://www.cs.ubc.ca/~krasic/cpsc538a/papers/Agents01.pdf)) | Bot simulates its own tactics from the enemy's position | Early "anticipation" work |
| Game AI (classic) | Isla, *Probabilistic Target Tracking and Search Using Occupancy Maps*, AI Game Programming Wisdom 3 (2006); *Third Eye Crime* ([paper](https://www.semanticscholar.org/paper/Third-Eye-Crime:-Building-a-Stealth-Game-Around-Isla/6eecffc9ea7870f97b8564a38730048710000222)) | Probability grid diffused over the map and erased where visible | Simplest non-learned baseline; a whole game was built on it |
| Game AI (classic) | Bererton, *State Estimation for Game AI Using Particle Filters*, AAAI workshop 2004 ([PDF](https://cdn.aaai.org/Workshops/2004/WS-04-04/WS04-04-008.pdf)) | Particle filter; teammates share particles | Team-shared belief |
| Game AI (classic) | Booth, *The Official Counter-Strike Bot*, GDC 2004 ([slides](https://media.gdcvault.com/gdc04/slides/making_of_official.pdf)) | Nav mesh annotated with hiding, sniper and encounter spots and approach areas | Hand-built tactical priors for CS maps |
| Basketball | Felsen, Lucey, Ganguly, *Where Will They Go?*, ECCV 2018 ([paper](https://link.springer.com/chapter/10.1007/978-3-030-01252-6_45)) | Conditional VAE, personalized per player | Multi-agent, identity-aware forecasting |
| Valorant | *Round Outcome Prediction in VALORANT Using Tactical Features from Video Analysis*, 2025 ([arXiv](https://arxiv.org/abs/2510.17199)); [valorant-minimap-coach](https://github.com/natiixnt/valorant-minimap-coach) | Minimap video features; screen-read minimap with 1.5 s linear extrapolation | Shows the screen-reading route for live use (naive model) |

---

## 4. Data and tooling (CS2)

- **[demoparser2](https://github.com/LaihoE/demoparser)** (Rust core, Python/JS bindings, CS2):
  - Per-tick player fields include `spotted` (`m_bSpotted`) and `approximate_spotted_by`
    (`m_bSpottedByMask`). From these you can **reconstruct what each team had on radar at every tick**.
    The same demo gives the true enemy positions as labels, so you get a clean supervised dataset.
  - `last_place_name` (`m_szLastPlaceName`) gives each player's callout zone, a natural discrete state space.
- **[awpy](https://github.com/pnxenopoulos/awpy)** (MIT, CS2, Python ≥ 3.11, built on demoparser2):
  - Parses ticks, kills, damages, grenades, smokes, infernos, shots and footsteps into Polars frames.
  - Parses nav meshes, runs fast line-of-sight checks using `.tri` map geometry, and computes an
    HLTV-style Rating. `awpy get` downloads map, nav and tri data.
  - [docs](https://awpy.readthedocs.io/en/latest/)
- **Datasets:**
  - HLTV pro demos (CS2).
  - FACEIT demos.
  - [ESTA](https://github.com/pnxenopoulos/esta): 1,558 **CS:GO** pro demos from 2021–22, 2 Hz frames
    ([arXiv](https://arxiv.org/abs/2209.09861)). CS:GO, not CS2; the maps differ slightly.
  - CSKnow (CS:GO, de_dust2).
  - X-Ego-CS (CS2, Mirage).

---

## 5. Where the gap is (what would be new)

1. **Scale and recency:** thousands of CS2 pro rounds across several maps, versus 190 logs on one CS:S map.
2. **A full probability distribution** over nav areas or callouts, with calibration. Evaluate with
   log-likelihood or Brier score, not only point distance.
3. **Conditioning on game state:** 1vX, economy/weapons, bomb state and site, time left, active
   utility, and audible events (shots, footsteps).
4. **Coordination between enemies:** a joint or set-based model, which Hladky's per-opponent models
   could not capture.
5. **Player and team identity or role**, which is probably more predictive of *where* someone goes
   than a scalar skill rating.
6. **A clear baseline ladder:**
   1. last-seen position
   2. occupancy-map diffusion over the nav mesh (Isla-style, no learning)
   3. learned HMM / particle filter (a Hladky replication)
   4. neural model (defogger- or transformer-style)

---

## 6. Rules caveat for live, in-match use

- Hladky pitched a live overlay and noted it would be "nearly undetectable". Undetectable does not
  mean allowed.
- FACEIT prohibits third-party software used to gain an unfair advantage
  ([FACEIT support](https://support.faceit.com/hc/en-us/articles/360015788779-What-is-deemed-to-be-a-cheat)).
- Close precedent: Riot made League companion apps (Blitz, Porofessor, Mobalytics) remove **enemy
  ultimate timer tracking** by 13 Mar 2025, whether the tracking was automatic or manual
  ([report](https://www.zleague.gg/theportal/league-of-legends-riot-bans-third-party-enemy-ultimate-timer-apps-players-react/)).
- Safe uses:
  - post-round or demo review and coaching
  - observer and broadcast overlays
  - bot AI
  - anti-cheat research
