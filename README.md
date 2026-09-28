# cspredict: where are they now?

Probabilistic enemy-position estimates for **Counter-Strike 2 demo review**, starting with Mirage.

For every enemy who has dropped off one team's radar, it keeps a probability map of where that enemy
is now. The map is built only from what that team knew during the round:
- radar sightings
- where teammates stood and were looking
- smokes, flashes and burning molotovs
- the kill feed
- the bomb site

Enemies are not tracked in isolation: once some of them have been seen, recorded whole teams in
similar situations correct where the others probably are. The percentages are calibrated, so "60%"
means the enemy is there about 60% of the time.

`review` renders a round as a GIF or contact sheet. `evaluate` scores the maps against where the
enemies really were.

![A 1v4 round from a held-out match, seen from the CT side](docs/review_example.png)

*A held-out round (EAC vs lilmix, round 9), CT view: a 1v4 from 48 s with the bomb on A. Blue: CTs
and where they look. Ring: an enemy's last sighting and its age. Grey: smokes. Orange: burning
molotovs, at the fitted fire radius. Heat: the most likely 80% of hidden-enemy probability. Green ×:
the true positions, drawn only for review.*

See [prior_art.md](prior_art.md) for related work. The closest is Hladky & Bulitko (2008), which did
this for CS:Source with 190 LAN logs on one map.

## Quick start

```bash
python3.12 -m venv .venv                             # Python 3.11–3.13; awpy 2 does not install on 3.14
.venv/bin/pip install -e ".[dev]"
.venv/bin/python -m cspredict.fetch_xego             # dev demos -> data/raw/xego/ (~17 GB; --limit 3 to try)
.venv/bin/python -m cspredict.parse                  # demos in data/raw/<source>/ -> data/parsed/
.venv/bin/python -m cspredict.build --sources xego   # fit grid, spotting, motion, fire and team models (~1 min)
.venv/bin/python -m cspredict.evaluate --split val --models ens_uncal --out outputs/eval_val
.venv/bin/python -m cspredict.calibrate --run outputs/eval_val --model ens_uncal   # honest percentages
.venv/bin/python -m cspredict.evaluate --split test  # score all filters on held-out matches
.venv/bin/python -m cspredict.review --list <demo_id>
.venv/bin/python -m cspredict.review <demo_id> <round> --side ct --truth
.venv/bin/python -m pytest
```

Without the `calibrate` step `ens` still runs, uncalibrated, with a warning. Demos parsed before
the fix to smoke and fire end times can be repaired with `python -m cspredict.parse --refresh-grenades`,
which only reads the expiry events from the raw demos.

`review` writes `outputs/review/<demo>_r<round>_<side>_<model>.gif` and `.png`. Use `--truth` to
overlay the real enemy positions, and `--model` to pick another filter (default `ens`). To use the
pro-trained models, add `--train-sources hltv xego` at the end of the command, after the demo id and
round, because the option takes several values.

## Adding HLTV pro demos

HLTV sits behind a Cloudflare challenge, so download demos in your browser:

1. On hltv.org, go to **Results** and filter the map to Mirage (add event or star filters if you like).
2. Open a match and click **GOTV Demo**. You get one `.rar` per series.
3. Put the `.rar` files in `data/raw/hltv/`.
4. Run `python -m cspredict.parse`. HLTV names each demo in a series `…-m<k>-<map>.dem`, so only the
   Mirage demos are unpacked (with `bsdtar`) and the series' other maps stay in the archive. The map is
   then confirmed from each demo's header. Most HLTV servers don't record the `player_blind` event, so
   flashed players are rebuilt from each player's flash duration, which matches the event exactly on
   demos that have both. `--refresh-blinds` redoes only this for demos parsed earlier (about 3 minutes).
5. Train, calibrate and score:
   - `python -m cspredict.build --sources hltv`, or `--sources hltv xego` to pool with the FACEIT data
   - `python -m cspredict.evaluate --train-sources hltv --sources hltv --split val --models ens_uncal --out outputs/eval_val`,
     then `python -m cspredict.calibrate --train-sources hltv --run outputs/eval_val --model ens_uncal`
   - `python -m cspredict.evaluate --train-sources hltv --split test`

Each demo gets a stable random split (70/10/20 train/val/test). To hold out, say, the newest events,
add `data/raw/hltv/splits.csv` with columns `demo_id,split`.

## Data used so far

**HLTV (pro):** 71 Mirage maps, one from each of 71 series downloaded from HLTV in September 2026.
Events include CCT, ESEA, ESL Challenger League, iBuyPower Masters and Stake Pulse, so most teams are
tier 2–3, with some tier 1. The stable hash split gives 47 / 8 / 16 train / val / test maps. The CS2
patch is newer than the FACEIT set (14181–14185 vs 14094), but the Mirage layout matches: 99.9% of pro
positions fall on cells the FACEIT data had already mapped.

**FACEIT (development):** the 46 Mirage matches of **X-Ego-CS**
([paper](https://arxiv.org/abs/2510.19150), [data](https://huggingface.co/datasets/wangyz1999/X-EGO-CS), MIT).
- These are FACEIT matches between top-100 FACEIT players, not pro teams, so strategies are less
  coordinated than in HLTV demos.
- They ship with 35 / 5 / 6 train / val / test matches.
- `python -m cspredict.fetch_xego` downloads them with the split file. The raw demos are 17 GB;
  the parsed tables in `data/parsed/` are 72 MB.

## How it works

| Step | Module | What it does |
|---|---|---|
| What the team knew | `infostate.py` | Rebuilds the radar from the demo's per-player `spotted_by` lists. Every 0.25 s it records teammates' positions and view angles, which enemies are on the radar, active smokes, burning molotovs, flashed teammates, kill-feed entries and the bomb site. A smoke counts as blocking from 1 s after it bursts until 1.5 s before it expires: that is when spotting through smokes stops and starts again in the training demos. Fires are assumed known to both teams (the thrower saw where it landed; the other team sees and hears it). True enemy positions are kept for scoring only. |
| Map grid | `grid.py` | awpy's map downloads were returning 403, so the walkable area is learned from where players stand: about 2,000 cells of 64 units (1,999 with the pooled training data), two floors where levels overlap (underpass/catwalk), and Mirage's 23 callouts. |
| Spotting model | `visibility.py` | P(an enemy standing here would show up on our radar), given where each teammate stands and looks. Built from ray casts through the walked area plus "carved" see-through space (every real spotting proves its sight line was clear), then a logistic model fitted on about 27 million observer–enemy pairs (9 million with the FACEIT data alone). Held-out AUC 0.99. This supplies the negative evidence: my teammate is looking at A ramp and sees nobody, so the enemy is probably not on ramp. |
| Grid motion | `motion.py` | Markov model over cells. Transitions depend on side, bomb phase (not planted / A / B) and time since the enemy was last spotted, because a player who was just seen usually holds or falls back instead of running on. |
| Particle motion | `particles.py` | Each hypothesis follows a real recorded trajectory from a similar situation: same side, nearby cell, heading, round time, bomb phase, time since spotted, and (softly) the team's buy. This captures fast rotations that the grid model smears out. Matching the weapon seen in the enemy's hands is implemented but off (see Results). |
| Fires | `fires.py` | Fitted on the training demos: players stand within 120 units of a burning molotov at 5–15% of their usual density, and at half density 180 units out (the fire radius). A second profile, fitted by maximum likelihood on 66,000 moves next to fires, scales how likely a player is to step into (or stay in) a cell at each distance, per 0.25 s. |
| Team coordination | `team.py` | Snapshots of each side's alive players every 2 s of every training round (133,000). At each step, snapshots from a similar situation (bomb phase, clock, buy, players alive) are weighted by how well they fit all enemies' own beliefs at once, summing over every way to pair our enemies with the recorded players. The result rescales each enemy's callout probabilities, relative to what independent teammates would give; the correction shrinks when few recorded teams fit. |
| Filters | `filters.py` | Bayes filter per enemy: predict with a motion model; update on radar sightings, on "not on the radar", and on the kill feed (the killer must have had a sight line to the victim). `ens` = 75% particles + 15% grid + 10% "usual positions at this round time"; the grid motion avoids burning fires, the output is weighted by the fire occupancy profile, and from the first sighting the team correction is applied. |
| Calibration | `calibrate.py` | A monotone map of the callout probabilities, fitted on validation rounds separately for enemies seen 0–5 s, 5–20 s and 20+ s ago: shortly after a sighting the filters are too unsure, long after it too sure. Stored as `calibration.json` with the models and applied by `ens`. |

## Results

A sample is an enemy who was seen earlier in the round and is off the radar now, which is the
mid/late-round case. "Callout top-1" is how often the most likely of Mirage's 23 callouts is where the
enemy really is. Higher is better except for log-loss. A uniform guess scores 3.14 callout log-loss
and 7.58 cell log-loss.

### Pro matches: 16 held-out HLTV maps, 102,486 samples

The models were trained on 47 HLTV + 35 FACEIT maps (`build --sources hltv xego`). Every setting was
chosen on the 8 validation maps first. All numbers in this section then come from a single
evaluation of the final code and data (`outputs/final/test_pooled`). The intervals are 95%
bootstrap intervals that resample whole rounds. Differences between models are paired: both are
scored on the same moments, so the noise they share cancels and a difference can be clear even when
the two separate intervals overlap.

| Model | Callout top-1 | Callout top-3 | Callout log-loss | Cell log-loss |
|---|---|---|---|---|
| **ens** (particles + grid + prior, fires, team, calibrated) | **0.481** (0.456–0.508) | **0.727** | **1.665** (1.581–1.747) | **5.822** |
| ens before the four enhancements below | 0.479 (0.454–0.505) | 0.726 | 1.689 (1.600–1.777) | 5.849 |
| pf (particles) | 0.472 | 0.718 | 1.875 | 6.917 |
| hmm (grid motion) | 0.466 | 0.704 | 2.591 | 7.060 |
| diffuse (random walk, no learning) | 0.444 | 0.689 | 3.023 | 7.850 |
| last_seen ("they're where I last saw them") | 0.422 | 0.584 | 7.196 | 16.671 |
| prior (where that side usually is at this time) | 0.213 | 0.451 | 2.607 | 7.257 |

Against the ensemble before these enhancements, `ens` improves callout log-loss by 0.024 (0.012 to
0.036) and cell log-loss by 0.027 (0.015 to 0.040). Its top-1 gain, +0.002 (−0.001 to +0.005), is
within noise. The data fixes made along the way (smoke and fire end times, rebuilt flashes) changed
the old ensemble by almost nothing: it scored 0.478 / 1.688 / 5.856 on the demos as first parsed.

What each enhancement did, as the paired difference to the step before it:

| Step | Callout top-1 | Callout log-loss | Cell log-loss | Where it shows |
|---|---|---|---|---|
| 1. Weapon seen at the last sighting (tested, left off) | −0.001 (−0.005, +0.003) | −0.006 (−0.014, +0.002) | −0.012 (−0.029, +0.007) | Nowhere beyond noise: the same model with another random seed moves by up to ±0.020 |
| 2. Molotovs | +0.001 (−0.000, +0.002) | −0.001 (−0.002, +0.000) | −0.004 (−0.007, −0.002) | While a fire burns (19% of samples): top-1 +0.004 (−0.000, +0.009), cell log-loss −0.017 (−0.027, −0.006) |
| 3. Team coordination | +0.001 (−0.002, +0.005) | −0.002 (−0.007, +0.003) | −0.002 (−0.007, +0.003) | Enemies not seen yet after a teammate has been: top-1 +0.016 (+0.011, +0.022), log-loss −0.031 (−0.040, −0.022); team-level count error −0.021 (−0.034, −0.007) |
| 4. Calibration | 0 (never reorders) | −0.021 (−0.032, −0.010) | −0.021 (−0.032, −0.010) | The percentages below |

**Honest percentages.** When `ens` says a callout holds the enemy with a given probability, how
often is the enemy really there (every callout probability)?

| Says | 20–30% | 30–40% | 40–50% | 50–60% | 60–70% | 70–80% | 80–90% | 90%+ |
|---|---|---|---|---|---|---|---|---|
| before calibration | 21% | 32% | 40% | 50% | 62% | 75% | 89% | 99% |
| **after** | **24%** | **33%** | **44%** | **53%** | **67%** | **75%** | **85%** | **97%** |

The weighted gap between "says" and "is there" fell from 0.0040 to 0.0012. Before, the top bin
averaged a claim of 92% and was right 99% of the time: the 10% prior in the mixture capped how sure
the model could ever be. The fix depends on how long ago the enemy was seen. Within 5 s of a
sighting the raw filters are too unsure and get sharpened; after 20 s they are too sure (on
validation, "55%" meant 40%) and get softened. One map for all horizons could only compromise.

Callout top-1 and log-loss by seconds since the enemy was last seen (ens / ens before the
enhancements / last_seen / prior_neg, the prior with negative information):

| | 0–2 s | 2–5 s | 5–10 s | 10–20 s | 20–40 s | 40 s+ |
|---|---|---|---|---|---|---|
| top-1 | 0.89 / 0.89 / 0.88 / 0.20 | 0.72 / 0.72 / 0.68 / 0.20 | 0.55 / 0.54 / 0.50 / 0.21 | 0.42 / 0.43 / 0.37 / 0.25 | 0.30 / 0.30 / 0.23 / 0.20 | 0.25 / 0.25 / 0.15 / 0.21 |
| log-loss | 0.31 / 0.37 / 1.82 / 2.57 | 0.81 / 0.83 / 4.59 / 2.58 | 1.34 / 1.36 / 6.72 / 2.55 | 1.85 / 1.85 / 7.96 / 2.53 | 2.31 / 2.35 / 9.30 / 2.66 | 2.52 / 2.53 / 9.67 / 2.77 |

Clutches with one friendly player left (callout top-1, ens / ens before / last_seen / prior_neg):

| 1v1 | 1v2 | 1v3 | 1v4 | 1v5 |
|---|---|---|---|---|
| 0.60 / 0.59 / 0.43 / 0.36 | 0.51 / 0.52 / 0.42 / 0.35 | 0.48 / 0.49 / 0.38 / 0.30 | 0.40 / 0.39 / 0.34 / 0.16 | 0.41 / 0.41 / 0.39 / 0.13 |

Against a clutcher facing four or five, the new model's log-loss is clearly better (−0.041 and
−0.038). Its top-1 changes, from −0.010 in 1v3 to +0.013 in 1v4, are all within noise. Each slice
holds only 44 to 98 rounds.

**Enemies nobody has seen yet, and the team as a whole.** 246,000 test moments have an enemy who has
not been on the radar yet while a teammate of theirs has. Their callout top-1 is 0.191 (0.175 before
the team model; the plain prior also gets 0.191) and callout log-loss 2.576 (2.607; prior 2.600). At
the team level, the expected number of hidden enemies per callout (the sum of their probabilities)
misses the actual count by a squared error of 2.91 per moment, against 2.93 before and 3.26 for the
prior. It still underestimates stacks: where it expects 2–3 enemies in one callout there are 2.5 on
average.

Which training data works best on the same 16 pro test maps (`ens` before calibration, so that
all three are compared the same way):

| Trained on | Callout top-1 | Callout log-loss | Cell log-loss |
|---|---|---|---|
| FACEIT only (35 maps) | 0.463 | 1.767 | 6.163 |
| HLTV only (47 maps) | 0.470 | 1.723 | 5.939 |
| **Both (82 maps)** | **0.481** | **1.686** | **5.843** |

Pooling beats the pro maps alone by top-1 +0.011 (0.001 to 0.021) and callout log-loss −0.037
(−0.057 to −0.018).

### FACEIT development set: 6 held-out matches, 48,876 samples (FACEIT-only models)

Calibrated on the 5 FACEIT validation matches.

| Model | Callout top-1 | Callout log-loss | Cell log-loss |
|---|---|---|---|
| **ens** | **0.463** | **1.724** | **6.037** |
| ens before the four enhancements | 0.459 | 1.786 | 6.104 |
| last_seen | 0.378 | 7.834 | 16.770 |
| prior | 0.219 | 2.603 | 7.586 |

Here the enhancements improve callout log-loss by 0.062 (0.042 to 0.082), more than on pro matches.
Pro play is more predictable than FACEIT pugs. "Last seen" alone does better on pro matches (42% vs
38%), because pros hold positions more, and the model's lead is largest in pro clutches (1v1: 60%
vs 25% on FACEIT).

What mattered, from validation ablations:
- **Negative information** improves every motion model.
- **Conditioning on time since last spotted** was the largest single fix: it roughly halved the cases
  where the truth got near-zero probability.
- **Correct A/B bomb site** helps after the plant.
- **Kill feed:** a small gain. It matters because 48% of killers are not on the radar at the moment
  of the kill.
- **Flashed teammates:** no measurable change overall. A flashed player really does stop spotting
  (from about a quarter of the flash's duration until 1.25× it), but only one mid/late-round moment in
  eight has a teammate flashed now or in the last 5 s. Rebuilding the flashes that 55 HLTV maps don't
  record moved val callout log-loss by −0.001 (−0.006 to +0.005). In the 5 s after a teammate is
  flashed, cell log-loss improved by 0.029 (0.003 to 0.062).
- **Features you proposed:**
  - Buy type helps a little, but only when matched softly.
  - Man-advantage (1vX) and same-player matching did not help on the FACEIT data. Even soft
    versions left too few matching trajectories. Both are implemented but off and haven't been
    re-tested on the HLTV data yet.
  - The weapon in an enemy's hands when last seen is real information: you see it, and pros call
    it. It also carries a real signal. AWPers move less in the 5 s after being seen (median 190 vs
    300 units for riflers), and players seen with a knife or grenade out move more. But over 80% of
    sightings are rifles or pistols, where the weapon mostly repeats what position and buy already
    say. Matching on it (every class, guns only, or AWP vs not) never beat the Monte Carlo noise on
    validation, or later on test, so it is off. The variants remain as `ens_w0.5`, `ens_guns0.5` and `ens_awp0.5`.
- **Molotovs:** keeping the grid motion out of burning cells and weighting the output by the fire
  profile help while a fire burns. The two particle versions did not help and were slightly worse,
  within noise. Making particles wait at the fire's edge fails because real players often re-route
  and the delayed particles fall out of step. Re-weighting particles in the fire every step counts
  one standing fact as a new observation each step.
- **Smoke timing:** 15% of smokes had been cut 4 s short by a parsing bug. Fixing only that made
  results slightly worse, until smokes were also treated as see-through for their last 1.5 s.
- **Team coordination** helps where it should: enemies nobody has seen yet, once a teammate has
  been, and the team-level counts. Two details mattered. Summing over every pairing of enemies with
  recorded players beat keeping each snapshot's best pairing (what Hungarian matching picks) by a
  wide margin. And correcting before first contact hurt, because with no sightings every enemy's
  belief is the same and the product over them only exaggerates the shared evidence.
- **Calibration by time since seen:** one map for all moments improved held-out log-loss by 0.005 on
  validation; separate maps for 0–5 s, 5–20 s and 20+ s since seen improved it by 0.022.
- **Scoring rule:** "probability within 300 units of the truth" favours the point guess
  `last_seen`. It is not a proper scoring rule, so the headline numbers use log-loss and callout
  accuracy.

## Benchmark: a general-purpose AI model (TypeSafe Jev)

(Run against the ensemble as it was before fires, team coordination and calibration were added.)

`typesafe_bench.py` asks [TypeSafe](https://docs.typesafe.ai/introduction)'s Jev (a hosted model that
answers multiple-choice questions with probabilities) the same question our filters answer. It uses
3,000 random held-out pro moments.
- `describe.py` writes each moment as JSON, using only what the team knew and naming places by callout.
- For each hidden enemy, Jev is asked which of the 23 callouts they're in.
- Its probabilities are scored exactly like ours. The 95% intervals resample whole rounds.

| Model | Callout top-1 | Top 3 | Callout log-loss |
|---|---|---|---|
| **ens** (ours) | **0.484** (0.456–0.515) | **0.735** | **1.67** |
| Jev + walking/running times to each callout (`--variant reach_walk`) | 0.406 (0.375–0.438) | 0.621 | 2.64 |
| Jev + running times (`--variant reach`) | 0.401 | 0.624 | 2.92 |
| Jev, plain description (`--variant base`) | 0.408 | 0.612 | 3.99 |
| last_seen | 0.420 | 0.584 | 7.15 |
| prior | 0.233 | 0.466 | 2.59 |

- **Jev mostly repeats the last sighting.** Its top pick is the last-seen callout 73–84% of the time.
- **It trails `ens` at every horizon.** For example, 47% vs 38% at 10–20 s after the last sighting.
- **It adds nothing to `ens`.** The best blend weight, tuned on half the rounds and checked on the
  other half, is 0.
- **Travel times computed in code** (TypeSafe's own advice) fixed most of its overconfidence, but not
  its accuracy.
- **Walking times matter.** In pro clutches the last player alive shift-walks for 68% of their
  movement (median moving speed 117 units/s, vs 250 running), so running times alone overstate how
  far a player gets.

To run it: `pip install -e ".[bench]"`, put `TYPESAFE_API_KEY` in the environment or in a
git-ignored `.env`, then `python -m cspredict.typesafe_bench --n 3000 --variant reach_walk`. It costs
about $0.25 per 3,000 moments, and answers are cached in `outputs/typesafe/`.

## Known limitations and next steps

- **More pro data.** 71 pro maps is still small, and teams recur across the train/test split.
  Team-specific tendencies and the switched-off 1vX, player and weapon matching are worth re-testing
  as the set grows.
- **Sound** (footsteps, gunfire) is not used, by choice. Gunfire in particular is heard map-wide in
  late-round fights, so it is probably the largest source of information left out.
- **Identity.** Enemies are assumed identifiable when they appear on the radar.
- **Coordination** is a correction on top of independent per-enemy filters. Recorded team snapshots
  reshape each enemy's callout probabilities, but the filters' motion stays independent, and the
  model still underestimates how many enemies stack in one callout.
- **Geometry is learned.** Windows, low boxes and rarely walked areas are approximate. If awpy's
  `.tri` files become downloadable again (`awpy get tris`), exact line-of-sight could replace the
  ray-cast prior.
- **One map.** Everything is learned per map, so other maps need their own demos.
