# cspredict: where are they now?

Probabilistic enemy-position estimates for **Counter-Strike 2 demo review**, starting with Mirage.

For every enemy who has dropped off one team's radar, it keeps a probability map of where that enemy
is now. The map is built only from what that team knew during the round:
- radar sightings
- where teammates stood and were looking
- smokes and flashes
- the kill feed
- the bomb site

`review` renders a round as a GIF or contact sheet. `evaluate` scores the maps against where the
enemies really were.

![A 1v1 round from a held-out match, seen from the CT side](docs/review_example.png)

*A held-out 1v1, CT view. Blue: CTs and where they look. Ring: an enemy's last sighting and its
age. Heat: the most likely 80% of hidden-enemy probability. Green ×: the true positions, drawn only
for review.*

See [prior_art.md](prior_art.md) for related work. The closest is Hladky & Bulitko (2008), which did
this for CS:Source with 190 LAN logs on one map.

## Quick start

```bash
python3.12 -m venv .venv                             # Python 3.11–3.13; awpy 2 does not install on 3.14
.venv/bin/pip install -e ".[dev]"
.venv/bin/python -m cspredict.fetch_xego             # dev demos -> data/raw/xego/ (~17 GB; --limit 3 to try)
.venv/bin/python -m cspredict.parse                  # demos in data/raw/<source>/ -> data/parsed/
.venv/bin/python -m cspredict.build --sources xego   # fit grid, spotting and motion models (~1 min)
.venv/bin/python -m cspredict.evaluate --split test  # score all filters on held-out matches
.venv/bin/python -m cspredict.review --list <demo_id>
.venv/bin/python -m cspredict.review <demo_id> <round> --side ct --truth
.venv/bin/python -m pytest
```

`review` writes `outputs/review/<demo>_r<round>_<side>_<model>.gif` and `.png`. Use `--truth` to
overlay the real enemy positions, and `--model` to pick another filter (default `ens`).

## Adding HLTV pro demos

HLTV sits behind a Cloudflare challenge, so download demos in your browser:

1. On hltv.org, go to **Results** and filter the map to Mirage (add event or star filters if you like).
2. Open a match and click **GOTV Demo**. You get one `.rar` per series.
3. Put the `.rar` files in `data/raw/hltv/`.
4. Run `python -m cspredict.parse`. It unpacks the archives with `bsdtar` and parses only the Mirage maps.
5. Train and score:
   - `python -m cspredict.build --sources hltv`, or `--sources hltv xego` to pool with the FACEIT data
   - `python -m cspredict.evaluate --train-sources hltv --split test`

Each demo gets a stable random split (70/10/20 train/val/test). To hold out, say, the newest events,
add `data/raw/hltv/splits.csv` with columns `demo_id,split`.

## Data used so far

Development used the 46 Mirage matches of **X-Ego-CS**
([paper](https://arxiv.org/abs/2510.19150), [data](https://huggingface.co/datasets/wangyz1999/X-EGO-CS), MIT).
- These are FACEIT matches between top-100 FACEIT players, not pro teams, so strategies are less
  coordinated than in HLTV demos.
- They ship with 35 / 5 / 6 train / val / test matches.
- `python -m cspredict.fetch_xego` downloads them with the split file. The raw demos are 17 GB;
  the parsed tables in `data/parsed/` are 82 MB.

## How it works

| Step | Module | What it does |
|---|---|---|
| What the team knew | `infostate.py` | Rebuilds the radar from the demo's per-player `spotted_by` lists. Every 0.25 s it records teammates' positions and view angles, which enemies are on the radar, active smokes, flashed teammates, kill-feed entries and the bomb site. True enemy positions are kept for scoring only. |
| Map grid | `grid.py` | awpy's map downloads were returning 403, so the walkable area is learned from where players stand: 1,949 cells of 64 units, two floors where levels overlap (underpass/catwalk), and Mirage's 23 callouts. |
| Spotting model | `visibility.py` | P(an enemy standing here would show up on our radar), given where each teammate stands and looks. Built from ray casts through the walked area plus "carved" see-through space (every real spotting proves its sight line was clear), then a logistic model calibrated on 8.8M observer–enemy pairs. Held-out AUC 0.99. This supplies the negative evidence: my teammate is looking at A ramp and sees nobody, so the enemy is probably not on ramp. |
| Grid motion | `motion.py` | Markov model over cells. Transitions depend on side, bomb phase (not planted / A / B) and time since the enemy was last spotted, because a player who was just seen usually holds or falls back instead of running on. |
| Particle motion | `particles.py` | Each hypothesis follows a real recorded trajectory from a similar situation: same side, nearby cell, heading, round time, bomb phase, time since spotted, and (softly) the team's buy. This captures fast rotations that the grid model smears out. |
| Filters | `filters.py` | Bayes filter per enemy: predict with a motion model; update on radar sightings, on "not on the radar", and on the kill feed (the killer must have had a sight line to the victim). `ens` = 75% particles + 15% grid + 10% "usual positions at this round time". |

## Results (6 held-out matches, 48,876 samples)

A sample is an enemy who was seen earlier in the round and is off the radar now, which is the
mid/late-round case. Higher is better except for log-loss.

| Model | Callout top-1 | Callout log-loss | Cell log-loss | True cell in top 20 |
|---|---|---|---|---|
| **ens** (particles + grid + prior, all evidence) | **0.461** | **1.78** | **6.10** | 0.283 |
| pf (particles) | 0.451 | 2.01 | 7.33 | 0.266 |
| hmm (grid motion) | 0.447 | 2.70 | 7.33 | **0.296** |
| diffuse (random walk, no learning) | 0.419 | 3.26 | 8.18 | 0.234 |
| last_seen ("they're where I last saw them") | 0.378 | 7.83 | 16.77 | 0.048 |
| prior (where that side usually is at this time) | 0.219 | 2.60 | 7.59 | 0.069 |

For reference, a uniform guess over the 1,949 cells has cell log-loss 7.58, and over the 23 callouts it has callout log-loss 3.14.

Callout top-1 by seconds since the enemy was last seen (ens / last_seen / prior):

| 0–2 s | 2–5 s | 5–10 s | 10–20 s | 20–40 s | 40 s+ |
|---|---|---|---|---|---|
| 0.89 / 0.88 / 0.19 | 0.70 / 0.64 / 0.18 | 0.50 / 0.41 / 0.21 | 0.35 / 0.27 / 0.23 | 0.26 / 0.16 / 0.24 | 0.19 / 0.04 / 0.25 |

Clutches with one friendly player left (callout top-1, ens / last_seen / prior):

| 1v1 | 1v2 | 1v3 | 1v4 | 1v5 |
|---|---|---|---|---|
| 0.25 / 0.20 / 0.18 | 0.37 / 0.31 / 0.35 | 0.40 / 0.32 / 0.27 | 0.30 / 0.25 / 0.17 | 0.36 / 0.30 / 0.12 |

What mattered, from validation ablations:
- **Negative information** improves every motion model.
- **Conditioning on time since last spotted** was the largest single fix: it roughly halved the cases
  where the truth got near-zero probability.
- **Correct A/B bomb site** helps after the plant.
- **Kill feed:** a small gain. It matters because 48% of killers are not on the radar at the moment
  of the kill.
- **Features you proposed:**
  - Buy type helps a little, but only when matched softly.
  - Man-advantage (1vX) and same-player matching did not help on this FACEIT data. Even soft
    versions left too few matching trajectories. Both are implemented but off; re-test them on HLTV
    data, where teams are coordinated and the same players recur.
- **Scoring rule:** "probability within 300 units of the truth" favours the point guess
  `last_seen`. It is not a proper scoring rule, so the headline numbers use log-loss and callout
  accuracy.

## Known limitations and next steps

- **Pro data.** Everything above is FACEIT pugs. Pro teams run coordinated strategies, so HLTV
  demos are the next step, and team-specific tendencies become learnable.
- **Sound.** Footsteps and gunshots aren't used yet. Gunfire in particular is heard map-wide in
  late-round fights and is probably the biggest remaining gain. Footsteps are already parsed.
- **Identity.** Enemies are assumed identifiable when they appear on the radar.
- **Coordination.** Each enemy is tracked independently, so trades, stacks and "one is lurking"
  logic are not modelled.
- **Geometry is learned.** Windows, low boxes and rarely walked areas are approximate. If awpy's
  `.tri` files become downloadable again (`awpy get tris`), exact line-of-sight could replace the
  ray-cast prior.
- **One map.** Everything is learned per map, so other maps need their own demos.
