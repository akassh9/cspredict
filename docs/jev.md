# Benchmark: a general-purpose AI model (TypeSafe Jev)

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
