# Protocol: Jev against OpenAI's GPT-6 Luna, asked the same questions

**Results:** [jev.md](jev.md#jev-against-openais-gpt-6-luna-asked-the-same-questions).

Written on 2026-09-28, **before any GPT answer was collected**. The code is
`src/cspredict/frontier_bench.py`. Any change made after the pilot is listed under
[Changes](#changes) with its reason.

## Question

Give TypeSafe's Jev and OpenAI's GPT-6 Luna the same information and the same questions. Which one
names the hidden enemy's callout better? Whose percentages are more honest? And what does each cost,
and how fast does it answer?

**Why Luna.** Luna is the cheapest GPT-6 tier, so it is Jev's natural rival on price. TypeSafe's
own evals put the two about level: Luna at 66.8%, Jev at 67.8%, at $0.0033 and $0.0004 per case
([evals.typesafe.ai](https://evals.typesafe.ai/)). But in those evals, "right" means "agrees with
GPT-6 Astra and Claude Fable 5.1". Here, right means where the enemy really was.

## Models

| | Jev | GPT-6 Luna |
|---|---|---|
| Model | `jev-1.13.0` (pinned) | `gpt-6-luna` (OpenAI lists no dated snapshot) |
| Answers | already cached by the Jev lab; not asked again | collected by this protocol |
| Reasoning | none (a "System One" model) | `medium`, OpenAI's default, set explicitly so a change of default can't alter the run. TypeSafe's evals also ran every model at its default. |
| Price (per million tokens) | $0.042 input, output free | $0.10 input, $0.50 output (reasoning counts as output); half on Flex or Batch |
| Tools | none | none: no web search, no code |

GPT calls use the Responses API with `store=false` and a single answer per question (no sampling
several answers to average). Prompt caching is off (`prompt_cache_options.mode = "explicit"` with no
breakpoints). Every prompt here is different, so caching would only add the 25% cache-write fee.
None of these settings change the answers.

## Moments

GPT answers the same cached moments Jev answered: 2,000 validation moments (8 pro maps) and 3,000
test moments (16 pro maps), in `outputs/jev/moments_{val,test}_seed0_n*.json`.

These descriptions were built before commit 7d312eb. Their `enemy_team_buy` field is therefore the
enemies' **true** buy class. The team's public estimate of it agrees at 92% of scored moments, so
the true class is a small leak. Both models see the same text, so the comparison stays fair.
Absolute numbers inherit the leak, as Jev's published numbers do.

## Variants

Both are primary, and both are decided now:

| Variant | What both models get | Requests (val / test) |
|---|---|---|
| `base` | the plain JSON description of the moment, one question per hidden enemy | 1,982 / 2,961 |
| `split_rates` | Jev's best setup: one enemy's context, facts computed in code on every option, "still where last seen?" and "if not, where?" as two questions, and movement rates from the training rounds | 2,000 / 3,000 |

The requests are built by the same code the Jev lab used (`jev_lab.build`). They are the same
objects Jev received. Rebuilding them gives the same request counts and the same sizes (by
characters) as the Jev run logged.

The other three Jev variants (`reach_walk`, `focus`, `split`) are not planned. If they are added
later, they will be reported as extra runs.

## How a question is shown to GPT

The questions are shown to GPT in TypeSafe's format, with TypeSafe's own definitions of its two
question types ([Choice](https://docs.typesafe.ai/primitives/choice),
[Noul](https://docs.typesafe.ai/primitives/noul)):

- **Instructions** (the same for every request):

  > You answer typed questions about a state. The input is JSON with a `state` and `questions`, a
  > map from each question's name to the question. Each question has a `type`:
  > - "choice": `instructions` is the question you answer. `criteria` holds the answer options, as a
  >   map: each key is an option name and each value is a description of that option. Give the
  >   probability that each option is the right answer. The probabilities sum to 1.
  > - "noul": `instructions` is the yes/no question you answer, or a statement for you to judge.
  >   `criteria`, if present, describes what counts as true and as false. Give the probability that
  >   the answer is yes, where 0 means no and 1 means yes.
  >
  > Answer every question in the JSON format requested.

- **Input:** the request as JSON, `{"state": ..., "questions": ...}`, exactly as sent to Jev.
- **Output:** a strict JSON schema (Structured Outputs). Each choice question gets one number per
  option, and each yes/no question gets one number.
- **Clean-up:**
  - negative numbers become 0
  - each choice question is rescaled to sum to 1, as Jev's answers are
  - yes/no answers are clipped to 0–1
  - split answers are combined into callout probabilities by the same code as Jev's (`jev_probs`)

The wrapper is fixed now. The pilot may change it only to fix format failures, such as API errors or
unreadable output, never to raise scores. Any change goes under [Changes](#changes).

**Failures.**
- A request that fails is retried on the next run.
- A question with no usable answer at the end is scored as **uniform** over its options, so failing
  can never help.
- Failures are counted and reported.
- Truncated answers (hitting the output limit, 25,000 tokens as OpenAI advises) count as failures.

## Scoring

Everything is scored as in the Jev lab, on the same moments:
- **Metrics:** right callout first (top-1), right callout in the top 3, and callout log-loss.
- **Intervals:** 95%, resampling whole rounds (2,000 bootstrap draws).
- **Differences:** paired, moment by moment.
- **Ties get split credit, for every model.** Two options tied for first each get half a top-1 hit,
  which is what breaking ties at random would give on average. The Jev lab counted ties as misses.
  That barely mattered for Jev: its top two options tie in about 0.1% of answers. But numbers written
  out by an AI ("30%, 30%") tie often, so the old rule would be unfair to GPT. The old rule is
  reported too, for continuity with `docs/jev.md`.
- **Calibration:** the same map as Jev's. It is the `platt` map: two numbers per group of seconds
  since last seen (0–5 s, 5–20 s, 20 s+). It is fitted on GPT's validation answers and applied to test.
- **Blend with the formula:** the weight is fitted on validation (0 to 1 in steps of 0.05), as for
  Jev. It shows whether GPT adds anything to the 9-number formula.

**Primary comparisons** (test, calibrated, paired):
1. GPT `split_rates` vs Jev `split_rates`: top-1 and log-loss
2. GPT `base` vs Jev `base`: top-1 and log-loss

**Secondary:**
- GPT vs the formula, "last seen" and cspredict
- top-3
- how often each model picks the last-seen callout
- accuracy by seconds since last seen
- cost per 1,000 questions
- speed: median and 90th percentile seconds per request

## Cost and speed

- **Tokens** are read from the API's usage counts: input, cached, cache-write, and output including
  reasoning.
- **Price:** OpenAI's list prices on 2026-09-28
  ([pricing](https://developers.openai.com/api/docs/pricing)).
- **Speed:** wall-clock seconds per request on the standard tier. It is measured in the pilot for
  both models on the same requests, with the same concurrency. Jev is re-asked only for this timing,
  and its cached answers are kept.

## Order of work and budget

1. **Pilot:** the first 100 validation moments in sampling order (a uniform random subset), both
   variants, standard tier. The hard cap is $2.
   - It measures tokens, cost, speed and format failures.
   - It checks that the pipeline is sane: GPT's top picks look reasonable and its answers score
     above zero. 100 moments are too few to compare accuracy, so no decision rests on it.
2. **Decision, on cost only:** project the full run from the pilot. The user sets its budget, under
   $100. The processing tier (standard, Flex or Batch) changes price and speed, not answers.
3. **Validation run:** 2,000 moments, both variants. It fits calibration and blend weights. The
   pilot's answers are reused, since they are validation moments.
4. **Test run:** 3,000 moments, both variants, **once**.
5. **Report everything,** including failures, cost, and every change to this protocol.

## Disclosure

This protocol and the benchmark code were written with an AI coding assistant, Claude (Anthropic),
which is a third party to both TypeSafe and OpenAI.

## Changes

None. The pilot (2026-09-28: 100 validation moments, 200 requests, $0.13) needed no change to
the wrapper or the settings:
- There were 0 failures.
- The longest output was 2,156 reasoning tokens, far below the 25,000 limit.
- The pilot projects the full run at $6.46 at standard prices, or $3.23 on Flex.

## Checks after the pilot

- **Same requests after db1271d.** That commit rewrote the moment files just before the pilot ran,
  with new cspredict probabilities but the same moments and descriptions. Verified:
  - every request key is in Jev's cache
  - the request sizes match the Jev run's log
  - Jev's own input-token count is identical for all 200 pilot requests sent to Jev again
- **Jev isn't deterministic.** When the same request was asked again, the largest change in any
  probability had a median of 0.02 (max 0.15). So each model's answers are one draw, as the rules
  already treat them.
- **Both models give exact zeros.** Jev's probabilities are rounded to 0.01. On the 100 pilot
  moments:

  | | Plain | Pushed |
  |---|---|---|
  | Jev gives the true callout 0% | 23 | 17 |
  | GPT gives the true callout 0% | 0 | 5 |
  | Options at exactly 0% per answer, Jev | 15.6 of 23 | 16.8 |
  | Options at exactly 0% per answer, GPT | 1.3 | 9.3 |

  Zeros are floored at 1e-6 for both, as before, so raw log-loss punishes them and calibration can
  soften them.

## The full runs

The validation and test runs followed the protocol with no changes. The test moments were run once.
- **Requests:** 9,943 in total (pilot, validation and test), with 0 failed or unusable answers.
- **Cost:** $6.20, against the pilot's projection of $6.46.
- **Fingerprint:** a hash of every request, taken before the validation run, was identical after it
  and again after the test run.
