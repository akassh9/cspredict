# Data

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

Demos parsed before the fix to smoke and fire end times can be repaired with
`python -m cspredict.parse --refresh-grenades`, which only reads the expiry events from the raw demos.

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
