# Big-filter — Dixon-Coles Flashscore scanner

Paste a Flashscore mobile page → fixtures are parsed, scored with a Dixon-Coles model, and each game is
checked on three legs: **1X, Home +1.5, Under 2.5**. The leg probabilities are **calibrated against past results**
(`calibration.json`), then one common line (default **55%**) applies to all three: a game is `★ QUALIFIED` only
when **every calibrated leg is at or above that line**. Rows show the three leg probabilities, `MinLeg` (weakest
leg) and `All3` (chance all three land together). Without `calibration.json` the raw model is used with a 69.5% line.
`rank_by="all3"` ranks by All3 instead.

Note the legs can never be equal: Home +1.5 is implied by 1X, so it is always at least as likely (median gap 17 pts).

## Run
```bash
pip install -r requirements.txt
python app.py            # http://localhost:10000   (PORT env var to change)
pip install -r requirements-dev.txt && pytest
```

## Where the probabilities come from
1. **1X2 odds** on the paste (Flashscore mobile **Football » Odds** page, `[ home | draw | away ]`):
   margin is removed and Dixon-Coles expected goals are solved to match them. This is the recommended input.
2. **Team ratings** (`ratings.json`) for fixtures without odds.
3. Otherwise the fixture is **skipped** (tick "include fixtures with no odds/ratings" to see them anyway;
   they all get identical league-average numbers).

Live/finished and postponed lines are skipped and counted.

## Team ratings (optional)
Generate them from a results CSV (home, away, home goals, away goals; optional date):
```bash
python make_ratings.py results.csv -o ratings.json --half-life 365 --check paste.txt
```
`--check` prints which teams in a Flashscore paste still have no rating (usually spelling differences).
Ratings fit the scanner's own goal formula with shrinkage toward 1.0; teams with < 5 games are left out.
Thresholds you enter in the web form are remembered in your browser.

Manual format:
Used only for fixtures that have no odds. Copy
`ratings.example.json` to `ratings.json` (or set `RATINGS_PATH`) and add
`attack` / `defence` multipliers per team (1.0 = average, defence > 1 = leaky).
Names are matched case-insensitively against the Flashscore spelling.

## API
`POST /scan` `{ "raw_text": "...", "threshold": 0.695, "min_threshold": 0.2, "max_xg": 3.25, "notify_telegram": false }`

Telegram alerts need `TELEGRAM_BOT_TOKEN` and `TELEGRAM_CHAT_ID`.

## Layout
`app.py` routes · `ui.py` HTML · `bot.py` Telegram · `scanner/` parser, model, pipeline · `tests/`


## Included ratings and validation
`ratings.json` was fitted from 11 European leagues (England tiers 1-4, Scotland, Germany, France, Italy, Spain,
Netherlands, Belgium, Portugal, Greece, Turkey) for games from 2022-07-01 to the end of 2024-25, and is **a season
out of date**: refresh it with `make_ratings.py` when newer results are available. It only covers those leagues,
so most fixtures on a typical Flashscore paste (national teams, youth, women's, Africa, Asia, lower leagues) have no
rating and rely on odds instead.

`python backtest.py <league csvs>` fits on past games and scores unseen ones. On 5,302 unseen 2024-25 games the
ratings beat flat league-average inputs and each market was well calibrated. `MinLeg` is only the *weakest single leg*; All3 is the real combined probability: predicted 35.3% vs actual 35.5% on average; it never exceeded about 64%, so
games with All3 at 50%+ were ~3% of the test set and hit ~51%.

## Data credit
Results data from engsoccerdata by James P. Curley (free for non-commercial use; please cite:
https://github.com/jalapic/engsoccerdata) and the footballcsv/openfootball "world" datasets.


## Tuning against a target (backtest result)
`python backtest.py <league csvs> --train-from 2019-07-01 --target 0.695` picks cutoffs on the season before
`--split` and confirms them on the games after it (walk-forward, so nothing is tuned on the games it is judged on).
On 2023-24 (validation) and 2024-25 (test), 10,800 games, 11 leagues:

| market | can it reach a 69.5% hit rate? | by predicted level (2024-25, unseen) |
|---|---|---|
| 1X | yes (base rate ~71%) | pred >=70%: 76% hit, >=75%: 79%, >=80%: 84%, >=85%: 90% |
| Home +1.5 | yes (base rate ~86%) | always |
| Under 2.5 | **no** | pred >=60%: 61% hit, >=65%: 64%, >=70%: 73% (only 15 games) |
| All three legs | **no** | pred 50%+: 51% hit; the model never predicts above ~61% |

Other things I tried and did **not** ship because they did not help out of sample: re-tuning the shrinkage,
recency half-life and Dixon-Coles rho (log-loss changed by ~0.001, noise), and Platt calibration of the
probabilities (made the next season slightly worse). Under 2.5 is the weak leg, so no cutoff on the three-leg
combination can honestly be called 69.5%. If you want picks that line up with 69.5%, qualify on 1X (or 1X + Home +1.5)
at a high predicted level; if you keep Under 2.5 in the bet, expect roughly 50% for the best games.


## What a common line delivers (two unseen seasons, 10,843 games)
"Every leg predicted >= line" does **not** mean every leg lands at that percent. Realised hit rates:

| every leg >= | games (share) | 1X | Home +1.5 | Under 2.5 | all three |
|---|---|---|---|---|---|
| 55% | 2,538 (23%) | 69% | 87% | 54% | 40% |
| 60% | 988 (9%) | 70% | 88% | 56% | 42% |
| 65% | 243 (2.2%) | 70% | 87% | 58% | 42% |
| 69.5% | 26 (0.2%) | 73% | 85% | 62% | 46% |

Under 2.5 is the leg that falls short of its predicted level; 1X and Home +1.5 land at or above it.
At the default 69.5% line only about 1 game in 400 qualifies.


## Calibrated common line (the default)
Raw Under 2.5 is overconfident at the top (predicted 67% lands about 57%). `calibration.json` maps each leg's
predicted probability to the rate it actually landed, pooled over 4 seasons (2021-22 to 2024-25, each predicted
with ratings fitted only on earlier games). Build it with
`python backtest.py <league csvs> --train-from 2016-07-01 --split 2025-07-01 --write-calibration calibration.json --cal-seasons 4`.

Out-of-sample check: calibration learned on 2021-22..2023-24 only, judged on 2024-25 (5,434 games):

| every calibrated leg >= | games (share) | 1X | Home +1.5 | Under 2.5 | all three |
|---|---|---|---|---|---|
| 50% | 1,610 (30%) | 69% | 88% | 55% | 40% |
| 54% | 608 (11%) | 70% | 89% | 61% | 45% |
| **55% (default)** | 325 (6%) | 68% | 88% | 60% | 45% |
| 56% | 181 (3%) | 68% | 88% | 62% | 48% |

So at the default line the weakest leg realised about 60%, above the 55% line; no line reached 58% with enough games.
Caveats: the map was learned from rating-based predictions and is **not validated on odds-derived expected goals**
(the output says so); it only reflects the 11 leagues in the data; and the high-line rows are a few hundred games.


## Checking the line on your own picks (pick log)
Tick **Save these picks to picks.csv** on the form (or send `"log_picks": true` to `/scan`). Later:
```bash
python picklog.py settle results.csv            # or: --score "Home|Away|2-1"
python picklog.py report --only-qualified       # predicted vs actual per leg, by source, with 95% ranges
```
Use this to see whether the calibrated 55% line holds on odds-based fixtures, which the backtest could not check.
Wait for 100+ settled picks before changing any line. On a host with an ephemeral disk, download `picks.csv`
before restarts (or set `PICKS_PATH` to a persistent volume).


## Validation on current data (football-data.co.uk, 2025/26 + 2026/27 to 9 Nov 2026)
`data/fd_2025-26_2026-27.csv` (10,045 games, 22 divisions, with average 1X2 and Over/Under odds) and
`data/arg_2024_2026.csv` (Argentina) feed everything below. Learn on 2025/26 (7,634 games), judge on 2026/27 (2,399 unseen games):
`python backtest_odds.py data/fd_2025-26_2026-27.csv --split 2026-07-01`

| test log-loss (lower is better) | 1X | Home +1.5 | Under 2.5 | All three |
|---|---|---|---|---|
| flat base rate | 0.625 | 0.412 | 0.689 | 0.631 |
| **odds-based (used for QUALIFIED)** | **0.565** | **0.375** | **0.670** | **0.614** |

Predicted vs actual (odds path): 1X 68.1% vs 68.3%, Home +1.5 84.1% vs 85.6%, Under 2.5 41.9% vs 43.8%, All3 31.4% vs 32.1%.
At the default line (every calibrated leg >= 55%): 105 games (4.4%); 1X 67%, Home +1.5 88%, **Under 2.5 60%**, all three 48%.
Under 2.5 stays the weakest leg (95% range 50-69% at that line).

**Ratings are only a fallback and cannot qualify.** Ratings fitted on 2025/26 and tested on 2026/27 beat a flat guess on 1X
but not on Under 2.5 or the three-leg combination (best All3 log-loss 0.6262 vs 0.6266 flat), and at a 55% line the weakest
leg landed only 45%. So rows built from `ratings.json` (and unrated "include anyway" rows) are shown as ◆ MODERATE, never
★ QUALIFIED (`qualify_unvalidated=True` overrides). The earlier four-season engsoccerdata ratings did better, so more history
may fix this: rebuild with `make_ratings.py` as more seasons are added (default shrinkage is now 8, best on both datasets).

`ratings.json`: 457 teams from football-data 2025/26-2026/27 (names follow football-data, e.g. "Man City", "Ein Frankfurt")
plus Argentina's Liga Profesional since 2024. Flashscore spellings differ for some clubs: run
`make_ratings.py ... --check paste.txt --aliases aliases.json` to see misses and map them.
`calibration_odds.json` is learned from all 10,033 odds-based games; `calibration.json` (ratings path) is the older
engsoccerdata one and is used only for non-qualifying rows.

Data: football-data.co.uk (free for personal use; please credit it).


## Recent form and match statistics (tested, not added to the odds path)
`python form_analysis.py data/fd_2025-26_2026-27.csv` runs two walk-forward tests (fit on 2025/26, judged on 2026/27).
The data file now keeps shots, shots on target, corners and (2026/27 only) xG.

1. **Odds path: form adds nothing.** A Poisson model of goals using each team's last-6-games form (goals, and a shots-based
   proxy), with the market's own expected goals as the baseline, gained +0.0009 log-likelihood per observation
   (95% range -0.0007 to +0.0029; early season +0.0004, range -0.0024 to +0.0034). Bookmaker odds already price recent form,
   including at the start of the season, so no form term is applied to odds-based rows. Shots-based form had the right sign
   (+0.15 / +0.22) but the effect is far too small to use.
2. **Ratings path: current-season games do help, shots statistics do not.** Refitting monthly with the new season's games
   improved 1X log-loss from 0.609 (ratings frozen at last season's end) to 0.599, so keep `ratings.json` fresh
   (it is fitted through 9 Nov 2026, recency-weighted). Blending in shots-based pseudo-goals did not help (0.6264 vs 0.6253 for
   All3). Even refreshed, ratings only reach All3 0.625 vs 0.628 for a flat guess, which is why ratings rows can't be QUALIFIED.
3. **xG:** only ~2,000 games (2026/27) carry it, too few to test yet; rerun the script as more arrive.
