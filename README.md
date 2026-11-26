# Big-filter — Dixon-Coles Flashscore scanner

## Current rules (read this first)
Statistics decide which fixtures are selected: recent form, goals against, historic form, standings, streaks, head-to-head and past match statistics
(shots on target, corners, fouls, cards, chances created in the previous match), all from earlier games only. Odds only price the legs.

| scanner | markets | rule to qualify |
|---|---|---|
| **1** | 1X, Home +1.5, Under 2.5 | All3 **strictly above 20%** and **every leg priced at least 1.25** |
| **2** | 1X, 12, Over 1.5 | All3 **strictly above 20%** and **every leg priced at least 1.25** |
| **3** | Draw | draw rating **above 40%** (watchlist from 30%) |

Only fixtures that meet a rule are shown (switch "Show" to *All scored fixtures* to see the rest), listed by kickoff time. All3 of 50-100% is badged **HIGHLY EXPECTED**.
Each fixture has a team profile in the recommended format (`√HOME TEAM` / `√AWAY TEAM`: goal record, corner, attack, near goal chances, winning streak, fouls, cards,
home and away form, head2head), each rated against the same league (Excellent / Good / Average / Poor / Bad; fouls and cards Lesser / Average / Plenty).
Open `examples/sample_results.html` in a browser for a real-data example. Settings: `python train_form_model.py`, `python train_draw_driver.py`, `python rules_report.py`.

**What these rules deliver** (`python rules_report.py data/fd_2025-26_2026-27.csv`; trained before 1 Jul 2026, 2,397 unseen games after):

| scanner | rule | games | All3 predicted | All3 landed |
|---|---|---|---|---|
| 1 | All3 > 20% | 99% | | |
| 1 | + every leg >= 1.25 | 340 (14.2%) | 30.7% | **20.3%** |
| 2 | All3 > 20% | 88% | | 38.1% |
| 2 | + every leg >= 1.25 | 202 (8.4%) | 31.0% | **28.7%** |
| 2 | ... of which All3 >= 50% | 4 | | 50% |
| 3 | draw rating > 40% | **0** | | |
| 3 | watchlist 30%+ | 387 | 31.8% | 31.0% draws |

* **Scanner 2 is well calibrated; Scanner 1's All3 is optimistic** (the 1.25 floor selects games the model overrates). Note that Home +1.5 is normally priced ~1.10, so a 1.25 floor on it
  selects games where the home side is *not* strong, the opposite of your "home Excellent / away Bad" profile.
* **All3 of 50%+ with every leg priced 1.25+ almost never happens** (4 games in 2,397): strong favourites are short-priced.
* **A draw rating of 40% does not occur**: the market's highest margin-free draw probability was 33.5% and calibrated models stop near 37%. The one model that produced 40%+ called
  42 games and 17% were draws. Ratings of 30-33% drew 32% of the time (315 games). The watchlist shows those; set the line to 0.30 if you want picks at that level.

**What was added to the statistics:** corners, fouls, cards and chances created improved the All3 predictions on both test splits (log-loss 0.6306 -> 0.6283 and 0.6320 -> 0.6296) and
gave Under 2.5 its first measurable skill; draw drivers (parity, low scoring, tight defences, draw-prone teams, draw runs, head-to-head draws, tempo, corner parity, fouls)
make the draw rating match the market (0.5449 vs 0.5433 market, 0.5513 flat) but not beat it. The original 4-league Dixon-Coles + booster blend is kept as a second opinion.


## Three scanners on one paste (Mode "All three scanners", the default)
Every pasted fixture is scored by all three scanners and listed **by kickoff time**:

1. **Classic**: 1X, Home +1.5, Under 2.5 (unchanged rules).
2. **1X + 12 + Over 1.5**: regular bookmaker markets (see below).
3. **Draw**: your `draw_predictor_pipeline.py`, embedded unchanged in its maths and rules. For Serie B, Ligue 2, Segunda and the Championship it blends a
   time-weighted Dixon-Coles model with a calibrated gradient-boosted model (shots on target, corners, fouls, market draw probability) and picks a draw when
   **probability - 1/odds >= 4 points at draw odds <= 4.10**, with a quarter-Kelly stake capped at 3% of the bankroll. Each line shows `DRAW p:.. (DC ../GB ..) @odds fair .. edge .. stake ..`
   (`DRAW –` for other leagues). It prices against the market by design (the draw price is a model input), so it is not part of the statistics-only rule; it uses the pasted
   1X2 odds, or the open fixtures feed when none are pasted.

Setup: `python train_draw_models.py [--web]` (writes `draw_models.pkl`; `--web` downloads three seasons per league like the original script, needs internet), then run the app.
`python draw_scanner.py` is your weekly script on the faster port (Excel export, needs internet); the original is kept unchanged in `legacy/`. Pick log profile `draw` records
return on stake. Booster: XGBoost if installed, else scikit-learn's HistGradientBoostingClassifier.

**Port check:** on identical simulated data the port's fitted Dixon-Coles parameters, draw probabilities (difference ~1e-16), features and Kelly stakes equal the original's, and the
vectorised likelihood is faster (about 8x on a small league, more on real data).

**Does it find value? Not shown yet.** `python draw_backtest.py data/fd_2025-26_2026-27.csv` refits monthly on past games only and scores the next month against the real average draw price
(428 games, Aug-Nov 2026, the four leagues; only one season of training data was available here, the original uses three):

| draw probability | log-loss (lower is better) |
|---|---|
| flat base rate (24.1% draws) | 0.5518 |
| market, margin removed | 0.5534 |
| Dixon-Coles / calibrated booster | 0.5629 / 0.5533 |
| **blend (the scanner)** | **0.5541** |

The 4-point rule fired on **7 games** (3 landed, 95% range 16-75%), a 2-point rule on 26 (19% landed, -29%), and backing every draw at <= 4.10 lost 18%. Too few picks to
conclude, but nothing here beats the market. In three of four leagues the learned blend weights were all-or-nothing (Dixon-Coles 1.0 / booster 0.0 and the reverse), a sign of
a short validation window; more seasons (`--web`) should stabilise them. Track real picks with `picklog.py` before staking.


## Statistics-only selection (default)
The trained statistical model is the **only selection tool**: recent form (last 5/10), goals against (per game, last 5/10, at home / away),
historic form (season to date, previous season, streaks, head-to-head), standings and past match statistics, all built from earlier games only.
Odds are **not** used for probabilities or selection; they only price the legs for the **1.30 floor** (1X and 12 from the 1X2 odds, Over 1.5
estimated from the market at a 6% margin). Fixtures whose teams are not in the history are skipped, not scored from odds. Switch "Selection tool" to
*Odds first* to use odds as the probability source instead. Each fixture shows the statistics behind it (`stats:` form, GA/g, season/previous ppg, head-to-head).

Both scanners list All3 and a **tier**: A >= 50%, B >= 35%, C = 20% to 35%. The All3 line defaults to **20%**. Out of sample (2,397 unseen games, `python train_form_model.py data/fd_2025-26_2026-27.csv --split 2026-07-01`):

| All3 predicted at least | 1X + 12 + Over 1.5: games (share) | predicted / actual | classic (1X + Under 2.5): games (share) | predicted / actual |
|---|---|---|---|---|
| 20% | 2,045 (85%) | 35.8% / 38.7% | 2,385 (99.5%) | 36.0% / 32.2% |
| 30% | 1,291 (54%) | 41.9% / 44.5% | 2,076 (87%) | 37.3% / 33.7% |
| 40% | 621 (26%) | 49.7% / 50.6% | 504 (21%) | 42.9% / 37.7% |
| 50% | 232 (9.7%) | 58.7% / 58.6% | 10 | 53.4% / 60.0% |

So a 20% line is not a filter (it passes 85-99% of games): the ranking and tiers do the selecting. The classic legs are over-predicted by about 4-5 points and Under 2.5 has no skill.

**With the 1.30 floor** (every leg priced >= 1.30) the 1X + 12 + Over 1.5 scanner keeps only **54 of 2,397 games (2.3%)** at the 20% line, and those landed
All3 **22%** (the model said 29%), against 38.7% for all 2,045 games at 20%+. Games priced that high on all three legs are exactly the ones the bookmaker rates weaker
than the statistics do, and there the bookmaker was right. Floors of 1.20 / 1.25 keep 24% / 8% of games (All3 27% / 30%). Use the floor knowingly.

**More statistics did not help.** Adding goals-against per game, home/away-venue goals against, last-10 form, clean-sheet / failed-to-score rates and previous-season
form (24 extra features) did not beat the 42-feature set on two walk-forward splits (All3 log-loss 0.6306 vs 0.6344 and 0.6320 vs 0.6329; no regularisation level fixed it).
The 42 features already hold recent form, goals against, season form, streaks and head-to-head, so the model uses them; the extras are only used for the displayed analysis.

## The first two scanners in detail (Mode "Both scanners")
Paste a Flashscore mobile page; every fixture is scored by **both** scanners on the same paste and listed **by kickoff time**:

| scanner | legs | rule |
|---|---|---|
| **Classic** (the first scanner, unchanged) | 1X, Home +1.5, Under 2.5 | odds-first: calibrated legs, every leg at the common line (55%); statistics-only: All3 line like the other scanner; All3 and tier shown |
| **1X + 12 + Over 1.5** | 1X, 12, Over 1.5 (all regular bookmaker markets; 1X and 12 together = Home win, so All3 = Home win AND Over 1.5) | **All3 line first (20% statistics-only, 50% odds-first), then every leg priced >= 1.30** (`◇` = passed All3, filtered out on price) |

Probability sources, same for both: the fixture's 1X2 odds (pasted, or looked up in the open fixtures feed) -> the trained
technical-analysis form model (standings, form, streaks, head-to-head, match statistics, psychology proxies; no odds needed) -> team ratings.
Odds and form rows can qualify the 1X + 12 + Over 1.5 scanner; for the classic scanner only odds rows can (the form model has no skill on Under 2.5:
log-loss 0.6885 vs 0.6888 flat, and where all its legs reached 55% the Under leg landed 50%), so form rows there show probabilities but never ★.
Options: Leg price >= (default 1.30), All3 >= (default 50%), Show only qualified fixtures, use the open fixtures feed, log picks.

**The price floor and All3 >= 50% cannot both hold** (2,397 unseen 2026/27 games; 1X/12 priced from the 1X2 odds, Over 1.5 estimated at 6% margin):

| every leg priced at least | games | of which All3 >= 50% | All3 landed |
|---|---|---|---|
| 1.05 | 2,225 | 225 | 63% |
| 1.10 | 1,899 | 72 | 57% |
| 1.15 | 1,261 | 0 | (highest All3 49%) |
| 1.20 | 643 | 0 | (43%) |
| **1.30** | **62** | **0** | (highest All3 31%) |

Strong home favourites are what push All3 over 50%, and they make 1X and Over 1.5 short-priced. Expect "◇ price<floor" rows and almost never a ★ with a 1.30 floor;
lower the floor to 1.10 for rows that meet both (about 3% of games, landing ~57%).

## Open data feeds
football-data.co.uk (free for personal use; credit it): `fixtures.csv` (upcoming fixtures with 1X2/over-under/Asian-handicap prices; collected Fridays for weekend
games and Tuesdays for midweek) and `Latest_Results.csv` (latest matchday with shots). `python update_data.py [--retrain]` or `POST /refresh-data` downloads them,
merges new games into the history the form model reads, caches the fixtures (`data/fixtures_cache.csv`) so fixtures without pasted odds get priced, and optionally refits
the model. Parsing/merging is tested on the real file layout; the live download needs internet and was **not** run in the build sandbox, so try it once on your host.
Env: `HISTORY_PATHS`, `FIXTURES_CACHE`, `FORM_MODEL_PATH`, `ALIASES_PATH`.

Paste a Flashscore mobile page. **Single-selection mode** (Mode: Single selection) works statistics first, price second:

1. **Statistics pick the market.** Each fixture's common result markets (Home win, Away win, 1X, X2, 12) are scored from the fixture's
   1X2 odds, or from the trained form model when there are no odds, and calibrated against past results (`calibration_select.json`).
   These are the most predictable markets (see below); goals totals and 12 are much weaker.
2. **Price filters it.** Prices come from the pasted 1X2 odds (1X = 1/(1/H+1/D), 12 = 1/(1/H+1/A), ...). Only selections priced
   **1.30-1.35** (adjustable: Price from/to) are kept, and the most probable one is the pick (`★ PICK`). Rows with no pasted price
   (form model) say what price you would need (`◇ CHECK PRICE`). Each row also shows the best market by statistics and the edge against the price.

The three-leg mode (1X + 12 + Over 1.5, "All3") remains available (Mode: Three legs); its notes follow.

**What this delivers (honest):** over 2,397 unseen 2026/27 games, selections priced 1.30-1.35 landed 72.1% against a 75.6% break-even,
a return of **-4.8%**, for every market; the engine's calibrated edge was never positive. The price window costs about the bookmaker margin;
it cannot create value. `python selection_report.py data/fd_2025-26_2026-27.csv` reproduces this; log picks (`picklog.py report`)
to measure real return on your own prices.

| market | model gain over base rate | top-quartile hit |
|---|---|---|
| X2 / Home win | 0.062 | 78% / 67% |
| 1X / Away win | 0.060 | 85% / 55% |
| Over / Under 2.5 | 0.019 | 67% / 56% |
| 12 (no draw) | 0.008 | 83% |

**Three-leg mode.** Each fixture is scored on three **regular** markets that every bookmaker lists:
**1X, 12 and Over 1.5 goals**. 1X and 12 together mean *Home win*, so **All3 = Home win AND Over 1.5**. Under 4.5 and Home +1.5 are not listed for most games, so they are not used (the old legs remain available as `profile="classic"`).

**Qualification (default): every leg priced 1.20-1.50.** 1X and 12 use the prices implied by the pasted 1X2 odds
(1/(1/H+1/D) and 1/(1/H+1/A)); Over 1.5 is estimated as fair odds / 1.06 (assumed 6% margin), as is every leg when there are no odds.
Rows show each leg's chance and estimated price, `in band`/`out of band`, and All3. Change the band with `price_min`/`price_max`.
`qualify_by="all3"` (form: "Qualify by -> All3 line") uses All3 >= 50% instead. **The two rules do not overlap**: see below.

Where the probabilities come from, in order:
1. **The fixture's 1X2 odds** (Flashscore Odds page). Out of 2,397 unseen 2026/27 games, All3 >= 50% happened in 372 and landed **64%**.
2. **The trained technical-analysis model** (`form_model.json`), for fixtures with no odds: standings, form, streaks, head-to-head,
   past match statistics and psychology proxies built from previous games, by the same engine used in training. All3 >= 50% picks landed **59%** (232 games).
   It needs both teams in the results history, in the same league and current season; add `aliases.json` for name differences.
3. Team ratings, then league averages: shown, but never QUALIFIED.

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
`data/fd_2025-26_2026-27.csv` (9,902 games, 22 divisions, with pre-match/closing/Pinnacle/best prices, shots and xG; 143 rows with impossible prices removed, e.g. a corrupted Asian-handicap weekend 29-30 Nov 2025) and
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


## Technical analysis battery (market value test)
`python technical_analysis.py data/fd_2025-26_2026-27.csv` tests the standard football indicators against the market prices
(fit on 2025/26, judged on 2,397 unseen 2026/27 games). Result: **the odds already contain all of it.**

| Test | Result |
|---|---|
| Elo rating gap (goal-difference Elo, home advantage 60) | slightly **worse** than the odds alone (gain -0.0004/obs) |
| Rest days / congestion | no effect (all buckets within 0.02 goals of the odds) |
| Home/away-split form | +0.0004, range includes 0 |
| Shots-based form | no gain (-0.0018) |
| Goal form | no gain (+0.00003) |
| All together | -0.0014, range [-0.0043, +0.0012]: nothing |
| Over/under price to pin total goals | Under 2.5 log-loss 0.6790 -> 0.6777, the same as the bookmakers' own price; negligible |
| Best bookmaker price vs Pinnacle no-vig fair price ("value bets") | **loses**: -9.5% ROI on 1,300 bets, worse as the claimed edge grows (-27% at 6%+) |
| Early average price vs closing line | -7.4% = the margin; no timing edge |
| Home price shortening >5% before kickoff | backing early: +3.8% over 1,836 games, but only known after the move |

Takeaway: with prices this efficient, probability and "technical" signals do not create value; cost does. The cheapest
markets are Asian handicap and main 1X2/Over-Under lines at sharp or best prices (margin 3-4% vs 7.5% at the average
book). The one signal that did show something, the price move toward the close, needs two price snapshots: log the
odds when you scan and again near kickoff to test whether following the move works at the price you can actually get.


## SofaScore-style optimiser and arbitrage (tested)
`python optimiser.py data/fd_2025-26_2026-27.csv` builds 42 pre-match features from earlier games only (standings: position, points,
goal difference, matches played; form; streaks; head-to-head; past match statistics; psychology proxies: bounce-back after a
heavy defeat, complacency after a big win, late-season stakes, dead rubbers) and stacks them on the market probability
(train 2025/26, test 2,397 unseen 2026/27 games). Log-loss gain over the market alone, for the 50%+ triple
"1X + Home to score + Under 4.5":

| features added | gain | 95% range |
|---|---|---|
| standings | +0.0017 | [-0.0002, +0.0033] |
| form / streaks / head-to-head / match stats / psychology | between -0.0001 and +0.0006 | all include 0 |
| everything, linear or gradient boosting | +0.0001 / +0.0010 | include 0 |

For "1X + Under 2.5" gradient boosting is significantly **worse** (-0.0059). Games predicted at 50%+ hit 58% on the market probability
alone and 57-58% with the features: the market already prices what these indicators describe. The psychology flags tied to
late-season stakes could only be fitted on 2025/26 (2026/27 is 12-18 rounds old), so they are the least tested.
What lifts All3 above 50% is the **choice of legs** (see the 1X + Home to score + Under 4.5 result), not extra analysis.

`python arbitrage.py quotes.csv` finds 2-way / 3-way arbitrage from several bookmakers' prices and gives the stake split.
In the 9,902 games, best prices across bookmakers gave a 3-way arbitrage in 0.9% of games (0.4% Over/Under, 0.5% Asian handicap).
A third involve a price >20% above the bookmaker average (stale or erroneous quotes that get voided); without them 0.6%,
median profit 0.95%, and by closing the margin was back to 7.1%. Real but rare, small and short-lived.


## Regular-market triple and the trained form model
**Regular markets cannot average 50%.** Searching all 64 non-redundant triples from 14 regular markets (1X2, double chance, over/under 1.5, 2.5, 3.5,
both teams to score) the best average All3 was 36% (1X + 12 + Over 1.5). The earlier 52% needed near-certain legs (Under 4.5, Home to score)
that most bookmakers don't list. What reaches 50% is **selection**: games where All3 is predicted >= 50%.

`python train_form_model.py data/fd_2025-26_2026-27.csv --split 2026-07-01` (trained on 2025/26, judged on 2,397 unseen 2026/27 games, no odds):

| All3 predicted at least | games | predicted | actual |
|---|---|---|---|
| 40% | 621 (26%) | 49.7% | 50.6% |
| 45% | 394 (16%) | 54.0% | 52.3% |
| **50%** | 232 (9.7%) | 58.7% | **58.6%** |
| 60% | 84 (3.5%) | 66.7% | 63.1% |

On the same games the market probability gave 372 games at 50%+ landing 64%, so odds beat the form model; the form model is for the many fixtures
without odds. It adds no value on top of odds (see the optimiser section), so odds rows do not use it.
`python train_form_model.py <csv> -o form_model.json` refits it; keep the history CSV current (the app reads `HISTORY_PATHS`).
The Home-win leg alone (1X + 12) is not predictable beyond what the odds say, so expect the same caveats as everywhere else:
no value over bookmaker prices is claimed, only a calibrated probability.


## The 1.20-1.50 leg-price band vs All3 >= 50% (they conflict)
Tested on 2,397 unseen 2026/27 games (1X2 prices from the data; Over 1.5 price estimated):

| rule | games | All3 actual |
|---|---|---|
| all three legs priced 1.20-1.50 | 495 (20.7%) | **29%** (predicted 32%) |
| All3 predicted >= 50% | 372 (15.5%) | **64%** |
| both | **0** | n/a |

Games with All3 >= 50% are strong home favourites, so 1X (and usually Over 1.5) is priced below 1.20. Inside the band the three legs
each land about 68-73% but only ~29% land together. The leg that most often breaks the band is Over 1.5 (estimated below 1.20 in 69% of games).
Widening the band does not help (1.10-1.60: 65% of games, All3 35%). Pick the rule you want; the default follows the stated 1.20-1.50 band.
