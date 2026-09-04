# Manilla

A Python implementation of the board game *Manilla* (Zoch Verlag, 2005): a fully
playable turn-based game with a Tkinter interface, an expected-value engine that
reasons about dice probabilities exactly, and computer players that can play a
whole game unattended — either by random choice or by a heuristic policy called
**REV** (Relative Expected Value).

The full rules are in [`Manila_rules_EN.pdf`](Manila_rules_EN.pdf). Build status
and what's planned next live in [`docs/manilla-roadmap.html`](docs/manilla-roadmap.html).

Stdlib only — no external dependencies (`tkinter`, `dataclasses`, `fractions`,
`json`, `random`).

## Running it

```bash
python main.py
```

That opens the board window. Start a new game (3–5 players) and set each seat to
Human, Computer (random), or Computer (REV); "Simulate" runs a whole game with all
REV computers while you watch.

Ticking **Hand-deal shares** in that dialog replaces the shuffled opening deal
with one you type in, for reproducing a position from a real table. Computer
seats say what they were dealt; a human seat doesn't have to, and its two
shares are held *unrecorded* — counted, but with no ware attached.

What makes such a position add up is the stock, which you also fill in: every
share that is neither dealt out nor for sale has to be in an unrecorded hand,
exactly. That accounting is the whole reason the stock is an input rather than
something derived (rules p.2 would otherwise pin it at "everything undealt").

Filling a hand in *sets* it without *revealing* it — the computers go on
inferring opponents' holdings from public signals — and the hidden ones stay
hidden until the rules call for them: an encumbered share is set aside
face-down (p.8), so only game-end scoring asks for the names.

Run the tests with:

```bash
python -m unittest discover -s tests
```

434 tests covering the data model, the probability/EV/REV engine, the headless
self-play engine, the learned share-buying model, and UI-integration tests that
drive the real Tkinter app.

## What's here

### Game engine

- **`manilla/engine/models.py`** — the data model for a game snapshot: players,
  punts, wares, the black market, every accomplice space (ware punts, port,
  shipyard, pirate boat, pilot island, insurance), prices and payouts. Round-trips
  to JSON.
- **`manilla/engine/probability.py`** — exact dice math. The pip-sum distribution
  over N rolls, and a punt's exact `arrived` / `caught_on_13` / `shipwrecked`
  probabilities from any position, honoring the mid-voyage overshoot rule. Uses
  `Fraction`, so these are exact values and not simulations.
- **`manilla/engine/expected_value.py`** — per-slot expected coin return built on
  those probabilities: ware-punt slots (accounting for how many accomplices split
  the cargo), the pirate boat, and port/shipyard (an exact Poisson-binomial
  fill-count distribution, since dock slots fill in strict arrival order).

### The REV agent

- **`manilla/engine/beliefs.py`** — infers hidden opponent share holdings from
  public signals (a harbor-master purchase, a punt started well forward), pooling
  what's still unknown and weighting by black-market price.
- **`manilla/engine/wealth.py`** — estimated total wealth, and **REV**: scoring an
  action by its ripple effect across every rival's wealth, not just your own gain.
  Includes `action_impact` (simulate a candidate action on a cloned board) plus
  lookahead scorers for pilot moves.
- **`manilla/engine/harbor_master.py`** — the auction decision: what the office is
  worth, which share to buy, how to load and position the punts, and how high to bid.
- **`manilla/engine/seat_value.py`** — what each seat in the turn rotation is
  actually worth, measured from self-play rather than assumed. Replaced a random
  0.5–2.0 coefficient that priced a later seat as linear in distance.

  The measured shape turns out to depend on the player count, which is why this
  is a lookup table rather than a formula:

  | players | seat 0 | 1 | 2 | 3 | 4 |
  |---|---|---|---|---|---|
  | 4 | 9.51 | 6.14 | 4.71 | 5.09 | — |
  | 5 | 7.84 | 4.14 | 3.35 | 2.89 | 2.22 |

  Four players is a *step* — the harbor master ahead, the rest flat, and seat 3
  even beating seat 2. Five is a clean *slope*. Fitting a curve to either would
  have baked in whichever count was measured first.
- **`manilla/engine/policy.py`** — `choose_accomplice_action`, the top-level "pick
  the best placement this turn" entry point.

### Interface

- **`manilla/ui/board_setup.py`** — the Tkinter board: sea lanes, docks, pirate
  boat, pilot island, insurance, black market, and a live player panel. Also holds
  the rules engine for live play (see the note below).

### Headless self-play

- **`manilla/engine/selfplay.py`** — the same voyage loop with no UI at all, as
  plain synchronous functions over a `GameState`. Seeded end-to-end, so a game or a
  whole batch replays identically. Random-policy games are nearly free (20 games in
  ~0.03s); REV costs roughly 6.5s per voyage, dominated by the punt-setup search.
- **`manilla/engine/selfplay_data.py`** — writes two JSON-Lines datasets to `data/`
  (gitignored) while games run: per-seat accomplice earnings, and per-voyage board
  features labeled with the game's eventual final black-market standings. Every row
  carries a `schema_version`, so a reader can tell which code produced it — twice a
  dataset had to be discarded because that wasn't recorded and the rows couldn't be
  told apart afterwards.
- **`scripts/run_selfplay_until.py`** — runs games until a wall-clock deadline,
  flushing each finished game to disk and surviving individual crashes, for long
  unattended batches.

### The learned model

- **`manilla/engine/share_model.py`** — a ridge regression (normal equations,
  no external library) predicting each ware's *final* black-market level, so the
  harbor master can buy whichever share is worth most net of its price. Trained on
  self-play rows, split by game rather than by row — every voyage in a game shares
  one label, so a row-wise split would score well by recognising the game.

  Scored in pesos rather than ranking accuracy, since the real rule maximises
  predicted level minus price: **+13.70 per voyage** against +10.76 for the
  heuristic, +10.79 for random choice, and a +19.95 oracle.

  Worth knowing how it got here: built on share counts and the favoured punts
  alone, it *lost* to the one-line heuristic "just buy the favoured ware". Adding
  each ware's *current* market level — which bounds the final one from below —
  is what made it work. Trained on 4- and 5-player games pooled, which was
  checked rather than assumed: ware value transfers across player counts, even
  though seat value emphatically doesn't.

## A note on the architecture

The original plan called for a separate `engine/rules.py`, but rules arrived
incrementally as features on top of the UI, so `board_setup.py` holds both the
widgets *and* the live-play rules. `selfplay.py` works around this by
reimplementing the same rules headlessly rather than waiting on that split — which
means **the two must be kept in step by hand** if either one's rules change, and
the headless engine is not a substitute for the UI-integration tests.
