"""JSON-Lines data collection for self-play games -- hooks into
`manilla.engine.selfplay`'s `run_voyage`/`run_game` callbacks to write two
independent datasets to disk as games are played:

1. **Harbor-master profitability log** (`harbor_master_profit.jsonl`) --
   one row per voyage: pesos earned *by each seat's accomplices* this
   voyage, keyed by seat offset from the harbor master (0 = the harbor
   master themselves, 1 = the next player in turn order, 2 = the one
   after that, ...). Measured from a snapshot taken after the auction
   and the harbor master's share purchase are both settled (see
   `run_voyage`'s `cash_after_setup`), so what the office *cost* is
   deliberately excluded: this is the value of the seat position
   itself -- placement costs, cargo/dock/plunder payouts, insurance --
   not a verdict on whether the current bidder paid too much for it.
   Keeping the price out matters for milestone 5: a bidding model
   trained on a figure its own bids had already moved would be
   circular. Meant for plain aggregate stats (mean pesos per offset)
   rather than a learned model -- the per-position advantage is a
   marginal question about a distribution, and with only 4-5 offsets
   there's no dimensionality problem a regression would solve. Note
   the offsets aren't comparable across different `player_count`s
   (offset 4 only exists in 5-player games), and rows from one
   `game_id` share a board and dice history, so they aren't
   independent draws -- stratify and cluster accordingly.

2. **Harbor-master bidding/buying training data**
   (`bid_buy_training.jsonl`) -- one row per voyage: the board state
   right after that voyage's harbor master finished loading and placing
   punts (how many shares of each ware are currently owned by anyone,
   and which 2 of the 3 loaded wares got the most favorable -- highest,
   i.e. shortest-remaining-distance -- start position), labeled with
   that *game's* eventual final black-market standings once the game
   actually ends (every voyage in a game shares the same label -- the
   final outcome the whole game converged to, not that voyage's own
   result). Feeds Roadmap milestone 5's regression model.

JSON Lines, not one big JSON array: self-play runs can be long, and this
lets each game's rows get appended as they're known rather than
rewriting an ever-growing file, keeps a crash mid-run from corrupting
already-written games, and streams into a trainer without loading
everything into memory at once. Matches the project's existing
`json`-only, zero-dependency convention (`GameState.save`/`load`) --
deliberately not a SQL database (sqlite3 is stdlib too, but there's no
querying need here beyond "load a file of rows," and JSONL is directly
appendable per-game in a way a single SQLite file writer would need more
ceremony to get equally safe under a crash mid-run).
"""

from __future__ import annotations

import json
import random
import uuid
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional, Set

from manilla.engine.models import GameState, Player, Ware
from manilla.engine.selfplay import GameResult, new_bot_game, run_game


# Bumped whenever a row's shape or the *meaning* of a recorded number
# changes, so a reader can tell which code produced a line instead of
# guessing from the fields present. Stamped on every row.
#
#   1  original rows (implicit -- these carry no schema_version at all):
#      earnings measured from the top of the voyage, so net of the auction
#      and share purchase; no black_market on training rows.
#   2  earnings measured from after setup is paid for (cash_after_setup),
#      and training rows carry the current black_market levels.
#   3  earnings net out mid-voyage encumbrance: taking a SHARE_LOAN_AMOUNT
#      loan against a share used to read as income, inflating whoever ran
#      short. Only harbor_master_profit rows change meaning; bid_buy rows
#      are identical to v2 in content.
#
# Twice now a dataset had to be set aside because nothing in it recorded
# this, and the rows couldn't be told apart after the fact -- see
# data/seat_value_measurement_old_bidding/README.txt.
SCHEMA_VERSION = 3


@dataclass
class HarborMasterProfitRow:
    game_id: str
    voyage_number: int
    player_count: int
    policy: str
    pesos_by_seat_offset: Dict[int, int]
    schema_version: int = SCHEMA_VERSION


@dataclass
class BidBuyTrainingRow:
    game_id: str
    voyage_number: int
    player_count: int
    policy: str
    shares_in_play: Dict[str, int]  # ware.value -> count currently owned by any player
    favored_wares: Set[str]  # the 2 of 3 loaded wares with the most-advanced start position -- an
    # unordered pair, not a sequence; serialized as a sorted list (see record_self_play_games) since
    # JSON has no set type
    black_market: Dict[str, int]  # ware.value -> current level, at the moment this voyage loaded.
    # Added 2026-08-30 after the first share-buying model underperformed a one-line heuristic: the
    # final level is bounded below by the current one (a ware at 20 needs one more rise; a ware at 0
    # needs four), so without this the model was guessing at the single most predictive input it
    # could have had. Rows written before that date don't carry it.
    schema_version: int = SCHEMA_VERSION
    final_black_market: Optional[Dict[str, int]] = None  # filled in once the game ends -- the "solution"


def _seat_offsets(state: GameState, harbor_master: Player) -> Dict[str, int]:
    """`player_id -> seat offset from the harbor master (0, 1, 2, ...)`,
    in the same turn order accomplice-round turns actually rotate
    through (see `selfplay.run_voyage`)."""
    players = state.players
    start_idx = next(i for i, p in enumerate(players) if p.id == harbor_master.id)
    order = players[start_idx:] + players[:start_idx]
    return {p.id: i for i, p in enumerate(order)}


def _favored_wares(state: GameState) -> Set[str]:
    """The 2 of the 3 currently-loaded wares with the most-advanced
    (highest) start position -- shortest remaining distance to arrival,
    the harbor master's clearest positional favoritism. An unordered
    pair (there's no meaningful "first" vs "second" favorite), so a set
    rather than a list. Only meaningful right after load-and-place,
    before any dice roll has moved anything."""
    loaded = sorted((p for p in state.punts if p.ware is not None), key=lambda p: p.position, reverse=True)
    return {p.ware.value for p in loaded[:2]}


def record_self_play_games(
    n: int,
    player_count: int = 4,
    policy: str = "rev",
    seed: Optional[int] = None,
    max_voyages: int = 200,
    out_dir: str = "data",
) -> List[GameResult]:
    """Runs `n` self-play games exactly like
    `manilla.engine.selfplay.run_self_play_games`, but also appends both
    datasets (see module docstring) to `<out_dir>/harbor_master_profit.jsonl`
    and `<out_dir>/bid_buy_training.jsonl` as they're produced -- one game
    fully played, then that game's rows written, before starting the next."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    profit_path = out / "harbor_master_profit.jsonl"
    training_path = out / "bid_buy_training.jsonl"

    rng = random.Random(seed)
    results: List[GameResult] = []

    with profit_path.open("a", encoding="utf-8") as profit_f, training_path.open(
        "a", encoding="utf-8"
    ) as training_f:
        for _ in range(n):
            game_id = uuid.uuid4().hex
            names = [f"Player {i + 1}" for i in range(player_count)]
            state = new_bot_game(names, policy=policy, seed=rng.randrange(2**31))

            profit_rows: List[HarborMasterProfitRow] = []
            training_rows: List[BidBuyTrainingRow] = []

            def on_loaded(state: GameState, harbor_master: Player) -> None:
                training_rows.append(
                    BidBuyTrainingRow(
                        game_id=game_id,
                        voyage_number=state.voyage_number,
                        player_count=len(state.players),
                        policy=policy,
                        shares_in_play={w.value: state.shares_owned(w) for w in Ware},
                        favored_wares=_favored_wares(state),
                        black_market={w.value: v for w, v in state.black_market.values.items()},
                    )
                )

            def on_voyage_end(state: GameState, harbor_master: Player, baseline) -> None:
                offsets = _seat_offsets(state, harbor_master)
                pesos_by_offset = {offsets[p.id]: baseline.earnings(p) for p in state.players}
                profit_rows.append(
                    HarborMasterProfitRow(
                        game_id=game_id,
                        voyage_number=state.voyage_number,
                        player_count=len(state.players),
                        policy=policy,
                        pesos_by_seat_offset=pesos_by_offset,
                    )
                )

            result = run_game(
                state,
                rng=random.Random(rng.randrange(2**31)),
                max_voyages=max_voyages,
                on_loaded=on_loaded,
                on_voyage_end=on_voyage_end,
            )
            results.append(result)

            final_black_market = {w.value: v for w, v in state.black_market.values.items()}
            for row in training_rows:
                row.final_black_market = final_black_market

            for row in profit_rows:
                profit_f.write(json.dumps(asdict(row)) + "\n")
            for row in training_rows:
                # `default=sorted` turns favored_wares (a set -- JSON has
                # no set type) into a deterministically-ordered list at
                # serialization time, rather than relying on set
                # iteration order (which varies run to run).
                training_f.write(json.dumps(asdict(row), default=sorted) + "\n")

    return results
