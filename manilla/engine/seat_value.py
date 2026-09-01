"""What each seat in the turn rotation is actually worth, in pesos.

This replaces the hand-picked random first-mover coefficient that
`harbor_master.first_mover_value` used to take (a per-voyage draw from
0.5-2.0, multiplied by seat distance and by 3) with figures measured from
real self-play games via `selfplay_data.record_self_play_games`.

The old model assumed the cost of sitting one spot further back was
*linear* in distance. The measurement says it plainly isn't -- being
harbor master is worth about 10.6 pesos over the next seat, while every
seat after that is within roughly a peso of the others and not even
ordered monotonically. It's a step, not a slope:

    offset 0 (harbor master)  +16.79
    offset 1                   +6.14
    offset 2                   +4.91
    offset 3                   +5.79

so the old formula badly undervalued winning the auction when the next
active bidder sat immediately behind (3.75 pesos assumed at the midpoint
coefficient, against 10.65 measured) while roughly matching it at three
spots back.

Two things to keep in mind when using or refreshing these numbers:

* **They are accomplice earnings, not the whole value of the office.**
  The snapshot they come from starts after the auction and share
  purchase are paid for (see `selfplay.run_voyage`'s `cash_after_setup`),
  which is what makes them safe to price a bid against -- but it also
  means the share the harbor master buys and their steering of the black
  market via punt positioning are not in here. Treat them as a floor.
* **Feeding them back into bidding is not circular.** What they measure
  is accomplice placement, and `policy.choose_accomplice_action` never
  reads the seat table or `first_mover_value` -- the placement rule is
  independent of the bidding rule, so changing how bots bid does not
  invalidate the figures they placed under. They were measured under the
  old random coefficient and stay valid under this one.

  The one channel that does connect the two is cash, not policy: a bid is
  paid out of the same purse the accomplices are placed from, so bidding
  higher leaves less to place with. That matters in principle because a
  player who runs short encumbers a share, which *raises* cash by
  `SHARE_LOAN_AMOUNT` and reads as earnings in this measurement (see the
  caveat above), so bidding harder could inflate the harbor master's own
  figure without any better play behind it.

  Measured, that effect is negligible. Re-running 14,108 voyages under the
  new bidding moved offset 0 by +0.25 against a standard error of 0.09 on
  each figure, and no seat moved by more than 0.41. The table above is the
  post-change measurement, so it is now a fixed point rather than a
  one-step iteration.
"""

from __future__ import annotations

import json
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Dict, Optional, Sequence, Tuple

# Mean pesos earned by a seat's accomplices in one voyage, indexed by that
# seat's offset from the harbor master (0 = the harbor master).
#
# Source: 14,108 four-player REV voyages across 2,324 games, measured
# 2026-08-31. Standard error on each figure is about 0.09 (clustered by
# game, since voyages within one game share a board and dice history), so
# the harbor-master gap is far beyond doubt while the ordering *among*
# the non-harbor-master seats is real but small.
#
# These are the *self-consistent* figures: they were measured under bidding
# that already used the previous measurement, so the table now reproduces
# itself rather than describing a policy that no longer exists. The
# previous run (11,604 voyages under the old random-coefficient bidding)
# gave 16.786 / 6.139 / 4.911 / 5.789 -- every seat moved by less than
# 0.41, and the shape is unchanged, which is the empirical answer to
# whether feeding these back into bidding would shift them. It doesn't,
# to any degree worth iterating on.
# !! KNOWN TO BE INFLATED, pending a re-measurement (2026-08-31). These
# were recorded under schema v2, before earnings netted out mid-voyage
# encumbrance -- a player who ran short and pledged a share had the
# SHARE_LOAN_AMOUNT loan counted as income. That flatters the harbor master
# most, since paying for the auction and the share is what makes anyone run
# short in the first place.
#
# How much it matters, measured on 15,037 random-policy voyages under the
# corrected v3 rule: offset 0 fell from +8.96 to +5.10 while the other
# seats moved by under 0.8, which all but erased the harbor master's edge
# *under random bidding*. The REV figures below have not been re-measured
# yet and the same correction will pull offset 0 down by some amount, so
# treat the harbor-master advantage here as an upper bound until a fresh
# REV run under v3 replaces them.
MEASURED_SEAT_PROFIT: Dict[int, Tuple[float, ...]] = {
    4: (17.036, 6.540, 5.045, 5.527),
}

DEFAULT_DATA_PATH = Path("data") / "harbor_master_profit.jsonl"


def seat_profit_means(player_count: int) -> Tuple[float, ...]:
    """Mean per-voyage accomplice earnings per seat offset, for a game of
    `player_count` players.

    Player counts we have measurements for return them directly. The rest
    are approximated from the shape the measurement actually shows -- the
    harbor master's own figure, then the average of every measured
    non-harbor-master seat repeated for the remaining seats -- rather than
    by extrapolating a per-seat slope, which is the very thing the data
    says does not exist. Replace with real figures once self-play has run
    at those player counts.
    """
    if player_count in MEASURED_SEAT_PROFIT:
        return MEASURED_SEAT_PROFIT[player_count]

    reference = MEASURED_SEAT_PROFIT[max(MEASURED_SEAT_PROFIT)]
    harbor_master, rest = reference[0], reference[1:]
    later_seat = statistics.mean(rest) if rest else harbor_master
    return (harbor_master,) + (later_seat,) * max(0, player_count - 1)


def measure_seat_profit_means(
    path: Optional[Path] = None,
    policy: str = "rev",
    player_count: int = 4,
) -> Optional[Tuple[float, ...]]:
    """Recompute the means from a `harbor_master_profit.jsonl` written by
    `selfplay_data.record_self_play_games`, or `None` if that file has no
    rows matching `policy`/`player_count`.

    Deliberately *not* called automatically during play: reloading as data
    accumulates would make the bidding policy drift mid-run, so a batch of
    self-play games would no longer be generated under one fixed policy.
    Call it when you want to refresh `MEASURED_SEAT_PROFIT` and update the
    table above by hand.
    """
    path = Path(path) if path is not None else DEFAULT_DATA_PATH
    if not path.exists():
        return None

    by_offset: Dict[int, list] = defaultdict(list)
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if row.get("policy") != policy or row.get("player_count") != player_count:
            continue
        for offset, pesos in row["pesos_by_seat_offset"].items():
            by_offset[int(offset)].append(pesos)

    if not by_offset:
        return None
    return tuple(statistics.mean(by_offset[offset]) for offset in sorted(by_offset))


def seat_advantage(seat_means: Sequence[float], spots_behind: int) -> float:
    """How much better the harbor master's seat is than one `spots_behind`
    it, in pesos -- the quantity `harbor_master.first_mover_value` prices.

    `spots_behind` of 0 means the same seat, so this is 0. Values past the
    end of `seat_means` clamp to the last seat rather than raising, since a
    caller's turn order can be shorter than the table when seats have
    dropped out of an auction.
    """
    if not seat_means or spots_behind <= 0:
        return 0.0
    index = min(spots_behind, len(seat_means) - 1)
    return seat_means[0] - seat_means[index]
