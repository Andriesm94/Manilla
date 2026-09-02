"""What each seat in the turn rotation is actually worth, in pesos.

This replaces the hand-picked random first-mover coefficient that
`harbor_master.first_mover_value` used to take (a per-voyage draw from
0.5-2.0, multiplied by seat distance and by 3) with figures measured from
real self-play games via `selfplay_data.record_self_play_games`.

The old model assumed the cost of sitting one spot further back was
*linear* in distance. The measurement says it plainly isn't: the harbor
master is clearly ahead, and every seat behind is within about a peso of
the others, not even ordered monotonically. It's a step, not a slope --
which is why `first_mover_value` looks the gap up in a table rather than
multiplying by seat distance.

Three things to keep in mind when using or refreshing these numbers:

* **They are accomplice earnings, not the whole value of the office.**
  The snapshot they come from starts after the auction and share purchase
  are paid for (`selfplay.VoyageBaseline`), which is what makes them safe
  to price a bid against -- but it also means the share the harbor master
  buys and their steering of the black market via punt positioning are not
  in here. Treat them as a floor.
* **Feeding them back into bidding is not circular.** What they measure is
  accomplice placement, and `policy.choose_accomplice_action` never reads
  the seat table or `first_mover_value` -- the placement rule is
  independent of the bidding rule, so changing how bots bid does not
  invalidate the figures they placed under. Confirmed empirically:
  re-measuring 14,108 voyages after bidding switched to these figures
  moved no seat by more than 0.41, against a standard error of 0.09.
* **An encumbrance is a loan, not income.** Until schema v3 (2026-08-31)
  a player who ran short mid-voyage and pledged a share had the
  `SHARE_LOAN_AMOUNT` counted as earnings. That inflated every figure and
  the harbor master's most, since paying for the auction and the share is
  what makes anyone run short. `measure_seat_profit_means` refuses rows
  below v3 rather than averaging them in -- see `MIN_SCHEMA_VERSION`.

  This was not a rounding error: correcting it cut the harbor master's
  measured earnings by 7.53 pesos and left every other seat within 0.44.
  Any future change to what counts as earnings deserves the same
  suspicion, because the seat this bites is never the average one -- it's
  whichever seat the change happens to correlate with.
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
# Source: 14,798 four-player REV voyages across 2,234 games, measured
# 2026-09-02 under schema v3 -- the first figures where a mid-voyage
# encumbrance is treated as the loan it is rather than as income. Standard
# error 0.08-0.09, clustered by game.
#
# The encumbrance correction mattered enormously, and almost entirely to
# the harbor master. Against the v2 figures these replace
# (17.036 / 6.540 / 5.045 / 5.527), offset 0 fell by 7.53 while no other
# seat moved by more than 0.44 -- roughly a hundred standard errors of
# difference on a seat whose whole apparent edge was, in large part, the
# loans it took to pay for the office. Paying for the auction and the
# share is exactly what makes a player run short, so the harbor master
# collected the phantom income nearly every voyage and nobody else did.
#
# What survives is a real but far smaller advantage: the office is worth
# about 3.4-4.8 pesos over a later seat rather than 10.5-12.0. The shape is
# unchanged -- a step at the harbor master, near-flat behind it, with
# offset 3 still ahead of offset 2 -- so `first_mover_value` still reads a
# table rather than scaling by distance. It just bids far less.
MEASURED_SEAT_PROFIT: Dict[int, Tuple[float, ...]] = {
    4: (9.506, 6.141, 4.708, 5.094),
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


# Rows below this counted a mid-voyage encumbrance loan as earnings, which
# inflates every figure and the harbor master's most -- see the note on
# MEASURED_SEAT_PROFIT. Mixing them into a fresh measurement would quietly
# bias it upward, so they're refused rather than averaged in.
MIN_SCHEMA_VERSION = 3


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
        if row.get("schema_version", 1) < MIN_SCHEMA_VERSION:
            continue
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
