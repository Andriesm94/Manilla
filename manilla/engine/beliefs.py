"""Share-identity inference for a single viewer's point of view.

Each player's starting hand is dealt secretly (`GameState.new_default_game`)
and grows privately whenever they buy a share as harbor master -- everyone
can see *how many* shares an opponent holds (it's a public count, like face-
down cards) and *how many* of each ware are missing from the public pool
(`GameState.shares_owned`), but not which opponent holds which ware, except
where a specific event has revealed it.

This module reconstructs a `viewer_id`'s best estimate of that hidden split:
every share whose identity has been confirmed for a specific player is
tracked exactly; every other share is treated as drawn from one shared
"secret pool", averaged uniformly across every unconfirmed slot -- e.g. if 4
nutmeg and 2 ginseng shares are unaccounted for across three opponents
holding 2 shares each, every one of those 6 unknown slots is assumed to be
4/6 nutmeg, 2/6 ginseng. Nothing here peeks at the real `Player.shares`
identities for anyone but the viewer -- that would defeat the point of a
belief model built to run once per computer player.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from fractions import Fraction
from typing import Dict, Iterable, List, Tuple

from manilla.engine.models import GameState, Ware

# A harbor master pushing a punt's start position this far forward (out of
# the 9-point budget split across 3 punts, MAX_START_SPACE=5 each) is taken
# as a tell that they hold at least one share of that punt's ware -- see
# `punt_start_signals`.
PUNT_START_SHARE_SIGNAL_THRESHOLD = 4


@dataclass
class ShareSignal:
    """One piece of public evidence that `player_id` holds at least one
    additional confirmed share of `ware`, beyond whatever was already
    confirmed for them.

    `source` is metadata for debugging/logging only (e.g. "purchase" for a
    harbor-master share buy, "punt_start" for the forward-start-position
    tell) -- it doesn't affect inference, since every signal this module
    accepts is treated as fully certain once observed.
    """

    player_id: str
    ware: Ware
    source: str = "unknown"


@dataclass
class ShareBeliefs:
    """A viewer's confirmed-share tally: `confirmed[player_id][ware]` is how
    many shares of that ware the viewer is certain that player holds.
    Everything else about that player's hand is unknown and drawn from the
    shared secret pool -- see `secret_pool` and `share_value_estimate`."""

    viewer_id: str
    confirmed: Dict[str, Dict[Ware, int]] = field(default_factory=dict)

    def confirmed_count(self, player_id: str, ware: Ware) -> int:
        return self.confirmed.get(player_id, {}).get(ware, 0)

    def known_total(self, player_id: str) -> int:
        return sum(self.confirmed.get(player_id, {}).values())


def punt_start_signals(
    harbor_master_id: str, punt_start_positions: Iterable[Tuple[Ware, int]]
) -> List[ShareSignal]:
    """The harbor master's own punt-placement choices double as a tell:
    starting a punt at position PUNT_START_SHARE_SIGNAL_THRESHOLD or higher
    is read as them holding at least one share of that punt's ware. Feed the
    result straight into `infer_beliefs`'s `signals` argument; a punt they
    only nudged forward a little produces no signal at all."""
    return [
        ShareSignal(player_id=harbor_master_id, ware=ware, source="punt_start")
        for ware, start in punt_start_positions
        if start >= PUNT_START_SHARE_SIGNAL_THRESHOLD
    ]


def infer_beliefs(state: GameState, viewer_id: str, signals: Iterable[ShareSignal] = ()) -> ShareBeliefs:
    """Build `viewer_id`'s belief state: their own shares are fully known by
    construction, and each signal confirms one more share for whichever
    player it names, capped at that player's actual share count so a
    redundant or over-eager signal can't over-attribute beyond what they
    could possibly hold."""
    beliefs = ShareBeliefs(viewer_id=viewer_id)

    viewer = state.player_by_id(viewer_id)
    if viewer is not None:
        tally: Dict[Ware, int] = {}
        for share in viewer.shares:
            tally[share.ware] = tally.get(share.ware, 0) + 1
        beliefs.confirmed[viewer_id] = tally

    share_counts = {p.id: len(p.shares) for p in state.players}

    for signal in signals:
        if signal.player_id == viewer_id:
            continue  # the viewer's own hand is never uncertain to begin with
        player_confirmed = beliefs.confirmed.setdefault(signal.player_id, {})
        total_known = sum(player_confirmed.values())
        if total_known >= share_counts.get(signal.player_id, 0):
            continue  # already fully accounted for -- nothing left to confirm
        player_confirmed[signal.ware] = player_confirmed.get(signal.ware, 0) + 1

    return beliefs


def secret_pool(state: GameState, beliefs: ShareBeliefs) -> Dict[Ware, int]:
    """How many shares of each ware are known to exist (from the public
    `shares_owned` totals) but aren't yet confirmed to any specific
    player."""
    pool: Dict[Ware, int] = {}
    for ware in Ware:
        confirmed_total = sum(
            beliefs.confirmed_count(player_id, ware) for player_id in beliefs.confirmed
        )
        pool[ware] = max(0, state.shares_owned(ware) - confirmed_total)
    return pool


def unknown_count(state: GameState, beliefs: ShareBeliefs, player_id: str) -> int:
    """How many of `player_id`'s shares have no confirmed ware yet."""
    player = state.player_by_id(player_id)
    if player is None:
        return 0
    return max(0, len(player.shares) - beliefs.known_total(player_id))


def total_secret_slots(state: GameState, beliefs: ShareBeliefs) -> int:
    return sum(unknown_count(state, beliefs, p.id) for p in state.players)


def average_secret_share_value(state: GameState, beliefs: ShareBeliefs) -> Fraction:
    """The uniform-average coin value of one unidentified share, weighted by
    the secret pool's ware composition. 0 if nothing is unaccounted for."""
    slots = total_secret_slots(state, beliefs)
    if slots == 0:
        return Fraction(0)
    pool = secret_pool(state, beliefs)
    total_value = sum(count * state.black_market.share_price(ware) for ware, count in pool.items())
    return Fraction(total_value, slots)


def share_value_estimate(state: GameState, beliefs: ShareBeliefs, player_id: str) -> Fraction:
    """`player_id`'s total share value as `viewer_id` would estimate it:
    confirmed shares at their exact black-market price, plus unconfirmed
    shares at the shared secret-pool average. Exact (not an estimate) for
    the viewer's own holdings, since none of their shares are unconfirmed."""
    known_value = sum(
        beliefs.confirmed_count(player_id, ware) * state.black_market.share_price(ware) for ware in Ware
    )
    avg = average_secret_share_value(state, beliefs)
    return Fraction(known_value) + unknown_count(state, beliefs, player_id) * avg
