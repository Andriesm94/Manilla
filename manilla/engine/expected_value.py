"""Expected-coin-return calculations built on the dice-only probabilities in
`manilla.engine.probability` and the price/payout tables in
`manilla.engine.models`.

Every function returns a plain `Fraction` coin value -- compare across
candidate placements to argmax, or subtract from a slot's price to see if
it's worth taking at all. None of this assumes anything about what other
players will do; where an outcome depends on a decision someone else makes,
that uncertainty is an explicit parameter rather than a guess baked in.
Modeling what a rational opponent would actually do with that decision is
milestone 5's job (a Monte Carlo rollout against a pluggable opponent
policy), not this module's.

The one place an opponent decision enters the picture, `p_safe_if_caught`,
covers whether a punt caught exactly on space SEA_ROUTE_LENGTH survives to
dock safely rather than being plundered -- see
`manilla.ui.board_setup.BoardSetupApp._resolve_plunder`. Everything else
here is pure dice math: a pirate accomplice's plunder payout is automatic
and mandatory once a punt is caught (no board/skip choice at that point, see
`_roll_dice_and_move`'s call into `_resolve_plunder`), and port/shipyard
slot rewards depend only on how many punts land there in total, not on
opponents' choices, so both are computed exactly.
"""

from __future__ import annotations

from fractions import Fraction
from typing import Dict, Iterable, Sequence, Tuple, Union

from manilla.engine.models import (
    DEFAULT_PORT_PAYOUTS,
    DEFAULT_PORT_PRICES,
    DEFAULT_SHIPYARD_PAYOUTS,
    DEFAULT_SHIPYARD_PRICES,
    DEFAULT_WARE_SLOT_PRICES,
    PIRATE_PRICE,
    PLUNDER_PAYOUTS,
    Ware,
)
from manilla.engine.probability import position_outcomes

Numeric = Union[int, Fraction]

# Port and shipyard slots fill in this priority order (see
# `BoardSetupApp._first_available_dock_key`): the first punt to dock takes
# A, the second takes B, the third takes C.
DOCK_SLOT_RANK: Dict[str, int] = {"A": 1, "B": 2, "C": 3}


def _as_fraction(value: Numeric) -> Fraction:
    return value if isinstance(value, Fraction) else Fraction(value)


def punt_port_probability(start: int, rounds_remaining: int, p_safe_if_caught: Numeric = 1) -> Fraction:
    """Probability a punt at `start` ends up docked in port: it either
    overshoots space SEA_ROUTE_LENGTH on some round, or lands exactly on it
    and survives any pirates present (`p_safe_if_caught`, default 1 = assume
    none are)."""
    outcomes = position_outcomes(start, rounds_remaining)
    return outcomes["arrived"] + outcomes["caught_on_13"] * _as_fraction(p_safe_if_caught)


def punt_shipyard_probability(
    start: int, rounds_remaining: int, p_sent_to_shipyard_if_caught: Numeric = 0
) -> Fraction:
    """Probability a punt at `start` ends up in the shipyard: either it
    falls short of space SEA_ROUTE_LENGTH after all rounds (shipwrecked,
    unaffected by pirates -- they only ever intercept a punt sitting
    exactly on that space), or it lands exactly on it and pirates present
    choose to send it to the shipyard instead of port
    (`p_sent_to_shipyard_if_caught`, default 0 = assume no pirates are
    aboard, or that they'd send it to port instead). This is
    `punt_port_probability`'s `p_safe_if_caught` complement for the same
    caught-on-13 event -- a caller resolving one should resolve the other
    the same way (`1 - p_safe_if_caught`), or the two probabilities won't
    add up to what actually happens to that punt.
    """
    outcomes = position_outcomes(start, rounds_remaining)
    return outcomes["shipwrecked"] + outcomes["caught_on_13"] * _as_fraction(p_sent_to_shipyard_if_caught)


def ware_slot_expected_payout(
    ware: Ware,
    start: int,
    rounds_remaining: int,
    accomplices_on_punt: int,
    p_safe_if_caught: Numeric = 1,
) -> Fraction:
    """Expected gross coin payout (no price subtracted) of holding a ware
    punt's accomplice slot -- what `ware_slot_ev` nets against the slot's
    price, and also what `manilla.engine.wealth` uses to value a slot an
    opponent has *already* paid for, where subtracting the price again would
    double-count a cost already reflected in their current cash.

    `accomplices_on_punt` is the total number of accomplices sharing this
    punt's cargo profit, PLUNDER_PAYOUTS[ware], including the one being
    valued -- it only pays out if the punt actually reaches port safely.
    """
    if not 1 <= accomplices_on_punt <= len(DEFAULT_WARE_SLOT_PRICES[ware]):
        raise ValueError(f"accomplices_on_punt must be between 1 and {len(DEFAULT_WARE_SLOT_PRICES[ware])}")

    payout_share = PLUNDER_PAYOUTS[ware] // accomplices_on_punt
    p_safe = punt_port_probability(start, rounds_remaining, p_safe_if_caught)
    return p_safe * payout_share


def ware_slot_ev(
    ware: Ware,
    slot_index: int,
    start: int,
    rounds_remaining: int,
    accomplices_on_punt: int,
    p_safe_if_caught: Numeric = 1,
) -> Fraction:
    """Expected net coin return of paying to place an accomplice on a ware
    punt's `slot_index`'th slot (0 = cheapest/first-filled -- placement
    always takes the cheapest vacant slot regardless of its position in
    DEFAULT_WARE_SLOT_PRICES, see
    `BoardSetupApp._place_or_remove_punt_accomplice`; the prices are sorted
    ascending here so `slot_index` means fill order, not raw array index),
    for that punt's ware currently sitting at `start` with `rounds_remaining`
    movement rounds left. See `ware_slot_expected_payout` for the gross
    payout this nets against the slot's price.
    """
    prices = sorted(DEFAULT_WARE_SLOT_PRICES[ware])
    if not 0 <= slot_index < len(prices):
        raise ValueError(f"{ware.value} has slots 0..{len(prices) - 1}, got {slot_index}")

    price = prices[slot_index]
    gross = ware_slot_expected_payout(ware, start, rounds_remaining, accomplices_on_punt, p_safe_if_caught)
    return gross - price


def pirate_expected_payout(
    punts: Iterable[Tuple[Ware, int, int]],
    pirate_count: int,
) -> Fraction:
    """Expected gross plunder payout (no PIRATE_PRICE subtracted) of holding
    a captain or second pirate slot, given the currently loaded punts as
    (ware, start, rounds_remaining) triples.

    Plunder is automatic and mandatory whenever any pirate slot is occupied
    and a punt lands exactly on space SEA_ROUTE_LENGTH after its rounds run
    out (`BoardSetupApp._resolve_plunder`) -- there's no board-or-skip
    choice at that point, so this sums cleanly over every currently loaded
    punt with no opponent-decision uncertainty. `pirate_count` is the total
    number of pirate slots occupied (1 if alone, 2 if both captain and
    second are filled) -- the payout for each qualifying punt splits between
    however many pirates are aboard.
    """
    if pirate_count not in (1, 2):
        raise ValueError("pirate_count must be 1 or 2 (only captain and second exist)")

    total = Fraction(0)
    for ware, start, rounds_remaining in punts:
        p_caught = position_outcomes(start, rounds_remaining)["caught_on_13"]
        total += p_caught * (PLUNDER_PAYOUTS[ware] // pirate_count)
    return total


def pirate_slot_ev(
    punts: Iterable[Tuple[Ware, int, int]],
    pirate_count: int,
) -> Fraction:
    """Expected net coin return of paying PIRATE_PRICE for a captain or
    second pirate slot. See `pirate_expected_payout` for the gross sum this
    nets against the price."""
    return pirate_expected_payout(punts, pirate_count) - PIRATE_PRICE


def dock_fill_distribution(arrival_probs: Sequence[Numeric]) -> Dict[int, Fraction]:
    """Exact distribution over how many of several independent punts end up
    docking at the same destination (port or shipyard), given each punt's
    own probability of doing so -- a Poisson-binomial sum via DP
    convolution. Returns {count: probability}, probabilities summing to 1."""
    dist: Dict[int, Fraction] = {0: Fraction(1)}
    for raw_p in arrival_probs:
        p = _as_fraction(raw_p)
        next_dist: Dict[int, Fraction] = {}
        for count, prob in dist.items():
            next_dist[count] = next_dist.get(count, Fraction(0)) + prob * (1 - p)
            next_dist[count + 1] = next_dist.get(count + 1, Fraction(0)) + prob * p
        dist = next_dist
    return dist


def dock_slot_fill_probability(arrival_probs: Sequence[Numeric], key: str) -> Fraction:
    """Probability that dock slot `key` ('A'/'B'/'C') ends up filled by
    *some* punt, given each candidate punt's own arrival probability.
    Since slots fill strictly in arrival order (A first, then B, then C, see
    `DOCK_SLOT_RANK`), slot `key` is filled exactly when at least
    DOCK_SLOT_RANK[key] of the punts arrive -- which specific punt takes it
    doesn't matter for this probability, only the total count does."""
    if key not in DOCK_SLOT_RANK:
        raise ValueError(f"key must be one of {sorted(DOCK_SLOT_RANK)}, got {key!r}")
    rank = DOCK_SLOT_RANK[key]
    dist = dock_fill_distribution(arrival_probs)
    return sum((p for count, p in dist.items() if count >= rank), Fraction(0))


def _dock_tables(dock: str) -> Tuple[Dict[str, int], Dict[str, int]]:
    if dock == "port":
        return DEFAULT_PORT_PAYOUTS, DEFAULT_PORT_PRICES
    if dock == "shipyard":
        return DEFAULT_SHIPYARD_PAYOUTS, DEFAULT_SHIPYARD_PRICES
    raise ValueError(f"dock must be 'port' or 'shipyard', got {dock!r}")


def dock_slot_expected_payout(dock: str, key: str, arrival_probs: Sequence[Numeric]) -> Fraction:
    """Expected gross coin payout (no price subtracted) of holding a
    port/shipyard slot `key`, given every currently loaded punt's own
    probability of ending up at that destination (see
    `punt_port_probability` / `punt_shipyard_probability`)."""
    payouts, _ = _dock_tables(dock)
    p_filled = dock_slot_fill_probability(arrival_probs, key)
    return p_filled * payouts[key]


def dock_slot_ev(dock: str, key: str, arrival_probs: Sequence[Numeric]) -> Fraction:
    """Expected net coin return of paying to place an accomplice on a
    port/shipyard slot `key`. See `dock_slot_expected_payout` for the gross
    payout this nets against the slot's price."""
    _, prices = _dock_tables(dock)
    return dock_slot_expected_payout(dock, key, arrival_probs) - prices[key]
