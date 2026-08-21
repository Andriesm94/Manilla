"""Estimated total wealth and Relative Expected Value (REV) -- the tie-break
rule from the original `ideas.txt` notes ("an action is justified if it
gives you more relative wealth than anyone else, OR the gap between you and
the players before you becomes smaller"), formalized on top of
`manilla.engine.beliefs` and `manilla.engine.expected_value`.

    opponent_wealth_est = opponent_pesos
                         + opponent_known_shares_value
                         + opponent_assumed_shares_value
                         - SHARE_REPAY_AMOUNT * opponent_encumbered_share_count
                         + expected_return_from_opponents_placed_accomplices

This is a deliberately simple turn-by-turn heuristic, not a re-implementation
of the real end-of-game scorer (`BoardSetupApp._show_game_over_dialog`'s
afford-what-you-can unencumbering logic) -- it values every share at full
price and charges a flat redemption cost per encumbered share, which is
accurate enough to rank actions mid-voyage without needing to simulate the
endgame every time.

A **rival** is any opponent whose estimated wealth exceeds the viewer's own;
REV is the coin gap to a specific rival (`rival_wealth_est - my_wealth_est`,
positive when they're ahead). Turning this into "pick the action that best
closes the largest gap" needs a per-action estimate of how that action moves
each side's wealth -- `rev_adjusted_score` is the composable primitive for
that once a caller has both deltas; wiring it into specific decision points
(which ware's black-market price to raise as harbor master, which punt to
send pirates after) is follow-up work, not done here.
"""

from __future__ import annotations

from fractions import Fraction
from typing import List

from manilla.engine.beliefs import ShareBeliefs, share_value_estimate
from manilla.engine.expected_value import (
    Numeric,
    dock_slot_expected_payout,
    pirate_expected_payout,
    punt_port_probability,
    punt_shipyard_probability,
    ware_slot_expected_payout,
)
from manilla.engine.models import GameState, Phase, PLUNDER_PAYOUTS, Player, PuntStatus, SHARE_REPAY_AMOUNT

# Once a voyage reaches these phases every pending accomplice payout has
# already been folded into players' cash (or is about to be, atomically),
# so counting it again here would double it.
_SETTLED_PHASES = (Phase.PROFIT_DISTRIBUTION, Phase.WARE_RISE)


def encumbered_penalty(player: Player) -> int:
    """The flat redemption cost of a player's encumbered shares -- what it
    would take to unencumber every one of them right now."""
    return SHARE_REPAY_AMOUNT * len(player.encumbered_shares)


def _rounds_remaining(state: GameState) -> int:
    return state.movement_rounds_total - state.movement_round_index


def _ware_slot_gross_per_accomplice(state: GameState, punt, p_safe_if_caught: Numeric) -> Fraction:
    """Gross expected payout *per occupied slot* on a ware punt -- what one
    accomplice on it is owed, given its punt's real status: certain if
    already docked in port, zero if shipwrecked or captured (ware profit
    never pays out there), and probabilistic while still at sea. Callers
    multiply by how many of those slots a specific player holds."""
    occupied = sum(1 for s in punt.ware_slots if s.occupant is not None)
    if punt.status == PuntStatus.IN_PORT:
        return Fraction(PLUNDER_PAYOUTS.get(punt.ware, 0) // max(1, occupied))
    if punt.status in (PuntStatus.IN_SHIPYARD, PuntStatus.CAPTURED):
        return Fraction(0)
    return ware_slot_expected_payout(
        punt.ware, punt.position, _rounds_remaining(state), max(1, occupied), p_safe_if_caught
    )


def _dock_arrival_probs(state: GameState, destination: str) -> List[Numeric]:
    rounds_remaining = _rounds_remaining(state)
    target_status = PuntStatus.IN_PORT if destination == "port" else PuntStatus.IN_SHIPYARD
    probs: List[Numeric] = []
    for punt in state.punts:
        if punt.ware is None:
            continue
        if punt.status == target_status:
            probs.append(1)
        elif punt.status == PuntStatus.ON_ROUTE:
            if destination == "port":
                probs.append(punt_port_probability(punt.position, rounds_remaining))
            else:
                probs.append(punt_shipyard_probability(punt.position, rounds_remaining))
        else:
            probs.append(0)  # docked at the other destination, or captured
    return probs


def expected_accomplice_return(state: GameState, player_id: str, p_safe_if_caught: Numeric = 1) -> Fraction:
    """The sum of gross expected payouts across every accomplice slot
    `player_id` currently occupies -- ware punts, port, shipyard, and the
    pirate boat -- valued gross (no price subtracted) since the price was
    already paid and is already reflected in their current cash.

    Returns 0 once the voyage has settled (`_SETTLED_PHASES`), since by then
    every pending payout is already in players' cash rather than still
    "expected".
    """
    if state.phase in _SETTLED_PHASES:
        return Fraction(0)

    total = Fraction(0)

    for punt in state.punts:
        if punt.ware is None:
            continue
        mine = [s for s in punt.ware_slots if s.occupant == player_id]
        if mine:
            total += len(mine) * _ware_slot_gross_per_accomplice(state, punt, p_safe_if_caught)

    for destination, dock in (("port", state.port), ("shipyard", state.shipyard)):
        arrival_probs = None
        for key, slot in dock.slots.items():
            if slot.occupant != player_id:
                continue
            if arrival_probs is None:
                arrival_probs = _dock_arrival_probs(state, destination)
            total += dock_slot_expected_payout(destination, key, arrival_probs)

    pirate_ids = {"captain": state.pirate_boat.captain.occupant, "second": state.pirate_boat.second.occupant}
    if player_id in pirate_ids.values():
        pirate_count = sum(1 for v in pirate_ids.values() if v is not None)
        punts = [
            (p.ware, p.position, _rounds_remaining(state))
            for p in state.punts
            if p.ware is not None and p.status == PuntStatus.ON_ROUTE
        ]
        total += pirate_expected_payout(punts, pirate_count)

    return total


def wealth_estimate(
    state: GameState, beliefs: ShareBeliefs, player_id: str, p_safe_if_caught: Numeric = 1
) -> Fraction:
    """`beliefs.viewer_id`'s best estimate of `player_id`'s total wealth --
    exact when `player_id == beliefs.viewer_id`, since a player always knows
    their own hand."""
    player = state.player_by_id(player_id)
    if player is None:
        raise ValueError(f"no player with id {player_id!r}")

    return (
        Fraction(player.cash)
        + share_value_estimate(state, beliefs, player_id)
        - encumbered_penalty(player)
        + expected_accomplice_return(state, player_id, p_safe_if_caught)
    )


def identify_rivals(
    state: GameState, beliefs: ShareBeliefs, my_id: str, p_safe_if_caught: Numeric = 1
) -> List[str]:
    """Every opponent `beliefs.viewer_id` (== `my_id`) estimates to be
    wealthier than themselves right now, ordered richest first."""
    my_wealth = wealth_estimate(state, beliefs, my_id, p_safe_if_caught)
    all_wealths = [
        (p.id, wealth_estimate(state, beliefs, p.id, p_safe_if_caught))
        for p in state.players
        if p.id != my_id
    ]
    rivals = [(pid, w) for pid, w in all_wealths if w > my_wealth]
    rivals.sort(key=lambda pair: pair[1], reverse=True)
    return [pid for pid, _ in rivals]


def rev(state: GameState, beliefs: ShareBeliefs, my_id: str, rival_id: str, p_safe_if_caught: Numeric = 1) -> Fraction:
    """Relative Expected Value: the coin gap between a specific rival and
    `my_id`, positive while the rival is still ahead. Can go negative --
    that just means this "rival" no longer is one."""
    return wealth_estimate(state, beliefs, rival_id, p_safe_if_caught) - wealth_estimate(
        state, beliefs, my_id, p_safe_if_caught
    )


def rev_adjusted_score(my_ev_gain: Numeric, rival_ev_gain: Numeric = 0) -> Fraction:
    """Score a candidate action by how it moves the REV gap: your own EV
    gain minus however much it also grows a rival's wealth (pass a negative
    `rival_ev_gain` for an action that actively hurts them, e.g. raising the
    black-market price of a ware they hold heavily). Matches the original
    `ideas.txt` tie-break: prefer the action that most narrows the gap to
    whoever is ahead of you, not just the one with the highest raw EV.
    """
    my_gain = my_ev_gain if isinstance(my_ev_gain, Fraction) else Fraction(my_ev_gain)
    rival_gain = rival_ev_gain if isinstance(rival_ev_gain, Fraction) else Fraction(rival_ev_gain)
    return my_gain - rival_gain
