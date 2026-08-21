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

An opponent's already-placed ware-punt accomplice is valued against its
*projected* final occupancy, not however many slots happen to be filled
right now -- see `project_final_occupancy`: a slot splitting a cargo profit
tends to fill up over the placement rounds since joining an occupied slot
still beats leaving that share on the table, so this assumes full capacity
by default and only falls back to the current count in the last two turns
before the third (final) dice throw, when there's no realistic time left
for another accomplice to join.

This projection isn't specific to valuing *someone else's* slot -- the same
logic applies when deciding whether to take a vacant slot yourself. A punt
that looks great right now (a big-payout ware, decent odds) looks great to
every other EV-seeking bot too, so a slot worth taking alone is usually a
slot several accomplices end up splitting; using "just me" instead of the
projection would systematically overvalue exactly the placements everyone
else also wants. This does mean a not-yet-occupied slot and its
already-occupied neighbor can price out to a similar expected return under
the same projection -- that's not a bug, it reflects that turn order and
slot scarcity, not the EV formula, are what decide who actually claims
which slot once a punt is genuinely worth joining at any point in its fill
sequence.

The pirate boat is deliberately valued differently (`_project_pirate_count`,
not `project_final_occupancy`): with only two possible slots, "will a
second pirate board" has an exact answer rather than a capacity worth
assuming -- they only will if a 2-way split still clears PIRATE_PRICE for
them. Assuming full capacity there regardless would undervalue a pirate
opportunity that's genuinely worth taking solo but not worth sharing, since
no rational second pirate actually boards a split that loses them money.

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
    pirate_slot_ev,
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


def _turns_remaining_in_final_accomplice_round(state: GameState):
    """How many accomplice-placement turns remain this round, counting the
    about-to-act player's own turn -- but only meaningful, so only
    returned, in the round immediately preceding the third (final)
    movement roll: accomplice rounds and dice rolls interleave one-for-one
    (`BoardSetupApp._advance_turn` rolls dice the moment a round's
    placements finish), so that's the one round after which no further
    placement opportunity exists at all before the voyage resolves. Each
    round restarts the rotation at the harbor master
    (`BoardSetupApp._finish_round_end_roll`), so position-in-rotation is
    recoverable from `current_turn_player_id` alone. Returns None outside
    that round or the accomplice-placement phase -- there's a whole extra
    round (or more) still ahead, plenty of time for a slot to fill."""
    if state.phase != Phase.ACCOMPLICE_ROUND or state.movement_round_index != 2 or not state.players:
        return None
    order = state.players
    harbor_master = next((p for p in order if p.is_harbor_master), order[0])
    current_player = state.player_by_id(state.current_turn_player_id) or harbor_master
    if current_player not in order:
        current_player = harbor_master
    position = (order.index(current_player) - order.index(harbor_master)) % len(order)
    return len(order) - position


def project_final_occupancy(state: GameState, current_occupied: int, max_slots: int) -> int:
    """How many accomplices a ware punt is projected to end up with once the
    voyage resolves. Defaults to full capacity -- a slot splitting a cargo
    profit tends to fill up over the placement rounds, since joining an
    already-occupied slot still beats leaving that share on the table --
    except in the last two turns before the third (final) dice throw, where
    so little placement time remains that whatever is occupied right now is
    assumed to be final.

    Pirates don't use this (see `_project_pirate_count`): with only two
    possible slots, whether a second pirate boards has one direct answer --
    they only join if a 2-way split is still worth it to them -- so there's
    an exact check to make rather than a capacity to assume.
    """
    turns_left = _turns_remaining_in_final_accomplice_round(state)
    if turns_left is not None and turns_left <= 2:
        return current_occupied
    return max_slots


def _project_pirate_count(state: GameState, current_pirate_count: int, punts) -> int:
    """How many pirates the boat is projected to end up with. Unlike a ware
    punt's `project_final_occupancy`, this doesn't need to assume full
    capacity and hope it holds up: with only two possible slots, a second
    pirate's own decision is the exact answer to whether they board at all
    -- they will, if and only if splitting every qualifying punt's plunder
    two ways is still worth PIRATE_PRICE to them (`pirate_slot_ev(punts,
    2)`). That's a fact to check, not a growth pattern to project, so a
    solo-but-not-shareable pirate opportunity is correctly valued as
    staying solo rather than being discounted as if a second were coming.
    Still respects the last-two-turns exception: even a profitable second
    boarding needs a placement turn left to actually happen in."""
    if current_pirate_count >= 2:
        return 2
    turns_left = _turns_remaining_in_final_accomplice_round(state)
    if turns_left is not None and turns_left <= 2:
        return current_pirate_count
    if pirate_slot_ev(punts, 2) >= 0:
        return 2
    return max(1, current_pirate_count)


def _ware_slot_gross_per_accomplice(state: GameState, punt, p_safe_if_caught: Numeric) -> Fraction:
    """Gross expected payout *per occupied slot* on a ware punt -- what one
    accomplice on it is owed, given its punt's real status: certain if
    already docked in port, zero if shipwrecked or captured (ware profit
    never pays out there), and otherwise probabilistic, valued against the
    *projected* final occupancy (see `project_final_occupancy`) rather than
    however many slots happen to be filled right now. Callers multiply by
    how many of those slots a specific player holds."""
    occupied = sum(1 for s in punt.ware_slots if s.occupant is not None)
    if punt.status == PuntStatus.IN_PORT:
        return Fraction(PLUNDER_PAYOUTS.get(punt.ware, 0) // max(1, occupied))
    if punt.status in (PuntStatus.IN_SHIPYARD, PuntStatus.CAPTURED):
        return Fraction(0)
    projected = project_final_occupancy(state, occupied, len(punt.ware_slots))
    return ware_slot_expected_payout(
        punt.ware, punt.position, _rounds_remaining(state), max(1, projected), p_safe_if_caught
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
        current_pirate_count = sum(1 for v in pirate_ids.values() if v is not None)
        punts = [
            (p.ware, p.position, _rounds_remaining(state))
            for p in state.punts
            if p.ware is not None and p.status == PuntStatus.ON_ROUTE
        ]
        pirate_count = _project_pirate_count(state, current_pirate_count, punts)
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
