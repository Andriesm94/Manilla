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

The pirate boat gets no such projection: its occupancy is valued exactly as
it stands right now, whether that's 0, 1, or 2 pirates. Whether a second
pirate eventually boards is entirely their own future call, made with their
own up-to-date numbers at the time -- not something worth guessing at here.
`expected_accomplice_return` recomputes fresh from the live state every
call, so if a second pirate does join later, the estimate simply reflects
that the moment it's observed.

A **rival** is any opponent whose estimated wealth exceeds the viewer's own;
REV is the coin gap to a specific rival (`rival_wealth_est - my_wealth_est`,
positive when they're ahead). `action_impact` turns this into "pick the
action that best closes the gap to every current rival at once": it
simulates a candidate action (any caller-supplied `GameState` mutation, from
placing a pirate to a pilot's nudge) and reports how it moves your own
wealth and every current rival's, plus the total post-action advantage
summed across them (`sum(my_wealth_after - rival_wealth_after)`, per the
user's own formula) -- the larger that total, the better the action is for
your standing against the field, not just for your own raw EV.

An action doesn't only change the acting player's own numbers. Placing a
pirate makes every currently-loaded punt genuinely vulnerable to plunder
(`pirate_threat`), which cuts into whichever opponents hold ware-punt
accomplices there; nudging a punt's position with a pilot shifts its
arrival odds for whoever holds accomplices on *that* punt. `p_safe_if_caught`
now defaults to `None` everywhere in this module, meaning "derive it from
whether the pirate boat is actually occupied" rather than a flat assumed
constant -- callers can still pass an explicit value for a counterfactual
("what if I ignore pirates"), but the default reflects the real board.

Valuing an accomplice slot that grants a *future choice* -- the pilot
island -- means looking ahead to that choice rather than treating the slot
itself as the action. `pilot_slot_value` enumerates every move a small or
large pilot could make (including doing nothing) and reports the best
`total_rev_after` among them, via `best_pilot_move`. That same function is
meant to be called again, fresh, once the pilot phase actually arrives
(`BoardSetupApp._show_pilot_dialogs`, right before the third movement
round) -- punt positions will very likely have shifted since the
accomplice was placed, so whatever looked best back then is a starting
estimate, not a plan to execute unquestioned.

An occupied pilot slot isn't just a threat to price into your own EV -- it
represents a move that hasn't happened yet but will, and every other
player's own decisions should account for it. `with_predicted_pilot_moves`
predicts what each currently-seated pilot will do by running the exact same
`best_pilot_move` logic *as if the predictor were that player*: their own
beliefs (`infer_beliefs` anchored to them, not you -- their own shares
exactly known, everyone else's, including yours, averaged from the secret
pool the same way any belief is inferred), their own rivals, their own
best move. This is one prediction layer deep, not a hall of mirrors: the
predicted pilot doesn't itself further account for anyone predicting them.
Whoever is about to place an accomplice of their own should run this first
and reason against the *predicted* board, not the literal current one --
the pilot isn't going to wait for anyone else to notice it.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass, field
from fractions import Fraction
from typing import Callable, Dict, Iterable, List, Optional, Tuple

from manilla.engine.beliefs import ShareBeliefs, ShareSignal, infer_beliefs, share_value_estimate
from manilla.engine.expected_value import (
    Numeric,
    dock_slot_expected_payout,
    pirate_expected_payout,
    punt_port_probability,
    punt_shipyard_probability,
    ware_slot_expected_payout,
)
from manilla.engine.models import (
    GameState,
    Phase,
    PIRATE_PRICE,
    PLUNDER_PAYOUTS,
    Player,
    PuntStatus,
    SEA_ROUTE_LENGTH,
    SHARE_REPAY_AMOUNT,
)

# Once a voyage reaches these phases every pending accomplice payout has
# already been folded into players' cash (or is about to be, atomically),
# so counting it again here would double it.
_SETTLED_PHASES = (Phase.PROFIT_DISTRIBUTION, Phase.WARE_RISE)


def encumbered_penalty(player: Player) -> int:
    """The flat redemption cost of a player's encumbered shares -- what it
    would take to unencumber every one of them right now."""
    return SHARE_REPAY_AMOUNT * len(player.encumbered_shares)


def pirate_threat(state: GameState) -> Fraction:
    """The realistic `p_safe_if_caught` for the board as it actually
    stands: 0 (certain plunder) if any pirate slot is occupied -- plunder
    is automatic and mandatory once pirates are aboard, no board-or-skip
    choice involved (see `expected_value`'s module docstring) -- else 1 (no
    pirates present, so a punt caught on 13 always docks safely)."""
    pb = state.pirate_boat
    return Fraction(0) if (pb.captain.occupant or pb.second.occupant) else Fraction(1)


def _resolve_p_safe(state: GameState, p_safe_if_caught: Optional[Numeric]) -> Numeric:
    return pirate_threat(state) if p_safe_if_caught is None else p_safe_if_caught


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

    Pirates don't use this -- see the module docstring: their occupancy is
    valued as-is, with no projection at all.
    """
    turns_left = _turns_remaining_in_final_accomplice_round(state)
    if turns_left is not None and turns_left <= 2:
        return current_occupied
    return max_slots


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


def expected_accomplice_return(
    state: GameState, player_id: str, p_safe_if_caught: Optional[Numeric] = None
) -> Fraction:
    """The sum of gross expected payouts across every accomplice slot
    `player_id` currently occupies -- ware punts, port, shipyard, and the
    pirate boat -- valued gross (no price subtracted) since the price was
    already paid and is already reflected in their current cash.

    `p_safe_if_caught` defaults to `None`, meaning "derive it from whether
    the pirate boat is actually occupied" (`pirate_threat`) rather than
    assume a fixed value -- pass an explicit override only for a
    counterfactual.

    Returns 0 once the voyage has settled (`_SETTLED_PHASES`), since by then
    every pending payout is already in players' cash rather than still
    "expected".
    """
    if state.phase in _SETTLED_PHASES:
        return Fraction(0)

    p_safe_if_caught = _resolve_p_safe(state, p_safe_if_caught)
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
        total += pirate_expected_payout(punts, current_pirate_count)

    return total


def wealth_estimate(
    state: GameState, beliefs: ShareBeliefs, player_id: str, p_safe_if_caught: Optional[Numeric] = None
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
    state: GameState, beliefs: ShareBeliefs, my_id: str, p_safe_if_caught: Optional[Numeric] = None
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


def rev(
    state: GameState,
    beliefs: ShareBeliefs,
    my_id: str,
    rival_id: str,
    p_safe_if_caught: Optional[Numeric] = None,
) -> Fraction:
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


@dataclass
class ActionImpact:
    """What a candidate action does to `my_id` and to every current rival,
    per `action_impact`. `total_rev_after` is the score to maximize when
    comparing candidate actions -- see `action_impact`."""

    my_gain: Fraction
    rival_gains: Dict[str, Fraction] = field(default_factory=dict)
    total_rev_after: Fraction = Fraction(0)


def action_impact(
    state: GameState,
    beliefs: ShareBeliefs,
    my_id: str,
    apply_action: Callable[[GameState], None],
    p_safe_if_caught: Optional[Numeric] = None,
) -> ActionImpact:
    """Simulate taking `apply_action` and report how it moves everyone's
    estimated wealth.

    Rivals are identified once, from the board as it stands *before* the
    action (they're who you're actually trying to get ahead of right now).
    `apply_action` receives a full copy of `state` (a real
    `GameState.from_dict(state.to_dict())` clone, so mutating it never
    touches the original) and should mutate it to represent the action --
    e.g. occupying a slot and deducting its price. `wealth_estimate` is
    then recomputed for `my_id` and every one of those rivals against the
    mutated copy.

    `total_rev_after` is `sum(my_wealth_after - rival_wealth_after)` over
    that same rival set -- the user's own formula for "advantage over the
    field": the action that *maximizes* this is the one that most widens
    (or best reverses) your combined gap to everyone currently ahead of
    you, which is not always the same action that maximizes your own raw
    EV gain (`my_gain`) -- an action that helps you a little while hurting
    a rival a lot can score better here than one that helps you more but
    leaves every rival untouched.
    """
    rivals = identify_rivals(state, beliefs, my_id, p_safe_if_caught)
    my_before = wealth_estimate(state, beliefs, my_id, p_safe_if_caught)
    rival_before = {r: wealth_estimate(state, beliefs, r, p_safe_if_caught) for r in rivals}

    after = GameState.from_dict(state.to_dict())
    apply_action(after)

    my_after = wealth_estimate(after, beliefs, my_id, p_safe_if_caught)
    rival_gains: Dict[str, Fraction] = {}
    total_rev_after = Fraction(0)
    for r in rivals:
        w_after = wealth_estimate(after, beliefs, r, p_safe_if_caught)
        rival_gains[r] = w_after - rival_before[r]
        total_rev_after += my_after - w_after

    return ActionImpact(my_gain=my_after - my_before, rival_gains=rival_gains, total_rev_after=total_rev_after)


def apply_ware_slot_placement(punt_id: int, player_id: str) -> Callable[[GameState], None]:
    """Build an `action_impact` mutator for placing `player_id` on punt
    `punt_id`'s cheapest vacant ware slot, paying that slot's price --
    matching `BoardSetupApp._place_or_remove_punt_accomplice`'s mandatory
    cheapest-first rule. No-ops if the punt doesn't exist or has no vacant
    slot.

    Joining a punt a rival already occupies does not, by itself, show up
    as reducing their `rival_gains` in `action_impact` -- and that's
    correct, not a gap: `project_final_occupancy` already valued their
    slot assuming the punt fills up regardless (see the module docstring),
    so taking another slot doesn't change what they were already assumed
    to get. It only shows up as a real cost to them in the last two turns
    before the third dice throw, where `project_final_occupancy` switches
    to actual occupancy -- by then there's no more room for "it would have
    filled anyway" to still be true, so an actual join is a real dilution.
    """

    def _apply(state: GameState) -> None:
        punt = next((p for p in state.punts if p.id == punt_id), None)
        if punt is None:
            return
        vacant = [s for s in punt.ware_slots if s.occupant is None]
        if not vacant:
            return
        cheapest = min(vacant, key=lambda s: s.price)
        cheapest.occupant = player_id
        player = state.player_by_id(player_id)
        if player is not None:
            player.cash -= cheapest.price

    return _apply


def apply_pirate_placement(role: str, player_id: str) -> Callable[[GameState], None]:
    """Build an `action_impact` mutator for placing `player_id` on the
    pirate boat's `role` ('captain' or 'second') slot, paying PIRATE_PRICE.
    This alone is often enough to make `pirate_threat` flip from 1 to 0 for
    every other player's wealth estimate -- placing the *first* pirate is
    what newly endangers everyone's ware-punt accomplices, not specifically
    the second."""
    if role not in ("captain", "second"):
        raise ValueError(f"role must be 'captain' or 'second', got {role!r}")

    def _apply(state: GameState) -> None:
        slot = state.pirate_boat.captain if role == "captain" else state.pirate_boat.second
        slot.occupant = player_id
        player = state.player_by_id(player_id)
        if player is not None:
            player.cash -= PIRATE_PRICE

    return _apply


def apply_pilot_move(punt_id: int, delta: int) -> Callable[[GameState], None]:
    """Build an `action_impact` mutator applying a pilot's positional nudge
    to punt `punt_id` by `delta` spaces, matching
    `BoardSetupApp._apply_pilot_move`'s clamp-at-0 and
    overshoot-docks-immediately rules (a small pilot moves one punt by
    ±1, a large pilot moves one punt by ±2 or two punts by ±1 each -- call
    this once per punt moved). No-ops if the punt isn't `ON_ROUTE`. Doesn't
    charge a price: piloting is a privilege of an already-placed (and
    already-paid-for) pilot accomplice, not a separate payment per move.
    This only reproduces what wealth estimation actually reads (position
    and status) -- it doesn't assign a port/shipyard dock_slot letter on
    overshoot, since no wealth calculation depends on which letter a punt
    ends up docked at, only on whether it did.
    """

    def _apply(state: GameState) -> None:
        punt = next((p for p in state.punts if p.id == punt_id), None)
        if punt is None or punt.status != PuntStatus.ON_ROUTE:
            return
        punt.position = max(0, punt.position + delta)
        if punt.position > SEA_ROUTE_LENGTH:
            punt.status = PuntStatus.IN_PORT
            punt.position = SEA_ROUTE_LENGTH

    return _apply


def _combine_actions(*mutators: Callable[[GameState], None]) -> Callable[[GameState], None]:
    def _apply(state: GameState) -> None:
        for mutator in mutators:
            mutator(state)

    return _apply


def pilot_move_candidates(state: GameState, pilot_size: str) -> List[Callable[[GameState], None]]:
    """Every legal pilot move for `pilot_size` ('small' or 'large'), as
    ready-to-use `action_impact` mutators, given whichever punts are
    currently `ON_ROUTE` -- plus skipping (a no-op), always included as a
    candidate, since not moving anything can be the right call. A small
    pilot moves one eligible punt by ±1; a large pilot moves one punt by
    ±2, or two *different* punts by ±1 each
    (`BoardSetupApp._show_large_pilot_dialog`)."""
    if pilot_size not in ("small", "large"):
        raise ValueError(f"pilot_size must be 'small' or 'large', got {pilot_size!r}")

    eligible = [p.id for p in state.punts if p.ware is not None and p.status == PuntStatus.ON_ROUTE]
    candidates: List[Callable[[GameState], None]] = [lambda s: None]

    if pilot_size == "small":
        for punt_id in eligible:
            candidates.append(apply_pilot_move(punt_id, 1))
            candidates.append(apply_pilot_move(punt_id, -1))
    else:
        for punt_id in eligible:
            candidates.append(apply_pilot_move(punt_id, 2))
            candidates.append(apply_pilot_move(punt_id, -2))
        for punt_a, punt_b in itertools.combinations(eligible, 2):
            for delta_a, delta_b in itertools.product((1, -1), repeat=2):
                candidates.append(
                    _combine_actions(apply_pilot_move(punt_a, delta_a), apply_pilot_move(punt_b, delta_b))
                )

    return candidates


def best_pilot_move(
    state: GameState,
    beliefs: ShareBeliefs,
    my_id: str,
    pilot_size: str,
    p_safe_if_caught: Optional[Numeric] = None,
) -> Tuple[Callable[[GameState], None], ActionImpact]:
    """The best pilot move currently available for `pilot_size`, and its
    `ActionImpact` -- ranked by `total_rev_after` (maximize), the same
    metric every other REV-based decision in this module uses. Ties keep
    whichever candidate was considered first (skipping, then single-punt
    moves, then two-punt combinations), so indifference defaults toward
    doing less.

    Call this fresh wherever a real decision is needed -- see the module
    docstring: the answer depends entirely on the punt positions in
    `state` right now, and is never assumed to still hold once the board
    has moved on.
    """
    candidates = pilot_move_candidates(state, pilot_size)
    best_action = candidates[0]
    best_impact = action_impact(state, beliefs, my_id, best_action, p_safe_if_caught)
    for action in candidates[1:]:
        impact = action_impact(state, beliefs, my_id, action, p_safe_if_caught)
        if impact.total_rev_after > best_impact.total_rev_after:
            best_action, best_impact = action, impact
    return best_action, best_impact


def pilot_slot_value(
    state: GameState,
    beliefs: ShareBeliefs,
    my_id: str,
    pilot_size: str,
    p_safe_if_caught: Optional[Numeric] = None,
) -> Fraction:
    """The value of placing an accomplice on the pilot island's
    `pilot_size` slot: the best `total_rev_after` achievable among every
    move currently available (including skipping) -- see `best_pilot_move`.
    Gross: doesn't net the slot's own price
    (`DEFAULT_PILOT_PRICES[pilot_size]`), since `total_rev_after` is a
    wealth-comparison figure across every rival, not a single payout with
    an obvious price to net it against.

    This is a placement-time estimate, not a locked-in plan -- punt
    positions will likely have moved by the time the pilot phase actually
    arrives, so `best_pilot_move` needs to be re-run against the live
    board to pick the real move, rather than reusing whatever was best
    here.
    """
    _, impact = best_pilot_move(state, beliefs, my_id, pilot_size, p_safe_if_caught)
    return impact.total_rev_after


def occupied_pilot_slots(state: GameState) -> List[Tuple[str, str]]:
    """`[(pilot_size, player_id), ...]` for every currently occupied pilot
    slot, small before large (the order `BoardSetupApp._show_pilot_dialogs`
    resolves them in)."""
    slots = []
    if state.pilot_island.small.occupant:
        slots.append(("small", state.pilot_island.small.occupant))
    if state.pilot_island.large.occupant:
        slots.append(("large", state.pilot_island.large.occupant))
    return slots


def predict_pilot_move(
    state: GameState,
    pilot_player_id: str,
    pilot_size: str,
    signals: Iterable[ShareSignal] = (),
    p_safe_if_caught: Optional[Numeric] = None,
) -> Callable[[GameState], None]:
    """Predict what `pilot_player_id` will do with the `pilot_size` pilot
    they occupy, by running `best_pilot_move` exactly as they would run it
    themselves: their own beliefs (`infer_beliefs(state, pilot_player_id,
    signals)` -- their own shares exactly known, everyone else's averaged
    from the secret pool, the same inference any viewer gets), their own
    rivals, their own best move. `signals` should be whatever public
    share-reveal events (harbor-master purchases, forward punt starts) are
    already known -- see `beliefs.infer_beliefs` -- since those are common
    knowledge, not something specific to the predictor.

    Returns the predicted mutator, ready to feed into `action_impact` or
    apply directly to a state clone -- see `with_predicted_pilot_moves` for
    folding every currently-seated pilot's prediction in at once.
    """
    pilot_beliefs = infer_beliefs(state, pilot_player_id, signals)
    action, _ = best_pilot_move(state, pilot_beliefs, pilot_player_id, pilot_size, p_safe_if_caught)
    return action


def with_predicted_pilot_moves(
    state: GameState,
    signals: Iterable[ShareSignal] = (),
    p_safe_if_caught: Optional[Numeric] = None,
) -> GameState:
    """A clone of `state` with every currently-occupied pilot slot's
    predicted move already applied (see `predict_pilot_move`). Use this as
    the starting board for your own wealth/REV calculations whenever
    anyone -- including yourself -- already holds a pilot accomplice: their
    move isn't waiting on you noticing it, so reasoning against the literal
    current board would be reasoning against a state that's about to
    change out from under you.

    If both pilot slots are occupied, small is predicted and applied
    before large is even evaluated, matching the order they actually
    resolve in and letting the large pilot's prediction see the small
    pilot's (predicted) result, not the untouched board.
    """
    predicted = GameState.from_dict(state.to_dict())
    for pilot_size, player_id in occupied_pilot_slots(predicted):
        action = predict_pilot_move(predicted, player_id, pilot_size, signals, p_safe_if_caught)
        action(predicted)
    return predicted
