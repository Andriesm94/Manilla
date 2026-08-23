"""Relative-Expected-Value inputs for the harbor-master auction/setup
decision, built on `manilla.engine.wealth` and `manilla.engine.beliefs`.

Per the user, "is winning harbor master worth it, and how much should I
bid" breaks into three separately-heuristic-driven value components:

1. `share_buying_value` -- the harbor master's privilege of buying one
   share at its current black-market price. Coupled with the harbor
   master's own upcoming punt-loading choice (`best_punt_setup`): a
   ware's chance of rising *this* voyage is the real dice-based arrival
   probability of whatever punt position that choice assigns it, and 0
   for the one ware left ashore (it isn't sailing this voyage at all).
   On top of that probabilistic first step, per the user, every ware is
   assumed to rise one further step later in the game regardless
   (`_share_buying_values`) -- there will be many more voyages after this
   one. Special case: while every ware is still at 0 (game start), there's
   no signal to couple with punt positioning at all, so whichever share
   gets bought is just assumed to be worth 20 eventually.
2. `best_punt_setup` -- the harbor master's privilege of choosing which 3
   of 4 wares to load and their start positions. Valued exactly via dice
   probabilities, per the user: both the direct effect on ware-punt-
   accomplice payouts (already covered by `wealth.action_impact` once a
   punt's `ware`/`position` change) and the indirect effect on share
   values from *this voyage's* probabilistic black-market rises
   (`expected_black_market_rise_value`) -- something nothing in
   `wealth.py` accounts for on its own, since `wealth_estimate` only ever
   reads the black market's current, static values.
3. `first_mover_value` -- the value of placing accomplices before
   everyone else this voyage, using a per-player-per-voyage random cost
   coefficient the caller generates and supplies (deliberately not this
   module's job -- see `first_mover_value`'s docstring).

`harbor_master_value` sums all three into what winning the auction is
worth to `my_id` this voyage; compare that against how much more bidding
would cost to decide whether, and how much, to bid.

Of course there's a lot of heuristics going on here, per the user's own
framing -- these are deliberately simple, biddable numbers, not a full
game-tree search of the auction.
"""

from __future__ import annotations

from fractions import Fraction
from typing import Callable, Dict, List, Optional, Tuple

from manilla.engine.beliefs import ShareBeliefs, assumed_share_count
from manilla.engine.expected_value import Numeric, punt_port_probability
from manilla.engine.models import (
    BLACK_MARKET_LEVELS,
    GameState,
    MAX_START_SPACE,
    PUNT_START_SUM,
    Punt,
    PuntStatus,
    Ware,
)
from manilla.engine.wealth import action_impact, identify_rivals


# ------------------------------------------------------------------
# 1. Share-buying value
# ------------------------------------------------------------------


def _next_black_market_level(value: int) -> int:
    """One step up the 0-5-10-20-30 track, capped at 30 (the game-ending
    value -- there's nowhere further to project past it)."""
    idx = BLACK_MARKET_LEVELS.index(value)
    return BLACK_MARKET_LEVELS[min(idx + 1, len(BLACK_MARKET_LEVELS) - 1)]


def _share_price_gain(from_level: int, to_level: int) -> int:
    """The change in a share's black-market *price*
    (`BlackMarket.share_price`'s `max(5, level)`) between two track
    levels -- 0 between level 0 and 5 (the 5-peso price floor already
    covers both), 5 between 5 and 10, 10 between 10 and 20 or 20 and 30."""
    return max(5, to_level) - max(5, from_level)


def _share_buying_values(
    state: GameState,
    beliefs: ShareBeliefs,
    my_id: str,
    p_safe_if_caught: Optional[Numeric] = None,
    planned_punt_setup: Optional[Tuple[List[Ware], Dict[Ware, int]]] = None,
) -> Dict[Ware, Fraction]:
    """Per-ware value of the harbor master's privilege of buying one share
    at its current price, for every ware still available to buy -- see
    `share_buying_value` and `best_shares_to_buy`, which both just reduce
    this dict differently (max value / every tied-best ware).

    While every ware sits at 0 (game start), there's nothing to couple
    with punt positioning -- every ware is identically placed -- so this
    just assumes whichever share gets bought is eventually worth 20.

    Otherwise: each ware's chance of rising *this* voyage is 0 unless it's
    one of the 3 wares `best_punt_setup` actually loads (the ware left
    ashore can't rise this voyage at all), and otherwise the real
    dice-based arrival probability of the punt position that choice
    assigns it (`punt_port_probability`) -- not a flat guess. That
    probability is multiplied by the one-step price gain
    (`_share_price_gain`) to get the immediate, this-voyage term. Per the
    user, on top of that: assume every ware also rises one further step
    later in the game regardless, since there will be many more voyages
    after this one -- added as a second, flat (unconditional, not
    itself probability-weighted) copy of that same one-step gain, since
    it's a separate, later event from this voyage's own chance. So each
    ware's total is `(1 + p) * _share_price_gain(current, next_level)`.
    """
    if all(v == 0 for v in state.black_market.values.values()):
        return {
            ware: Fraction(max(0, 20 - state.black_market.share_price(ware)))
            for ware in Ware
            if state.shares_available(ware) > 0
        }

    if planned_punt_setup is None:
        (wares_loaded, positions), _ = best_punt_setup(state, beliefs, my_id, p_safe_if_caught)
    else:
        wares_loaded, positions = planned_punt_setup
    rounds_remaining = state.movement_rounds_total - state.movement_round_index

    values: Dict[Ware, Fraction] = {}
    for ware in Ware:
        if state.shares_available(ware) <= 0:
            continue
        current = state.black_market.values[ware]
        if current >= BLACK_MARKET_LEVELS[-1]:
            continue  # already maxed -- the game would already be over
        gain = _share_price_gain(current, _next_black_market_level(current))
        p = punt_port_probability(positions[ware], rounds_remaining) if ware in wares_loaded else Fraction(0)
        values[ware] = (1 + p) * gain
    return values


def share_buying_value(
    state: GameState,
    beliefs: ShareBeliefs,
    my_id: str,
    p_safe_if_caught: Optional[Numeric] = None,
    planned_punt_setup: Optional[Tuple[List[Ware], Dict[Ware, int]]] = None,
) -> Fraction:
    """Value of the harbor master's privilege of buying one share: the
    best value among every ware still available to buy, per
    `_share_buying_values`. Every value that function produces is already
    non-negative by construction (the black-market track only ever moves
    up), so this never needs to floor a bad deal at 0 -- buying is
    optional, but nothing here can look worse than not buying at all.

    Pass `planned_punt_setup` (the `(wares_loaded, positions)` half of
    `best_punt_setup`'s return) when the caller has already computed it
    this turn, to skip recomputing that expensive (~76-candidate)
    enumeration a second time.
    """
    values = _share_buying_values(state, beliefs, my_id, p_safe_if_caught, planned_punt_setup)
    return max(values.values(), default=Fraction(0))


def best_shares_to_buy(
    state: GameState,
    beliefs: ShareBeliefs,
    my_id: str,
    p_safe_if_caught: Optional[Numeric] = None,
    planned_punt_setup: Optional[Tuple[List[Ware], Dict[Ware, int]]] = None,
) -> List[Ware]:
    """Every available ware tied for the best share-buying value -- empty
    if buying isn't worthwhile at all. Per the user, when several tie, the
    actual purchase should be picked randomly among them; this only
    narrows down that tied-best set, it doesn't do the picking."""
    values = _share_buying_values(state, beliefs, my_id, p_safe_if_caught, planned_punt_setup)
    if not values:
        return []
    best = max(values.values())
    if best <= 0:
        return []
    return [ware for ware, value in values.items() if value == best]


# ------------------------------------------------------------------
# 2. Punt-setup value
# ------------------------------------------------------------------


def apply_punt_setup(wares_loaded: List[Ware], positions: Dict[Ware, int]) -> Callable[[GameState], None]:
    """Build an `action_impact` mutator for loading `wares_loaded` (any 3
    of the 4 wares) onto the punts at `positions`, matching
    `BoardSetupApp._apply_load_and_place`'s state changes exactly (minus
    the UI-only phase/turn bookkeeping, which wealth estimation doesn't
    read)."""

    def _apply(state: GameState) -> None:
        unloaded = next(w for w in Ware if w not in wares_loaded)
        for punt, ware in zip(state.punts, wares_loaded):
            punt.ware = ware
            punt.position = positions[ware]
            punt.status = PuntStatus.ON_ROUTE
            punt.dock_slot = None
            punt.ware_slots = Punt.new(punt.id, ware).ware_slots
        state.unloaded_ware = unloaded

    return _apply


def punt_setup_candidates(state: GameState) -> List[Tuple[List[Ware], Dict[Ware, int]]]:
    """Every legal `(wares_loaded, positions)` combination the harbor
    master could choose: any 3 of the 4 wares, at any start positions
    0..MAX_START_SPACE each summing to PUNT_START_SUM. Every ordered
    assignment of positions to the chosen wares is covered -- which ware
    gets which position matters (it changes both that ware's cargo-payout
    odds and its black-market-rise odds), so this doesn't collapse
    position triples that are the same multiset but a different
    assignment."""
    candidates = []
    for leave_ashore in Ware:
        loaded = [w for w in Ware if w != leave_ashore]
        for a in range(MAX_START_SPACE + 1):
            for b in range(MAX_START_SPACE + 1):
                c = PUNT_START_SUM - a - b
                if 0 <= c <= MAX_START_SPACE:
                    candidates.append((loaded, {loaded[0]: a, loaded[1]: b, loaded[2]: c}))
    return candidates


def expected_black_market_rise_value(state: GameState, beliefs: ShareBeliefs, player_id: str) -> Fraction:
    """Expected gain in `player_id`'s share value from *this voyage's*
    probabilistic black-market rises
    (`BoardSetupApp._raise_ware_values_for_arrivals`): each loaded, still-
    on-route ware whose punt reaches port this voyage rises exactly one
    step on the 0-5-10-20-30 track. For each such punt, its arrival
    probability (`punt_port_probability`) times the coin value that rise
    would add to `player_id`'s expected holdings of that ware
    (`beliefs.assumed_share_count`) is summed across every loaded ware.
    Nothing in `wealth.wealth_estimate` accounts for this on its own --
    it only ever reads the black market's current, static values."""
    rounds_remaining = state.movement_rounds_total - state.movement_round_index
    total = Fraction(0)
    for punt in state.punts:
        if punt.ware is None or punt.status != PuntStatus.ON_ROUTE:
            continue
        current = state.black_market.values[punt.ware]
        if current >= BLACK_MARKET_LEVELS[-1]:
            continue  # already at 30 -- the game would already be over
        price_gain = _share_price_gain(current, _next_black_market_level(current))
        p_arrival = punt_port_probability(punt.position, rounds_remaining)
        holdings = assumed_share_count(state, beliefs, player_id, punt.ware)
        total += p_arrival * price_gain * holdings
    return total


def best_punt_setup(
    state: GameState,
    beliefs: ShareBeliefs,
    my_id: str,
    p_safe_if_caught: Optional[Numeric] = None,
) -> Tuple[Tuple[List[Ware], Dict[Ware, int]], Fraction]:
    """The best `(wares_loaded, positions)` combination and its score:
    `action_impact`'s `total_rev_after` (the direct ware-punt-accomplice
    effect) plus the *net* expected black-market-rise effect --
    `expected_black_market_rise_value` for `my_id` minus the same for
    every rival identified from the board as it stands now -- keeping the
    same "sum my advantage over every current rival" shape
    `total_rev_after` already uses.
    """
    rivals = identify_rivals(state, beliefs, my_id, p_safe_if_caught)

    best_candidate = None
    best_score = None
    for wares_loaded, positions in punt_setup_candidates(state):
        mutator = apply_punt_setup(wares_loaded, positions)
        impact = action_impact(state, beliefs, my_id, mutator, p_safe_if_caught)

        after = GameState.from_dict(state.to_dict())
        mutator(after)
        rise_score = expected_black_market_rise_value(after, beliefs, my_id)
        for rival_id in rivals:
            rise_score -= expected_black_market_rise_value(after, beliefs, rival_id)

        score = impact.total_rev_after + rise_score
        if best_score is None or score > best_score:
            best_score = score
            best_candidate = (wares_loaded, positions)

    return best_candidate, best_score


# ------------------------------------------------------------------
# 3. First-mover value
# ------------------------------------------------------------------


def _spots_behind(turn_order: List[str], harbor_master_id: str, player_id: str) -> int:
    """How many turn-order spots `player_id` sits behind
    `harbor_master_id` once the placement rotation restarts at the harbor
    master (0 if `player_id` *is* the harbor master)."""
    hm_idx = turn_order.index(harbor_master_id)
    p_idx = turn_order.index(player_id)
    return (p_idx - hm_idx) % len(turn_order)


def _nearest_active_bidder_after(turn_order: List[str], my_id: str, active_bidder_ids: List[str]) -> Optional[str]:
    """The first player, scanning forward from just after `my_id` in
    `turn_order` (wrapping around), who is still in `active_bidder_ids`.
    `None` if no one else is still bidding."""
    if my_id not in turn_order:
        return None
    start = turn_order.index(my_id)
    n = len(turn_order)
    for offset in range(1, n):
        candidate = turn_order[(start + offset) % n]
        if candidate != my_id and candidate in active_bidder_ids:
            return candidate
    return None


def first_mover_value(
    turn_order: List[str],
    my_id: str,
    active_bidder_ids: List[str],
    late_cost_per_spot: Numeric,
) -> Fraction:
    """The coin-equivalent value, to `my_id`, of winning the harbor-master
    auction specifically from the first-mover advantage of placing
    accomplices before everyone else this voyage.

    `late_cost_per_spot` is a per-voyage, per-player random cost
    coefficient (the user: "a random floating point cost ... between 0.5
    and 2") representing the disadvantage of placing one turn-order spot
    later than the harbor master -- generating and persisting it for a
    voyage is the *caller's* job, not this function's, since it needs to
    stay stable across every decision made during that voyage, not be
    redrawn per call.

    Per the user's worst-case heuristic: if `my_id` passes instead of
    winning, assume the nearest still-active bidder after them in
    `turn_order` (wrapping around) becomes harbor master instead --
    that's close to the worst possible outcome for `my_id`'s own position,
    since the placement rotation then restarts right behind them. The
    value is that assumed outcome's cost: `late_cost_per_spot` times how
    many spots behind the new harbor master `my_id` would sit, times 3
    (one hit per accomplice placed, three per player per voyage). 0 if no
    one else is still bidding (passing wouldn't cost this component
    anything -- there's no one left to hand the title to).
    """
    nearest = _nearest_active_bidder_after(turn_order, my_id, active_bidder_ids)
    if nearest is None:
        return Fraction(0)
    spots = _spots_behind(turn_order, nearest, my_id)
    return Fraction(late_cost_per_spot) * spots * 3


def _neutral_punt_setup(state: GameState) -> Tuple[List[Ware], Dict[Ware, int]]:
    """A non-strategic baseline punt setup -- the first 3 wares in `Ware`'s
    definition order, at roughly equal positions summing to
    `PUNT_START_SUM`. Used only to baseline `best_punt_setup`'s score down
    to a genuine marginal value -- see `harbor_master_static_value`."""
    loaded = list(Ware)[:3]
    base, remainder = divmod(PUNT_START_SUM, 3)
    positions = {loaded[0]: base + remainder, loaded[1]: base, loaded[2]: base}
    return loaded, positions


def _punt_setup_score(
    state: GameState,
    beliefs: ShareBeliefs,
    my_id: str,
    wares_loaded: List[Ware],
    positions: Dict[Ware, int],
    rivals: List[str],
    p_safe_if_caught: Optional[Numeric],
) -> Fraction:
    """`action_impact`'s `total_rev_after` for this one `(wares_loaded,
    positions)` choice, plus its net expected black-market-rise effect --
    the same per-candidate scoring formula `best_punt_setup` uses, factored
    out so `harbor_master_static_value` can apply it to a baseline
    candidate too."""
    mutator = apply_punt_setup(wares_loaded, positions)
    impact = action_impact(state, beliefs, my_id, mutator, p_safe_if_caught)

    after = GameState.from_dict(state.to_dict())
    mutator(after)
    rise_score = expected_black_market_rise_value(after, beliefs, my_id)
    for rival_id in rivals:
        rise_score -= expected_black_market_rise_value(after, beliefs, rival_id)

    return impact.total_rev_after + rise_score


def harbor_master_static_value(
    state: GameState,
    beliefs: ShareBeliefs,
    my_id: str,
    p_safe_if_caught: Optional[Numeric] = None,
) -> Fraction:
    """The board-dependent components of `harbor_master_value` --
    `share_buying_value` plus the *marginal* value of `best_punt_setup`'s
    choice -- deliberately split out because neither one changes during a
    single auction (only who's bidding what does): the board itself
    doesn't move while people bid on it. `best_punt_setup` enumerates
    dozens of candidates and is the expensive part of this whole module by
    a wide margin, so compute this once per player per auction and reuse
    it across every bid decision (`decide_harbor_master_bid`'s
    `precomputed_static_value`) rather than recomputing it on every single
    one-step raise.

    "Marginal" matters here: `best_punt_setup`'s raw score is
    `action_impact`'s `total_rev_after`, an *absolute* post-action
    comparison against every current rival, which (per
    `wealth.DEFENSIVE_WEALTH_MARGIN`) carries a large, roughly constant
    negative offset that has nothing to do with punt-setup skill -- it's
    there whether you choose brilliantly or not at all. Comparing that
    absolute figure against a real bid price in PESOS would make bidding
    look worthless even when the underlying board is perfectly biddable
    (with several rivals, the offset alone can swamp any genuine gain).
    Subtracting the same score for a non-strategic baseline setup
    (`_neutral_punt_setup`) cancels that offset -- it's present in both
    figures identically -- leaving the genuine value of choosing well over
    choosing arbitrarily, which *is* directly comparable to a bid price.
    """
    rivals = identify_rivals(state, beliefs, my_id, p_safe_if_caught)

    best_candidate, punt_setup_score = best_punt_setup(state, beliefs, my_id, p_safe_if_caught)
    neutral_loaded, neutral_positions = _neutral_punt_setup(state)
    neutral_score = _punt_setup_score(
        state, beliefs, my_id, neutral_loaded, neutral_positions, rivals, p_safe_if_caught
    )

    # Reuse best_candidate as share_buying_value's planned_punt_setup
    # instead of letting it recompute best_punt_setup a second time --
    # see share_buying_value's docstring.
    buying_value = share_buying_value(state, beliefs, my_id, p_safe_if_caught, planned_punt_setup=best_candidate)
    return buying_value + (punt_setup_score - neutral_score)


def harbor_master_value(
    state: GameState,
    beliefs: ShareBeliefs,
    my_id: str,
    turn_order: List[str],
    active_bidder_ids: List[str],
    late_cost_per_spot: Numeric,
    p_safe_if_caught: Optional[Numeric] = None,
) -> Fraction:
    """Total value, to `my_id`, of winning the harbor-master auction this
    voyage: the sum of all three components above. Compare this against
    how much more bidding would cost (the gap between the current highest
    bid and what `my_id` would need to bid to win) to decide whether, and
    how much, to bid.
    """
    static_value = harbor_master_static_value(state, beliefs, my_id, p_safe_if_caught)
    return static_value + first_mover_value(turn_order, my_id, active_bidder_ids, late_cost_per_spot)


def decide_harbor_master_bid(
    state: GameState,
    beliefs: ShareBeliefs,
    my_id: str,
    turn_order: List[str],
    active_bidder_ids: List[str],
    current_highest_bid: int,
    late_cost_per_spot: Numeric,
    p_safe_if_caught: Optional[Numeric] = None,
    precomputed_static_value: Optional[Numeric] = None,
) -> Optional[int]:
    """Whether, and how much, `my_id` should bid right now: `None` to
    pass, or `current_highest_bid + 1` to raise by exactly one step. Per
    the user, this always increments by a single step rather than jumping
    straight to some computed "maximum worth it" bid -- call it again
    fresh every time it's `my_id`'s turn to act (recalibrating against
    whatever `active_bidder_ids` looks like *then*, since it shrinks every
    time someone passes) rather than reusing an earlier answer.

    Pass `precomputed_static_value` (see `harbor_master_static_value`)
    once computed for `my_id` in this auction to skip recomputing the
    expensive, unchanging share/punt-setup components on every bid --
    only the cheap `first_mover_value` term genuinely needs to be fresh
    each call, since it's the only one that depends on who's still
    bidding.

    Bids the next step exactly when the total value clears it --
    `> current_highest_bid + 1`, not `>=`, so a bid that would only break
    even isn't taken (there's no benefit to winning at a price where the
    value gained equals the price paid, and every further step raises the
    price without raising the value).
    """
    static_value = (
        harbor_master_static_value(state, beliefs, my_id, p_safe_if_caught)
        if precomputed_static_value is None
        else precomputed_static_value
    )
    value = static_value + first_mover_value(turn_order, my_id, active_bidder_ids, late_cost_per_spot)
    next_bid = current_highest_bid + 1
    return next_bid if value > next_bid else None
