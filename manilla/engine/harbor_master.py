"""Relative-Expected-Value inputs for the harbor-master auction/setup
decision, built on `manilla.engine.wealth` and `manilla.engine.beliefs`.

Per the user, "is winning harbor master worth it, and how much should I
bid" breaks into three separately-heuristic-driven value components:

1. The harbor master's privilege of buying one share at its current
   black-market price. As of 2026-08-31 this is priced by the learned
   model (`manilla.engine.share_model`) via `harbor_master_bid_context`,
   which asks it for the *exact* purchase it would make against this
   voyage's planned punt setup and values that purchase's net gain. So a
   bidder knows precisely which share it is bidding for, and buying it
   later cannot contradict the bid.

   The older heuristic (`share_buying_value` / `_share_buying_values`) is
   still here: it's the baseline the model was measured against, and it
   documents the reasoning the model now does statistically -- a ware's
   chance of rising *this* voyage is the real dice-based arrival
   probability of whatever punt position the setup assigns it, 0 for the
   ware left ashore, plus a flat assumed further rise later in the game.
   Nothing in the live path calls it any more.
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
   everyone else this voyage, priced from the *measured* mean earnings of
   each seat in the turn rotation (`manilla.engine.seat_value`) rather
   than the hand-picked random coefficient this used to take. The caller
   supplies the seat table, computed once per game. Valued as a worst
   case over the field: whichever still-active rival would leave `my_id`
   in the lowest-earning seat is assumed to take the office if `my_id`
   passes.

`harbor_master_value` sums all three into what winning the auction is
worth to `my_id` this voyage; compare that against how much more bidding
would cost to decide whether, and how much, to bid.

Of course there's a lot of heuristics going on here, per the user's own
framing -- these are deliberately simple, biddable numbers, not a full
game-tree search of the auction.
"""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from typing import Callable, Dict, List, Optional, Sequence, Tuple

from manilla.engine.beliefs import ShareBeliefs, assumed_share_count
from manilla.engine.expected_value import Numeric, punt_port_probability
from manilla.engine.models import (
    BLACK_MARKET_LEVELS,
    GameState,
    MAX_START_SPACE,
    PUNT_START_SUM,
    Punt,
    PuntStatus,
    SHARE_LOAN_AMOUNT,
    SHARE_REPAY_AMOUNT,
    Ware,
)
from manilla.engine import share_model
from manilla.engine.seat_value import seat_advantage
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
    narrows down that tied-best set, it doesn't do the picking.

    **No longer drives the live purchase** (2026-08-31): both
    `selfplay._run_buy_share` and `BoardSetupApp._bot_buy_share` now choose
    through `manilla.engine.share_model`, which predicts each ware's final
    black-market level from self-play data and beats this heuristic by a
    wide margin (54.4% top-pick accuracy against 42.9%). Kept because
    `_share_buying_values` underneath it still prices the share-buying
    *component of a bid* -- see `harbor_master_bid_context` -- and as the
    baseline the model is measured against."""
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


def _rival_bidders(turn_order: List[str], my_id: str, active_bidder_ids: List[str]) -> List[str]:
    """Everyone other than `my_id` who is still bidding and has a place in
    `turn_order` -- i.e. every player who could still take the office if
    `my_id` drops out."""
    if my_id not in turn_order:
        return []
    return [pid for pid in turn_order if pid != my_id and pid in active_bidder_ids]


def first_mover_value(
    turn_order: List[str],
    my_id: str,
    active_bidder_ids: List[str],
    seat_means: Sequence[Numeric],
) -> Fraction:
    """The coin-equivalent value, to `my_id`, of winning the harbor-master
    auction specifically from the first-mover advantage of placing
    accomplices before everyone else this voyage.

    `seat_means` is the mean per-voyage accomplice earnings of each seat,
    indexed by offset from the harbor master -- see
    `manilla.engine.seat_value`, which measures it from self-play rather
    than assuming it. Computing it is the *caller's* job, once per game,
    so it stays fixed across every decision made during that game.

    Worst-case over the remaining field, per the user: if `my_id` passes
    instead of winning, assume whichever still-active bidder would leave
    them in the *worst* seat takes the office. Every remaining rival is
    considered as a possible winner, `my_id`'s resulting offset behind
    each is priced through `seat_advantage`, and the largest of those is
    the value -- the most `my_id` can lose by dropping out, given who is
    actually still in the auction. 0 if no one else is still bidding
    (passing wouldn't cost this component anything -- there's no one left
    to hand the title to).

    Note this is a maximum over *seat value*, not over seat distance, so
    which rival counts as worst depends on the measured table rather than
    on who sits furthest back. With the current figures the worst landing
    spot is offset 2 (the lowest-earning seat), not offset 3 -- so a rival
    two spots ahead is more threatening than the one directly behind, and
    if that rival has already passed the worst case falls to whoever is
    left. An earlier version assumed the nearest active bidder after
    `my_id` always won, which understated this whenever a further-away
    rival would have been worse.

    This replaced a `late_cost_per_spot * spots * 3` formula built on a
    random 0.5-2.0 coefficient. Two things changed with it: the cost is no
    longer assumed linear in seat distance (the measurement shows a step
    at the harbor master and near-flatness behind it), and the `* 3` is
    gone -- a seat's measured mean already covers the whole voyage, all
    three accomplices included, so multiplying would double-count.
    """
    rivals = _rival_bidders(turn_order, my_id, active_bidder_ids)
    if not rivals:
        return Fraction(0)
    return max(
        Fraction(
            seat_advantage(seat_means, _spots_behind(turn_order, rival, my_id))
        ).limit_denominator(10**6)
        for rival in rivals
    )


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


@dataclass
class HarborMasterBidContext:
    """The board-dependent, bid-to-bid-stable pieces `decide_harbor_master_bid`
    needs -- see `harbor_master_bid_context`, which builds one of these."""

    static_value: Fraction
    preferred_share_price: int
    wares_loaded: List[Ware]


def harbor_master_bid_context(
    state: GameState,
    beliefs: ShareBeliefs,
    my_id: str,
    p_safe_if_caught: Optional[Numeric] = None,
    model: Optional["share_model.ShareValueModel"] = None,
) -> HarborMasterBidContext:
    """The board-dependent, bid-to-bid-stable pieces `decide_harbor_master_bid`
    needs -- `harbor_master_static_value`, the price of whichever share
    `my_id` would go on to buy as harbor master (0 if buying isn't
    worthwhile), and which 3 wares `best_punt_setup` would load this
    voyage. Bundled into one call because all three depend on the same
    expensive `best_punt_setup` enumeration (~76 candidates) and none of
    them change during a single auction (only who's bidding what does,
    and the board itself doesn't move while people bid on it) -- compute
    this once per player per auction and reuse it across every bid
    decision (`decide_harbor_master_bid`'s `precomputed_bid_context`)
    rather than recomputing it on every single one-step raise.

    `static_value`'s "marginal" punt-setup component matters here:
    `best_punt_setup`'s raw score is `action_impact`'s `total_rev_after`,
    an *absolute* post-action comparison against every current rival,
    which (per `wealth.DEFENSIVE_WEALTH_MARGIN`) carries a large, roughly
    constant negative offset that has nothing to do with punt-setup skill
    -- it's there whether you choose brilliantly or not at all. Comparing
    that absolute figure against a real bid price in PESOS would make
    bidding look worthless even when the underlying board is perfectly
    biddable (with several rivals, the offset alone can swamp any genuine
    gain). Subtracting the same score for a non-strategic baseline setup
    (`_neutral_punt_setup`) cancels that offset -- it's present in both
    figures identically -- leaving the genuine value of choosing well over
    choosing arbitrarily, which *is* directly comparable to a bid price.

    The share-buying component comes from the *same* decision the purchase
    itself will make (`share_model.plan_share_purchase`, against the punt
    setup already computed here), so bidding knows exactly which share it
    is bidding for: its value is that purchase's net value, and the
    preferred price is that ware's actual price rather than a worst case
    over ties. These used to disagree -- bidding valued shares with the
    heuristic while buying went through the model -- which meant a bot
    could bid for one ware's prospects and then buy a different one.

    `model` defaults to the shipped share model; pass one to bid against a
    retrained or deliberately different valuation.
    """
    rivals = identify_rivals(state, beliefs, my_id, p_safe_if_caught)

    best_candidate, punt_setup_score = best_punt_setup(state, beliefs, my_id, p_safe_if_caught)
    wares_loaded, _positions = best_candidate
    neutral_loaded, neutral_positions = _neutral_punt_setup(state)
    neutral_score = _punt_setup_score(
        state, beliefs, my_id, neutral_loaded, neutral_positions, rivals, p_safe_if_caught
    )

    purchase = share_model.plan_share_purchase(
        model if model is not None else share_model.default_model(),
        state,
        share_model.favored_wares_from_setup(*best_candidate),
    )
    buying_value = (
        Fraction(purchase.net_value).limit_denominator(10**6) if purchase is not None else Fraction(0)
    )
    preferred_share_price = purchase.price if purchase is not None else 0
    static_value = buying_value + (punt_setup_score - neutral_score)

    return HarborMasterBidContext(static_value, preferred_share_price, wares_loaded)


def harbor_master_static_value(
    state: GameState,
    beliefs: ShareBeliefs,
    my_id: str,
    p_safe_if_caught: Optional[Numeric] = None,
) -> Fraction:
    """The board-dependent components of `harbor_master_value` --
    `share_buying_value` plus the *marginal* value of `best_punt_setup`'s
    choice. See `harbor_master_bid_context` for the full derivation; this
    is a thin wrapper around it for callers that only need the value, not
    the rest of the bundled context."""
    return harbor_master_bid_context(state, beliefs, my_id, p_safe_if_caught).static_value


def at_risk_encumbered_share_count(state: GameState, my_id: str, wares_loaded: List[Ware]) -> int:
    """How many of `my_id`'s encumbered shares are at meaningful risk of
    the game ending while still encumbered -- already sitting at
    black-market level 20 (one step from the game-ending 30), or at 10
    and about to get a push toward 20 because the harbor master's own
    chosen punt setup this voyage loads that ware. Used only to price the
    "assume I'll need to urgently unencumber" liquidity cost during
    bidding -- see `decide_harbor_master_bid`."""
    player = state.player_by_id(my_id)
    if player is None:
        return 0
    count = 0
    for share in player.shares:
        if not share.encumbered:
            continue
        level = state.black_market.values[share.ware]
        if level == 20 or (level == 10 and share.ware in wares_loaded):
            count += 1
    return count


def harbor_master_value(
    state: GameState,
    beliefs: ShareBeliefs,
    my_id: str,
    turn_order: List[str],
    active_bidder_ids: List[str],
    seat_means: Sequence[Numeric],
    p_safe_if_caught: Optional[Numeric] = None,
) -> Fraction:
    """Total value, to `my_id`, of winning the harbor-master auction this
    voyage: the sum of all three components above. Compare this against
    how much more bidding would cost (the gap between the current highest
    bid and what `my_id` would need to bid to win) to decide whether, and
    how much, to bid.
    """
    static_value = harbor_master_static_value(state, beliefs, my_id, p_safe_if_caught)
    return static_value + first_mover_value(turn_order, my_id, active_bidder_ids, seat_means)


def decide_harbor_master_bid(
    state: GameState,
    beliefs: ShareBeliefs,
    my_id: str,
    turn_order: List[str],
    active_bidder_ids: List[str],
    current_highest_bid: int,
    seat_means: Sequence[Numeric],
    p_safe_if_caught: Optional[Numeric] = None,
    precomputed_bid_context: Optional[HarborMasterBidContext] = None,
) -> Optional[int]:
    """Whether, and how much, `my_id` should bid right now: `None` to
    pass, or `current_highest_bid + 1` to raise by exactly one step. Per
    the user, this always increments by a single step rather than jumping
    straight to some computed "maximum worth it" bid -- call it again
    fresh every time it's `my_id`'s turn to act (recalibrating against
    whatever `active_bidder_ids` looks like *then*, since it shrinks every
    time someone passes) rather than reusing an earlier answer.

    Pass `precomputed_bid_context` (see `harbor_master_bid_context`) once
    computed for `my_id` in this auction to skip recomputing the
    expensive, unchanging share/punt-setup components on every bid --
    only the cheap `first_mover_value` term genuinely needs to be fresh
    each call, since it's the only one that depends on who's still
    bidding.

    Bids the next step exactly when the total value clears its cost --
    `> next_bid`, not `>=`, so a bid that would only break even isn't
    taken (there's no benefit to winning at a price where the value
    gained equals the price paid, and every further step raises the price
    without raising the value). Two mutually-exclusive liquidity penalties
    can raise that cost, per the user:

    - If `my_id` holds any encumbered share at meaningful risk of the
      game ending while still encumbered (`at_risk_encumbered_share_count`
      -- already at black-market level 20, or at 10 and about to get a
      push toward 20 because this voyage's own chosen punt setup loads
      that ware), assume each such share will need to be urgently
      unencumbered before the game can end: `SHARE_REPAY_AMOUNT` (15)
      pesos of imaginary liquidity cost, once per at-risk share.
    - Otherwise, if paying `next_bid` *and* then buying the share `my_id`
      would go on to prefer as harbor master
      (`harbor_master_bid_context`'s `preferred_share_price`) would drop
      `my_id`'s cash below 10, assume that means encumbering a share to
      cover it instead, which nets a permanent loss of
      `SHARE_REPAY_AMOUNT - SHARE_LOAN_AMOUNT` (3) pesos (borrow 12 now,
      repay 15 later).

    Per the user, these don't stack: an at-risk encumbered share already
    implies a liquidity concern, so the smaller below-10 cost is skipped
    whenever the bigger one applies. Both are deliberately bidding-only:
    accomplice placement (`policy.py`) doesn't apply either, since a
    low-cash accomplice placement doesn't commit real cash the way an
    auction loss does right now.
    """
    context = (
        harbor_master_bid_context(state, beliefs, my_id, p_safe_if_caught)
        if precomputed_bid_context is None
        else precomputed_bid_context
    )
    value = context.static_value + first_mover_value(turn_order, my_id, active_bidder_ids, seat_means)
    next_bid = current_highest_bid + 1

    my_player = state.player_by_id(my_id)
    cost = next_bid
    at_risk_count = at_risk_encumbered_share_count(state, my_id, context.wares_loaded)
    if at_risk_count > 0:
        cost += SHARE_REPAY_AMOUNT * at_risk_count
    elif my_player is not None and my_player.cash - next_bid - context.preferred_share_price < 10:
        cost += SHARE_REPAY_AMOUNT - SHARE_LOAN_AMOUNT

    return next_bid if value > cost else None
