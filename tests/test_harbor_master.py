import math
import os
import sys
from fractions import Fraction
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from manilla.engine.models import (
    AccompliceSlot,
    GameState,
    PUNT_START_SUM,
    MAX_START_SPACE,
    Phase,
    Punt,
    PuntStatus,
    SHARE_LOAN_AMOUNT,
    SHARE_REPAY_AMOUNT,
    SHARES_PER_WARE,
    Share,
    Ware,
)
from manilla.engine.beliefs import infer_beliefs
from manilla.engine.expected_value import punt_port_probability
from manilla.engine.harbor_master import (
    _next_black_market_level,
    _share_buying_values,
    _share_price_gain,
    apply_punt_setup,
    at_risk_encumbered_share_count,
    best_punt_setup,
    best_shares_to_buy,
    decide_harbor_master_bid,
    expected_black_market_rise_value,
    first_mover_value,
    harbor_master_bid_context,
    harbor_master_static_value,
    harbor_master_value,
    punt_setup_candidates,
    share_buying_value,
)
from manilla.engine.wealth import action_impact, identify_rivals


def _make_state(player_names=("Me", "P1", "P2")):
    state = GameState.new_default_game(list(player_names))
    for p in state.players:
        p.shares = []
    state.players[0].is_harbor_master = True
    state.phase = Phase.ACCOMPLICE_ROUND
    state.movement_round_index = 0
    return state


def _sell_out(state: GameState, ware: Ware) -> None:
    """Make `ware` unavailable to buy, by allocating all its shares."""
    state.players[0].shares.extend(Share(ware=ware) for _ in range(SHARES_PER_WARE))


class TestNextBlackMarketLevel(unittest.TestCase):
    def test_steps_up_the_ladder_and_caps_at_thirty(self):
        self.assertEqual(_next_black_market_level(0), 5)
        self.assertEqual(_next_black_market_level(5), 10)
        self.assertEqual(_next_black_market_level(10), 20)
        self.assertEqual(_next_black_market_level(20), 30)
        self.assertEqual(_next_black_market_level(30), 30)


class TestSharePriceGain(unittest.TestCase):
    def test_zero_between_the_floor_levels_five_or_ten_above_it(self):
        self.assertEqual(_share_price_gain(0, 5), 0)  # both floored at 5
        self.assertEqual(_share_price_gain(5, 10), 5)
        self.assertEqual(_share_price_gain(10, 20), 10)
        self.assertEqual(_share_price_gain(20, 30), 10)


class TestShareBuyingValues(unittest.TestCase):
    def test_beginning_of_game_assumes_every_available_ware_worth_twenty(self):
        # Every ware still at 0 -- no signal yet to couple with punt
        # positioning, so per the user this falls back to a flat "worth
        # 20 eventually" assumption for whichever share gets bought.
        state = _make_state()
        beliefs = infer_beliefs(state, "p0")
        _sell_out(state, Ware.JADE)
        values = _share_buying_values(state, beliefs, "p0")
        self.assertEqual(values, {w: Fraction(20 - 5) for w in Ware if w != Ware.JADE})

    def test_matches_the_flat_additive_formula_using_the_chosen_punt_setup(self):
        # Once the market has moved off its all-zero start, each ware's
        # value should be (1 + p) times its one-step price gain: the
        # probabilistic this-voyage term (using the real arrival
        # probability of whatever position best_punt_setup assigns it)
        # plus a flat, unconditional further-step term of the same size
        # (per the user's "assume all shares will rise 1 step further
        # later in the game") -- see _share_buying_values' docstring.
        state = _make_state()
        state.black_market.values[Ware.GINSENG] = 5
        state.black_market.values[Ware.NUTMEG] = 10
        state.black_market.values[Ware.SILK] = 0
        state.black_market.values[Ware.JADE] = 0
        beliefs = infer_beliefs(state, "p0")

        (wares_loaded, positions), _ = best_punt_setup(state, beliefs, "p0")
        rounds_remaining = state.movement_rounds_total - state.movement_round_index

        expected = {}
        for ware in Ware:
            current = state.black_market.values[ware]
            gain = _share_price_gain(current, _next_black_market_level(current))
            p = punt_port_probability(positions[ware], rounds_remaining) if ware in wares_loaded else Fraction(0)
            expected[ware] = (1 + p) * gain

        self.assertEqual(_share_buying_values(state, beliefs, "p0"), expected)

    def test_the_ware_left_ashore_has_no_this_voyage_probability(self):
        # A ware not loaded onto any punt this voyage can't rise this
        # voyage at all, so its value should be exactly the flat
        # guaranteed-later-rise term with no probabilistic contribution.
        state = _make_state()
        state.black_market.values[Ware.GINSENG] = 5
        beliefs = infer_beliefs(state, "p0")
        (wares_loaded, positions), _ = best_punt_setup(state, beliefs, "p0")
        left_out = next(w for w in Ware if w not in wares_loaded)

        current = state.black_market.values[left_out]
        expected = Fraction(_share_price_gain(current, _next_black_market_level(current)))

        self.assertEqual(_share_buying_values(state, beliefs, "p0")[left_out], expected)

    def test_excludes_sold_out_and_maxed_out_wares(self):
        state = _make_state()
        state.black_market.values[Ware.GINSENG] = 30
        _sell_out(state, Ware.NUTMEG)
        beliefs = infer_beliefs(state, "p0")
        values = _share_buying_values(state, beliefs, "p0")
        self.assertNotIn(Ware.GINSENG, values)
        self.assertNotIn(Ware.NUTMEG, values)

    def test_reuses_a_planned_punt_setup_instead_of_recomputing(self):
        state = _make_state()
        state.black_market.values[Ware.GINSENG] = 5
        beliefs = infer_beliefs(state, "p0")
        planned = best_punt_setup(state, beliefs, "p0")[0]
        self.assertEqual(
            _share_buying_values(state, beliefs, "p0"),
            _share_buying_values(state, beliefs, "p0", planned_punt_setup=planned),
        )


class TestShareBuyingValue(unittest.TestCase):
    def test_is_the_max_of_the_per_ware_values(self):
        state = _make_state()
        state.black_market.values[Ware.GINSENG] = 20
        state.black_market.values[Ware.NUTMEG] = 0
        state.black_market.values[Ware.SILK] = 0
        _sell_out(state, Ware.JADE)
        beliefs = infer_beliefs(state, "p0")
        values = _share_buying_values(state, beliefs, "p0")
        self.assertEqual(share_buying_value(state, beliefs, "p0"), max(values.values()))

    def test_zero_when_nothing_is_available_to_buy(self):
        state = _make_state()
        for w in Ware:
            _sell_out(state, w)
        beliefs = infer_beliefs(state, "p0")
        self.assertEqual(share_buying_value(state, beliefs, "p0"), 0)


class TestBestSharesToBuy(unittest.TestCase):
    def test_returns_every_tied_best_ware(self):
        state = _make_state()
        # All four still at 0 -- the beginning-of-game special case makes
        # every remaining ware equally "worth 20", so all three available
        # ones tie.
        _sell_out(state, Ware.JADE)
        beliefs = infer_beliefs(state, "p0")
        self.assertEqual(set(best_shares_to_buy(state, beliefs, "p0")), {Ware.GINSENG, Ware.NUTMEG, Ware.SILK})

    def test_empty_when_nothing_is_worth_buying(self):
        state = _make_state()
        for w in Ware:
            _sell_out(state, w)
        beliefs = infer_beliefs(state, "p0")
        self.assertEqual(best_shares_to_buy(state, beliefs, "p0"), [])


class TestApplyPuntSetup(unittest.TestCase):
    def test_loads_wares_and_positions_and_marks_the_unloaded_one(self):
        state = _make_state()
        loaded = [Ware.SILK, Ware.GINSENG, Ware.JADE]
        positions = {Ware.SILK: 2, Ware.GINSENG: 3, Ware.JADE: 4}
        apply_punt_setup(loaded, positions)(state)

        self.assertEqual(state.punts[0].ware, Ware.SILK)
        self.assertEqual(state.punts[0].position, 2)
        self.assertEqual(state.punts[1].ware, Ware.GINSENG)
        self.assertEqual(state.punts[1].position, 3)
        self.assertEqual(state.punts[2].ware, Ware.JADE)
        self.assertEqual(state.punts[2].position, 4)
        self.assertTrue(all(p.status == PuntStatus.ON_ROUTE for p in state.punts))
        self.assertEqual(state.unloaded_ware, Ware.NUTMEG)

    def test_resets_ware_slots_to_fresh_vacant_ones(self):
        state = _make_state()
        state.punts[0].ware_slots = [AccompliceSlot(price=1, occupant="p1")]  # stale
        apply_punt_setup([Ware.JADE, Ware.SILK, Ware.GINSENG], {Ware.JADE: 3, Ware.SILK: 3, Ware.GINSENG: 3})(state)
        self.assertTrue(all(s.occupant is None for s in state.punts[0].ware_slots))
        self.assertEqual(len(state.punts[0].ware_slots), len(Punt.new(0, Ware.JADE).ware_slots))


class TestPuntSetupCandidates(unittest.TestCase):
    def test_covers_every_valid_position_triple_for_every_leave_ashore_choice(self):
        state = _make_state()
        candidates = punt_setup_candidates(state)

        expected_triples = sum(
            1
            for a in range(MAX_START_SPACE + 1)
            for b in range(MAX_START_SPACE + 1)
            if 0 <= PUNT_START_SUM - a - b <= MAX_START_SPACE
        )
        self.assertEqual(len(candidates), len(Ware) * expected_triples)

    def test_every_candidate_sums_to_punt_start_sum_within_bounds(self):
        state = _make_state()
        for loaded, positions in punt_setup_candidates(state):
            self.assertEqual(len(loaded), 3)
            self.assertEqual(sum(positions.values()), PUNT_START_SUM)
            for pos in positions.values():
                self.assertTrue(0 <= pos <= MAX_START_SPACE)

    def test_every_ware_gets_a_turn_being_left_ashore(self):
        state = _make_state()
        left_ashore = {next(w for w in Ware if w not in loaded) for loaded, _ in punt_setup_candidates(state)}
        self.assertEqual(left_ashore, set(Ware))


class TestExpectedBlackMarketRiseValue(unittest.TestCase):
    def test_matches_a_hand_assembled_single_ware_computation(self):
        state = _make_state()
        state.punts[0].ware = Ware.GINSENG
        state.punts[0].position = 8
        state.punts[0].status = PuntStatus.ON_ROUTE
        state.punts[1].ware = None
        state.punts[2].ware = None
        state.players[0].shares = [Share(ware=Ware.GINSENG), Share(ware=Ware.GINSENG)]

        beliefs = infer_beliefs(state, "p0")
        p_arrival = punt_port_probability(8, 3)
        # ginseng starts at 0 -> rises to 5 on the track, but share_price
        # floors at 5 regardless, so the actual PESOS gain is 0 even though
        # the black-market *level* moves.
        expected = p_arrival * (max(5, 5) - max(5, 0)) * 2
        self.assertEqual(expected_black_market_rise_value(state, beliefs, "p0"), expected)

    def test_nonzero_gain_once_the_floor_no_longer_masks_it(self):
        state = _make_state()
        state.punts[0].ware = Ware.GINSENG
        state.punts[0].position = 8
        state.punts[0].status = PuntStatus.ON_ROUTE
        state.punts[1].ware = None
        state.punts[2].ware = None
        state.black_market.values[Ware.GINSENG] = 5  # next rise is 5 -> 10, a real price jump
        state.players[0].shares = [Share(ware=Ware.GINSENG)]

        beliefs = infer_beliefs(state, "p0")
        p_arrival = punt_port_probability(8, 3)
        expected = p_arrival * (10 - 5) * 1
        self.assertEqual(expected_black_market_rise_value(state, beliefs, "p0"), expected)

    def test_zero_for_a_ware_the_player_holds_no_shares_of(self):
        state = _make_state()
        state.punts[0].ware = Ware.GINSENG
        state.punts[0].position = 8
        state.punts[0].status = PuntStatus.ON_ROUTE
        state.punts[1].ware = None
        state.punts[2].ware = None
        state.black_market.values[Ware.GINSENG] = 5
        beliefs = infer_beliefs(state, "p0")
        self.assertEqual(expected_black_market_rise_value(state, beliefs, "p0"), 0)

    def test_zero_for_a_ware_already_at_the_top_of_the_track(self):
        state = _make_state()
        state.punts[0].ware = Ware.GINSENG
        state.punts[0].position = 8
        state.punts[0].status = PuntStatus.ON_ROUTE
        state.punts[1].ware = None
        state.punts[2].ware = None
        state.black_market.values[Ware.GINSENG] = 30
        state.players[0].shares = [Share(ware=Ware.GINSENG)]
        beliefs = infer_beliefs(state, "p0")
        self.assertEqual(expected_black_market_rise_value(state, beliefs, "p0"), 0)

    def test_ignores_punts_that_are_not_on_route(self):
        state = _make_state()
        state.punts[0].ware = Ware.GINSENG
        state.punts[0].position = 13
        state.punts[0].status = PuntStatus.IN_PORT
        state.punts[1].ware = None
        state.punts[2].ware = None
        state.black_market.values[Ware.GINSENG] = 5
        state.players[0].shares = [Share(ware=Ware.GINSENG)]
        beliefs = infer_beliefs(state, "p0")
        self.assertEqual(expected_black_market_rise_value(state, beliefs, "p0"), 0)


class TestBestPuntSetup(unittest.TestCase):
    def test_favors_a_high_start_position_for_a_ware_the_player_holds(self):
        # No accomplices placed anywhere, so the only thing distinguishing
        # candidates is the black-market-rise effect on p0's own jade
        # holdings -- the best setup should push jade as far forward as
        # the 0..MAX_START_SPACE budget allows.
        state = _make_state()
        state.black_market.values[Ware.JADE] = 5  # a real price jump on the next rise
        state.players[0].shares = [Share(ware=Ware.JADE), Share(ware=Ware.JADE)]
        beliefs = infer_beliefs(state, "p0")

        (loaded, positions), score = best_punt_setup(state, beliefs, "p0")
        self.assertIn(Ware.JADE, loaded)
        self.assertEqual(positions[Ware.JADE], MAX_START_SPACE)

    def test_score_matches_a_manually_assembled_alternative_comparison(self):
        # Note: with every opponent's wealth_estimate padded by
        # DEFENSIVE_WEALTH_MARGIN, p1/p2 (equal cash to p0) now register as
        # phantom rivals, which can drag the absolute best_score negative
        # regardless of how good the choice is -- so this only checks the
        # *relative* ordering (best vs. worse), not an absolute floor.
        state = _make_state()
        state.black_market.values[Ware.JADE] = 5
        state.players[0].shares = [Share(ware=Ware.JADE), Share(ware=Ware.JADE)]
        beliefs = infer_beliefs(state, "p0")

        (loaded, positions), best_score = best_punt_setup(state, beliefs, "p0")

        # A candidate that leaves jade at the back should score no better.
        worse_loaded = [Ware.NUTMEG, Ware.SILK, Ware.JADE]
        worse_positions = {Ware.NUTMEG: 5, Ware.SILK: 4, Ware.JADE: 0}
        worse_after = GameState.from_dict(state.to_dict())
        apply_punt_setup(worse_loaded, worse_positions)(worse_after)
        worse_rise = expected_black_market_rise_value(worse_after, beliefs, "p0")
        best_after = GameState.from_dict(state.to_dict())
        apply_punt_setup(loaded, positions)(best_after)
        best_rise = expected_black_market_rise_value(best_after, beliefs, "p0")
        self.assertGreaterEqual(best_rise, worse_rise)


class TestFirstMoverValue(unittest.TestCase):
    def test_zero_when_no_one_else_is_still_bidding(self):
        order = ["p0", "p1", "p2"]
        self.assertEqual(first_mover_value(order, "p0", ["p0"], 1.0), 0)

    def test_uses_the_nearest_active_bidder_after_me(self):
        order = ["p0", "p1", "p2"]
        # p1 is immediately after p0 and still active -- if p0 passes,
        # assume p1 wins; p0 would then sit 2 spots behind p1 (rotation
        # restarts at p1: p1(0), p2(1), p0(2)).
        value = first_mover_value(order, "p0", ["p0", "p1", "p2"], 1.0)
        self.assertEqual(value, Fraction(1) * 2 * 3)

    def test_skips_inactive_players_to_find_the_nearest_active_one(self):
        order = ["p0", "p1", "p2", "p3"]
        # p1 has already passed; p2 is the nearest still-active bidder.
        value = first_mover_value(order, "p0", ["p0", "p2", "p3"], 1.0)
        # rotation restarts at p2: p2(0), p3(1), p0(2), p1(3) -- p0 is 2 behind.
        self.assertEqual(value, Fraction(1) * 2 * 3)

    def test_wraps_around_the_turn_order(self):
        order = ["p0", "p1", "p2"]
        # Only p0 and p2 remain active; p2 comes before p0 in raw order but
        # wraps around to be the "nearest after" p0.
        value = first_mover_value(order, "p0", ["p0", "p2"], 1.0)
        # rotation restarts at p2: p2(0), p0(1) -- p0 is 1 behind.
        self.assertEqual(value, Fraction(1) * 1 * 3)

    def test_scales_with_the_cost_coefficient(self):
        order = ["p0", "p1"]
        low = first_mover_value(order, "p0", ["p0", "p1"], 0.5)
        high = first_mover_value(order, "p0", ["p0", "p1"], 2.0)
        self.assertEqual(high, low * 4)

    def test_accepts_a_float_coefficient_exactly(self):
        order = ["p0", "p1"]
        value = first_mover_value(order, "p0", ["p0", "p1"], 0.5)
        self.assertEqual(value, Fraction(1, 2) * 1 * 3)


class TestHarborMasterValue(unittest.TestCase):
    def test_sums_all_three_components(self):
        # The punt-setup component is best_punt_setup's score *net of* a
        # neutral baseline (see harbor_master_static_value) -- not the raw
        # score share_buying_value would otherwise be added to directly.
        state = _make_state()
        state.black_market.values[Ware.GINSENG] = 0
        beliefs = infer_beliefs(state, "p0")
        order = ["p0", "p1", "p2"]
        active = ["p0", "p1", "p2"]

        total = harbor_master_value(state, beliefs, "p0", order, active, 1.0)

        static_part = harbor_master_static_value(state, beliefs, "p0")
        mover_part = first_mover_value(order, "p0", active, 1.0)
        self.assertEqual(total, static_part + mover_part)


class TestDecideHarborMasterBid(unittest.TestCase):
    def test_bids_one_step_when_the_value_clears_it(self):
        state = _make_state()
        beliefs = infer_beliefs(state, "p0")
        order = ["p0", "p1", "p2"]
        active = ["p0", "p1", "p2"]
        value = harbor_master_value(state, beliefs, "p0", order, active, 1.0)

        # A highest bid comfortably below the computed value should be
        # worth raising by exactly one step.
        highest = int(value) - 5
        self.assertEqual(
            decide_harbor_master_bid(state, beliefs, "p0", order, active, highest, 1.0),
            highest + 1,
        )

    def test_passes_when_the_next_step_would_not_clear_the_value(self):
        state = _make_state()
        beliefs = infer_beliefs(state, "p0")
        order = ["p0", "p1", "p2"]
        active = ["p0", "p1", "p2"]
        value = harbor_master_value(state, beliefs, "p0", order, active, 1.0)

        # A highest bid already at (or past) the value should not be
        # raised further -- winning at that price wouldn't be worth it.
        highest = int(value) + 100
        self.assertIsNone(decide_harbor_master_bid(state, beliefs, "p0", order, active, highest, 1.0))

    def test_recalibrates_as_active_bidders_shrink(self):
        # Same board, same highest bid -- but fewer active bidders after
        # someone passes should change the first-mover component (a
        # different, possibly farther, nearest active bidder), and so can
        # change the decision.
        state = _make_state()
        beliefs = infer_beliefs(state, "p0")
        order = ["p0", "p1", "p2", "p3"]
        highest = 5

        full_field = decide_harbor_master_bid(state, beliefs, "p0", order, ["p0", "p1", "p2", "p3"], highest, 2.0)
        after_p1_passes = decide_harbor_master_bid(state, beliefs, "p0", order, ["p0", "p2", "p3"], highest, 2.0)
        value_full = harbor_master_value(state, beliefs, "p0", order, ["p0", "p1", "p2", "p3"], 2.0)
        value_after_pass = harbor_master_value(state, beliefs, "p0", order, ["p0", "p2", "p3"], 2.0)
        self.assertNotEqual(value_full, value_after_pass)
        self.assertEqual(full_field, highest + 1 if value_full > highest + 1 else None)
        self.assertEqual(after_p1_passes, highest + 1 if value_after_pass > highest + 1 else None)

    def test_encumbrance_penalty_can_flip_a_clearing_bid_to_a_pass(self):
        # Per the user: if paying the next bid, *and* then buying the
        # share my_id would go on to prefer as harbor master, would drop
        # my_id's cash below 10, assume that forces encumbering a share, a
        # permanent 3-peso loss -- so bidding should refuse a bid that
        # clears the raw value but not value-minus-3. harbor_master_value
        # and the preferred share's price don't depend on my_id's own
        # absolute cash (the former is built entirely from wealth
        # *differences*; the latter only from market/board state), so
        # changing p0's cash between the two calls below only exercises
        # the penalty, not the underlying numbers.
        state = _make_state()
        beliefs = infer_beliefs(state, "p0")
        order = ["p0", "p1", "p2"]
        active = ["p0", "p1", "p2"]
        value = harbor_master_value(state, beliefs, "p0", order, active, 1.0)
        preferred_share_price = harbor_master_bid_context(state, beliefs, "p0").preferred_share_price

        # A next_bid that clears the raw value (value > next_bid) but
        # sits within the assumed penalty of it (value <= next_bid + 3).
        next_bid = math.floor(value) - 2
        highest = next_bid - 1
        self.assertGreater(value, next_bid)
        self.assertLessEqual(value, next_bid + 3)

        me = state.player_by_id("p0")
        # Winning would leave only 5 pesos after ALSO buying the
        # preferred share -- below 10.
        me.cash = next_bid + preferred_share_price + 5
        self.assertIsNone(decide_harbor_master_bid(state, beliefs, "p0", order, active, highest, 1.0))

        # Winning would comfortably clear 10 pesos even after that purchase.
        me.cash = next_bid + preferred_share_price + 50
        self.assertEqual(
            decide_harbor_master_bid(state, beliefs, "p0", order, active, highest, 1.0), next_bid
        )

    def test_encumbrance_penalty_matches_share_repay_minus_loan(self):
        state = _make_state()
        state.player_by_id("p0").cash = 12
        beliefs = infer_beliefs(state, "p0")
        order = ["p0", "p1", "p2"]
        active = ["p0", "p1", "p2"]
        value = harbor_master_value(state, beliefs, "p0", order, active, 1.0)
        preferred_share_price = harbor_master_bid_context(state, beliefs, "p0").preferred_share_price
        highest = int(value) - 2
        next_bid = highest + 1
        self.assertLess(state.player_by_id("p0").cash - next_bid - preferred_share_price, 10)

        expected_cost = next_bid + (SHARE_REPAY_AMOUNT - SHARE_LOAN_AMOUNT)
        expected = next_bid if value > expected_cost else None
        self.assertEqual(decide_harbor_master_bid(state, beliefs, "p0", order, active, highest, 1.0), expected)

    def test_at_risk_penalty_replaces_rather_than_stacks_with_below_ten(self):
        # Per the user, the two liquidity penalties don't stack: an
        # at-risk encumbered share (15 pesos) is a bigger concern than the
        # plain below-10-cash one (3 pesos), so when it applies, the
        # smaller one is skipped entirely -- even if cash would ALSO
        # trigger it. Confirmed by picking a next_bid that clears
        # value-minus-15 but would NOT clear value-minus-18 (what
        # 15 + 3 stacked would require).
        state = _make_state()
        state.black_market.values[Ware.GINSENG] = 20
        state.player_by_id("p0").shares = [Share(ware=Ware.GINSENG, encumbered=True)]
        beliefs = infer_beliefs(state, "p0")
        order = ["p0", "p1", "p2"]
        active = ["p0", "p1", "p2"]
        value = harbor_master_value(state, beliefs, "p0", order, active, 1.0)
        preferred_share_price = harbor_master_bid_context(state, beliefs, "p0").preferred_share_price

        next_bid = math.floor(value) - 16
        highest = next_bid - 1
        self.assertGreater(value, next_bid + 15)
        self.assertLessEqual(value, next_bid + 15 + 3)

        me = state.player_by_id("p0")
        # Also below 10 after buying the preferred share -- both penalty
        # conditions are true here, but only the bigger one should apply.
        me.cash = next_bid + preferred_share_price + 5
        self.assertEqual(decide_harbor_master_bid(state, beliefs, "p0", order, active, highest, 1.0), next_bid)

    def test_precomputed_bid_context_gives_the_same_answer(self):
        # The whole point of precomputed_bid_context is to skip
        # recomputing harbor_master_bid_context on every bid -- confirm
        # it produces an identical decision to the fully-recomputed path.
        state = _make_state()
        beliefs = infer_beliefs(state, "p0")
        order = ["p0", "p1", "p2"]
        active = ["p0", "p1", "p2"]
        context = harbor_master_bid_context(state, beliefs, "p0")

        fresh = decide_harbor_master_bid(state, beliefs, "p0", order, active, 3, 1.5)
        cached = decide_harbor_master_bid(
            state, beliefs, "p0", order, active, 3, 1.5, precomputed_bid_context=context
        )
        self.assertEqual(fresh, cached)


class TestHarborMasterBidContext(unittest.TestCase):
    def test_static_value_matches_harbor_master_static_value(self):
        state = _make_state()
        beliefs = infer_beliefs(state, "p0")
        context = harbor_master_bid_context(state, beliefs, "p0")
        self.assertEqual(context.static_value, harbor_master_static_value(state, beliefs, "p0"))

    def test_preferred_share_price_is_the_priciest_tied_best_ware(self):
        state = _make_state()
        state.black_market.values[Ware.GINSENG] = 20
        state.black_market.values[Ware.NUTMEG] = 5
        _sell_out(state, Ware.SILK)
        _sell_out(state, Ware.JADE)
        beliefs = infer_beliefs(state, "p0")
        candidates = best_shares_to_buy(state, beliefs, "p0")
        context = harbor_master_bid_context(state, beliefs, "p0")
        if candidates:
            self.assertEqual(
                context.preferred_share_price, max(state.black_market.share_price(w) for w in candidates)
            )
        else:
            self.assertEqual(context.preferred_share_price, 0)

    def test_zero_when_buying_is_not_worthwhile(self):
        state = _make_state()
        for w in Ware:
            _sell_out(state, w)
        beliefs = infer_beliefs(state, "p0")
        context = harbor_master_bid_context(state, beliefs, "p0")
        self.assertEqual(context.preferred_share_price, 0)

    def test_wares_loaded_matches_best_punt_setups_own_choice(self):
        state = _make_state()
        beliefs = infer_beliefs(state, "p0")
        (expected_wares_loaded, _), _ = best_punt_setup(state, beliefs, "p0")
        context = harbor_master_bid_context(state, beliefs, "p0")
        self.assertEqual(context.wares_loaded, expected_wares_loaded)


class TestAtRiskEncumberedShareCount(unittest.TestCase):
    def test_counts_an_encumbered_share_already_at_twenty(self):
        state = _make_state()
        state.black_market.values[Ware.GINSENG] = 20
        state.player_by_id("p0").shares = [Share(ware=Ware.GINSENG, encumbered=True)]
        self.assertEqual(at_risk_encumbered_share_count(state, "p0", []), 1)

    def test_counts_an_encumbered_share_at_ten_only_if_its_ware_is_loaded(self):
        state = _make_state()
        state.black_market.values[Ware.GINSENG] = 10
        state.player_by_id("p0").shares = [Share(ware=Ware.GINSENG, encumbered=True)]
        self.assertEqual(at_risk_encumbered_share_count(state, "p0", [Ware.GINSENG]), 1)
        self.assertEqual(at_risk_encumbered_share_count(state, "p0", [Ware.NUTMEG]), 0)

    def test_ignores_shares_that_are_not_encumbered(self):
        state = _make_state()
        state.black_market.values[Ware.GINSENG] = 20
        state.player_by_id("p0").shares = [Share(ware=Ware.GINSENG, encumbered=False)]
        self.assertEqual(at_risk_encumbered_share_count(state, "p0", []), 0)

    def test_ignores_levels_below_the_risk_thresholds(self):
        state = _make_state()
        state.black_market.values[Ware.GINSENG] = 5
        state.player_by_id("p0").shares = [Share(ware=Ware.GINSENG, encumbered=True)]
        self.assertEqual(at_risk_encumbered_share_count(state, "p0", [Ware.GINSENG]), 0)

    def test_counts_every_qualifying_share_separately(self):
        state = _make_state()
        state.black_market.values[Ware.GINSENG] = 20
        state.black_market.values[Ware.SILK] = 10
        state.player_by_id("p0").shares = [
            Share(ware=Ware.GINSENG, encumbered=True),
            Share(ware=Ware.GINSENG, encumbered=True),
            Share(ware=Ware.SILK, encumbered=True),
            Share(ware=Ware.NUTMEG, encumbered=True),  # at 0 -- doesn't qualify
        ]
        self.assertEqual(at_risk_encumbered_share_count(state, "p0", [Ware.SILK]), 3)

    def test_zero_for_an_unknown_player(self):
        state = _make_state()
        self.assertEqual(at_risk_encumbered_share_count(state, "nobody", []), 0)


class TestHarborMasterStaticValue(unittest.TestCase):
    def test_matches_share_plus_marginal_punt_setup(self):
        # The punt-setup component is best_punt_setup's raw score *minus*
        # the same score for a neutral, non-strategic baseline setup (the
        # first 3 wares, equal 3/3/3 positions) -- not the raw score
        # added directly. See harbor_master_static_value's docstring for
        # why: the raw score alone carries a DEFENSIVE_WEALTH_MARGIN-driven
        # constant offset that has nothing to do with punt-setup skill,
        # and the neutral baseline is what cancels it out.
        state = _make_state()
        beliefs = infer_beliefs(state, "p0")
        rivals = identify_rivals(state, beliefs, "p0")

        best_candidate, punt_part = best_punt_setup(state, beliefs, "p0")

        neutral_loaded = list(Ware)[:3]
        neutral_positions = {w: 3 for w in neutral_loaded}
        neutral_mutator = apply_punt_setup(neutral_loaded, neutral_positions)
        neutral_impact = action_impact(state, beliefs, "p0", neutral_mutator)
        neutral_after = GameState.from_dict(state.to_dict())
        neutral_mutator(neutral_after)
        neutral_rise = expected_black_market_rise_value(neutral_after, beliefs, "p0")
        for rival_id in rivals:
            neutral_rise -= expected_black_market_rise_value(neutral_after, beliefs, rival_id)
        neutral_score = neutral_impact.total_rev_after + neutral_rise

        expected = (
            share_buying_value(state, beliefs, "p0", planned_punt_setup=best_candidate) + (punt_part - neutral_score)
        )
        self.assertEqual(harbor_master_static_value(state, beliefs, "p0"), expected)

    def test_harbor_master_value_equals_static_plus_first_mover(self):
        state = _make_state()
        beliefs = infer_beliefs(state, "p0")
        order = ["p0", "p1", "p2"]
        active = ["p0", "p1", "p2"]
        static_value = harbor_master_static_value(state, beliefs, "p0")
        mover = first_mover_value(order, "p0", active, 1.0)
        self.assertEqual(harbor_master_value(state, beliefs, "p0", order, active, 1.0), static_value + mover)


if __name__ == "__main__":
    unittest.main()
