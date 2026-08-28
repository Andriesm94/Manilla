import os
import sys
from fractions import Fraction
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from manilla.engine.models import (
    AccompliceSlot,
    GameState,
    INSURANCE_SHIPYARD_COST,
    Phase,
    PIRATE_PRICE,
    PLUNDER_PAYOUTS,
    Punt,
    PuntStatus,
    SHARE_REPAY_AMOUNT,
    Share,
    Ware,
)
from manilla.engine.beliefs import ShareSignal, infer_beliefs
from manilla.engine.expected_value import (
    dock_fill_distribution,
    dock_slot_expected_payout,
    pirate_expected_payout,
    punt_port_probability,
    punt_shipyard_probability,
    ware_slot_expected_payout,
)
from manilla.engine.probability import position_outcomes
from manilla.engine.wealth import (
    DEFENSIVE_WEALTH_MARGIN,
    _p_port_if_caught_on_13,
    action_impact,
    apply_dock_slot_placement,
    apply_insurance_placement,
    apply_pilot_move,
    apply_pilot_placement,
    apply_pirate_placement,
    apply_ware_slot_placement,
    best_pilot_move,
    encumbered_penalty,
    expected_accomplice_return,
    identify_rivals,
    insurance_ev,
    occupied_pilot_slots,
    pilot_move_candidates,
    pilot_slot_value,
    pirate_captain_boarding_bonus,
    pirate_slot_ev_if_taken_now,
    pirate_threat,
    predict_pilot_move,
    project_final_occupancy,
    rev,
    rev_adjusted_score,
    wealth_estimate,
    with_predicted_pilot_moves,
)


def _make_state():
    state = GameState.new_default_game(["Me", "P1", "P2"])
    for p in state.players:
        p.shares = []
    state.players[0].is_harbor_master = True
    state.phase = Phase.ACCOMPLICE_ROUND
    state.current_turn_player_id = "p0"
    state.movement_round_index = 0  # 3 movement rounds still ahead
    return state


class TestEncumberedPenalty(unittest.TestCase):
    def test_charges_the_flat_repay_amount_per_share(self):
        state = _make_state()
        state.players[2].shares = [
            Share(ware=Ware.JADE, encumbered=True),
            Share(ware=Ware.SILK, encumbered=True),
            Share(ware=Ware.NUTMEG, encumbered=False),
        ]
        self.assertEqual(encumbered_penalty(state.players[2]), 2 * SHARE_REPAY_AMOUNT)


class TestExpectedAccompliceReturn(unittest.TestCase):
    def _rigged_state(self):
        state = _make_state()

        # p1 alone on a ginseng punt, still at sea.
        state.punts[0].ware = Ware.GINSENG
        state.punts[0].position = 8
        state.punts[0].status = PuntStatus.ON_ROUTE
        state.punts[0].ware_slots = [
            AccompliceSlot(price=3, occupant="p1"),
            AccompliceSlot(price=2, occupant=None),
            AccompliceSlot(price=1, occupant=None),
        ]

        # Me, alone, on a silk punt that's already arrived -- certain payout.
        state.punts[1].ware = Ware.SILK
        state.punts[1].position = 13
        state.punts[1].status = PuntStatus.IN_PORT
        state.punts[1].ware_slots = [
            AccompliceSlot(price=5, occupant="p0"),
            AccompliceSlot(price=4, occupant=None),
            AccompliceSlot(price=3, occupant=None),
        ]

        # A third, unloaded punt stays at its default (ware=None).
        state.punts[2].ware = Ware.NUTMEG
        state.punts[2].position = 5
        state.punts[2].status = PuntStatus.ON_ROUTE

        state.port.slots["A"].occupant = "p2"
        state.pirate_boat.captain.occupant = "p0"
        return state

    def test_on_route_ware_slot_projects_full_occupancy_by_default(self):
        # Only p1 has actually claimed a slot so far, but with a whole
        # movement round still ahead (movement_round_index=0, well short of
        # the final round), the punt's other two ginseng slots are assumed
        # to fill before the voyage resolves -- so the payout is valued as
        # if split 3 ways, not kept whole for p1 alone. p0's pirate
        # captaincy (set up by _rigged_state) makes p_safe_if_caught auto-
        # derive to 0 -- plunder is certain if this punt gets caught on 13.
        state = self._rigged_state()
        beliefs = infer_beliefs(state, "p0")
        expected = ware_slot_expected_payout(Ware.GINSENG, 8, 3, accomplices_on_punt=3, p_safe_if_caught=0)
        self.assertEqual(expected_accomplice_return(state, beliefs, "p1"), expected)

    def test_docked_ware_slot_pays_out_with_certainty(self):
        state = self._rigged_state()
        beliefs = infer_beliefs(state, "p0")
        # Silk punt already in port: certain payout, no dice math needed --
        # 30 (silk's plunder payout) split across its 1 occupied slot
        # (already resolved, not projected). p0's lone pirate captaincy is
        # valued using exactly the 1 pirate actually aboard -- no
        # projection, see test_pirate_valuation_never_projects_occupancy --
        # plus the round-1/2 free-boarding bonus (movement_round_index=0
        # here), see TestPirateCaptainBoardingBonus.
        pirate_part = pirate_expected_payout(
            [(Ware.GINSENG, 8, 3), (Ware.NUTMEG, 5, 3)], pirate_count=1
        )
        boarding_bonus = pirate_captain_boarding_bonus(state)
        self.assertEqual(
            expected_accomplice_return(state, beliefs, "p0"), Fraction(30) + pirate_part + boarding_bonus
        )

    def test_pirate_valuation_never_projects_occupancy(self):
        # Unlike ware punts, the pirate boat gets no occupancy projection at
        # all -- a lone captain is valued as exactly 1 pirate regardless of
        # how profitable (or not) a hypothetical 2-way split would be, and
        # regardless of how many rounds or turns remain.
        state = _make_state()
        state.punts[0].ware = Ware.JADE
        state.punts[0].position = 6
        state.punts[0].status = PuntStatus.ON_ROUTE
        state.pirate_boat.captain.occupant = "p0"
        beliefs = infer_beliefs(state, "p0")

        for movement_round_index in (0, 1, 2):
            with self.subTest(movement_round_index=movement_round_index):
                state.movement_round_index = movement_round_index
                expected = pirate_expected_payout(
                    [(Ware.JADE, 6, 3 - movement_round_index)], pirate_count=1
                )
                self.assertEqual(expected_accomplice_return(state, beliefs, "p0"), expected)

    def test_pirate_valuation_uses_current_count_when_two_are_aboard(self):
        # Two pirates actually aboard is a fact, not a projection -- the
        # split is valued as 2-way regardless of profitability or timing.
        state = _make_state()
        state.punts[0].ware = Ware.JADE
        state.punts[0].position = 6
        state.punts[0].status = PuntStatus.ON_ROUTE
        state.pirate_boat.captain.occupant = "p0"
        state.pirate_boat.second.occupant = "p1"
        beliefs = infer_beliefs(state, "p0")

        expected = pirate_expected_payout([(Ware.JADE, 6, 3)], pirate_count=2)
        self.assertEqual(expected_accomplice_return(state, beliefs, "p0"), expected)
        self.assertEqual(expected_accomplice_return(state, beliefs, "p1"), expected)

    def test_last_turn_of_the_final_round_uses_actual_occupancy(self):
        state = self._rigged_state()
        state.movement_round_index = 2  # about to trigger the third dice throw -- 1 round left
        state.current_turn_player_id = "p2"  # last of 3 players in this round's rotation
        beliefs = infer_beliefs(state, "p0")
        # No more placement chances remain -- p1's slot is valued alone.
        expected = ware_slot_expected_payout(Ware.GINSENG, 8, 1, accomplices_on_punt=1, p_safe_if_caught=0)
        self.assertEqual(expected_accomplice_return(state, beliefs, "p1"), expected)

    def test_first_turn_of_the_final_round_also_uses_actual_occupancy(self):
        # Per the user: the "will fill up" assumption is dropped for the
        # *entire* final accomplice round, not just its last couple of
        # turns -- even the harbor master's own first turn of that round
        # (plenty of turns still to come) values p1's slot as-is.
        state = self._rigged_state()
        state.movement_round_index = 2
        state.current_turn_player_id = "p0"  # harbor master, first to act -- 3 turns still remain
        beliefs = infer_beliefs(state, "p0")
        expected = ware_slot_expected_payout(Ware.GINSENG, 8, 1, accomplices_on_punt=1, p_safe_if_caught=0)
        self.assertEqual(expected_accomplice_return(state, beliefs, "p1"), expected)

    def test_middle_turn_of_the_final_round_also_uses_actual_occupancy(self):
        state = self._rigged_state()
        state.movement_round_index = 2
        state.current_turn_player_id = "p1"  # 2nd of 3
        beliefs = infer_beliefs(state, "p0")
        expected = ware_slot_expected_payout(Ware.GINSENG, 8, 1, accomplices_on_punt=1, p_safe_if_caught=0)
        self.assertEqual(expected_accomplice_return(state, beliefs, "p1"), expected)

    def test_pirate_valuation_is_unaffected_by_the_final_round_rule(self):
        # The final-accomplice-round exception is a ware-punt-only concept
        # (see project_final_occupancy) -- pirates don't have a growth
        # projection to fall back from in the first place.
        state = self._rigged_state()
        state.movement_round_index = 2
        state.current_turn_player_id = "p2"  # last turn of the round
        beliefs = infer_beliefs(state, "p0")
        pirate_part = pirate_expected_payout(
            [(Ware.GINSENG, 8, 1), (Ware.NUTMEG, 5, 1)], pirate_count=1
        )
        self.assertEqual(expected_accomplice_return(state, beliefs, "p0"), Fraction(30) + pirate_part)

    def test_dock_slot_matches_the_underlying_ev_function(self):
        # _rigged_state makes p0 the pirate captain, and p0 (like every
        # player here) holds no shares of anything -- so per the user's
        # port-vs-shipyard rule, the captain isn't believed to hold *more*
        # ginseng or nutmeg shares than the viewer (p0 itself, tied at 0),
        # so a plundered catch on either on-route punt is assumed to go to
        # the shipyard, not port: p_safe_if_caught=0 for both.
        state = self._rigged_state()
        beliefs = infer_beliefs(state, "p0")
        arrival_probs = [
            punt_port_probability(8, 3, p_safe_if_caught=0),
            1,
            punt_port_probability(5, 3, p_safe_if_caught=0),
        ]
        expected = dock_slot_expected_payout("port", "A", arrival_probs)
        self.assertEqual(expected_accomplice_return(state, beliefs, "p2"), expected)

    def test_settled_phases_return_zero_even_with_slots_occupied(self):
        state = self._rigged_state()
        beliefs = infer_beliefs(state, "p0")
        state.phase = Phase.PROFIT_DISTRIBUTION
        self.assertEqual(expected_accomplice_return(state, beliefs, "p1"), 0)
        self.assertEqual(expected_accomplice_return(state, beliefs, "p0"), 0)

    def test_shipwrecked_ware_slot_pays_nothing(self):
        state = self._rigged_state()
        beliefs = infer_beliefs(state, "p0")
        state.punts[0].status = PuntStatus.IN_SHIPYARD
        self.assertEqual(expected_accomplice_return(state, beliefs, "p1"), 0)

    def test_player_with_no_slots_gets_zero(self):
        state = self._rigged_state()
        beliefs = infer_beliefs(state, "p0")
        self.assertEqual(expected_accomplice_return(state, beliefs, "nobody"), 0)


class TestProjectFinalOccupancy(unittest.TestCase):
    def test_defaults_to_full_capacity_outside_the_final_round(self):
        state = _make_state()
        state.movement_round_index = 0
        self.assertEqual(project_final_occupancy(state, current_occupied=1, max_slots=4), 4)

    def test_uses_actual_occupancy_from_the_very_start_of_the_final_round(self):
        # Per the user: the "will fill up" assumption is dropped for the
        # *entire* final accomplice round, not just its last couple of
        # turns -- even the harbor master's own first turn of that round
        # (plenty of turns still to come) should use actual occupancy.
        state = _make_state()
        state.movement_round_index = 2
        state.current_turn_player_id = "p0"  # harbor master, first turn of the round
        self.assertEqual(project_final_occupancy(state, current_occupied=1, max_slots=4), 1)

    def test_uses_actual_occupancy_on_the_last_turn_too(self):
        state = _make_state()
        state.movement_round_index = 2
        state.current_turn_player_id = "p2"  # last turn
        self.assertEqual(project_final_occupancy(state, current_occupied=1, max_slots=4), 1)

    def test_outside_accomplice_round_defaults_to_full_capacity(self):
        # Placement is over for this phase already; expected_accomplice_return
        # itself zeroes out settled phases, but the helper alone stays
        # conservative rather than guessing.
        state = _make_state()
        state.phase = Phase.PROFIT_DISTRIBUTION
        state.movement_round_index = 2
        state.current_turn_player_id = "p2"
        self.assertEqual(project_final_occupancy(state, current_occupied=1, max_slots=4), 4)


class TestWealthEstimate(unittest.TestCase):
    def test_matches_hand_assembled_components(self):
        state = _make_state()
        state.players[0].cash = 50
        state.players[0].shares = [Share(ware=Ware.SILK)]
        beliefs = infer_beliefs(state, "p0")

        expected = Fraction(50) + state.black_market.share_price(Ware.SILK) - 0 + 0
        self.assertEqual(wealth_estimate(state, beliefs, "p0"), expected)

    def test_unknown_rejects_missing_player(self):
        state = _make_state()
        beliefs = infer_beliefs(state, "p0")
        with self.assertRaises(ValueError):
            wealth_estimate(state, beliefs, "nobody")


class TestDefensiveWealthMargin(unittest.TestCase):
    def test_the_viewers_own_estimate_is_never_padded(self):
        state = _make_state()
        state.players[0].cash = 50
        beliefs = infer_beliefs(state, "p0")
        self.assertEqual(wealth_estimate(state, beliefs, "p0"), 50)

    def test_every_other_players_estimate_is_padded(self):
        state = _make_state()
        state.players[1].cash = 50
        beliefs = infer_beliefs(state, "p0")
        self.assertEqual(wealth_estimate(state, beliefs, "p1"), 50 + DEFENSIVE_WEALTH_MARGIN)

    def test_padding_applies_from_any_viewers_perspective_not_just_p0(self):
        # The margin isn't hardcoded to "p0 is me" -- it's about whoever
        # these beliefs are anchored to (beliefs.viewer_id) versus anyone
        # else, so it applies just as much when predicting another
        # player's own reasoning about a third player.
        state = _make_state()
        state.players[2].cash = 50
        beliefs_from_p1 = infer_beliefs(state, "p1")
        self.assertEqual(wealth_estimate(state, beliefs_from_p1, "p2"), 50 + DEFENSIVE_WEALTH_MARGIN)
        self.assertEqual(wealth_estimate(state, beliefs_from_p1, "p1"), state.players[1].cash)


class TestRivalsAndREV(unittest.TestCase):
    def test_only_strictly_wealthier_opponents_are_rivals(self):
        state = _make_state()
        state.players[0].cash = 50  # me
        state.players[1].cash = 500  # clearly ahead
        state.players[2].cash = 0  # clearly behind
        beliefs = infer_beliefs(state, "p0")

        self.assertEqual(identify_rivals(state, beliefs, "p0"), ["p1"])

    def test_no_rivals_when_already_in_the_lead(self):
        state = _make_state()
        state.players[0].cash = 500
        state.players[1].cash = 10
        state.players[2].cash = 5
        beliefs = infer_beliefs(state, "p0")
        self.assertEqual(identify_rivals(state, beliefs, "p0"), [])

    def test_rivals_are_ordered_richest_first(self):
        state = _make_state()
        state.players[0].cash = 0
        state.players[1].cash = 50
        state.players[2].cash = 100
        beliefs = infer_beliefs(state, "p0")
        self.assertEqual(identify_rivals(state, beliefs, "p0"), ["p2", "p1"])

    def test_rev_matches_the_wealth_gap(self):
        # p1's estimate is padded by DEFENSIVE_WEALTH_MARGIN (15) on top of
        # their raw cash, since p0 is only ever exact about its own wealth.
        state = _make_state()
        state.players[0].cash = 50
        state.players[1].cash = 53
        beliefs = infer_beliefs(state, "p0")
        self.assertEqual(rev(state, beliefs, "p0", "p1"), 3 + DEFENSIVE_WEALTH_MARGIN)

    def test_rev_can_go_negative_for_a_former_rival(self):
        state = _make_state()
        state.players[0].cash = 100
        state.players[1].cash = 53
        beliefs = infer_beliefs(state, "p0")
        self.assertEqual(rev(state, beliefs, "p0", "p1"), -47 + DEFENSIVE_WEALTH_MARGIN)


class TestRevAdjustedScore(unittest.TestCase):
    def test_plain_own_gain_when_no_rival_effect_given(self):
        self.assertEqual(rev_adjusted_score(10), 10)

    def test_subtracts_a_rivals_gain(self):
        self.assertEqual(rev_adjusted_score(10, 4), 6)

    def test_hurting_a_rival_adds_to_the_score(self):
        self.assertEqual(rev_adjusted_score(10, -4), 14)


class TestPirateThreat(unittest.TestCase):
    def test_no_pirates_is_safe(self):
        state = _make_state()
        self.assertEqual(pirate_threat(state), 1)

    def test_captain_alone_makes_it_certain_plunder(self):
        state = _make_state()
        state.pirate_boat.captain.occupant = "p0"
        self.assertEqual(pirate_threat(state), 0)

    def test_second_alone_also_makes_it_certain_plunder(self):
        state = _make_state()
        state.pirate_boat.second.occupant = "p1"
        self.assertEqual(pirate_threat(state), 0)


class TestPPortIfCaughtOn13(unittest.TestCase):
    def test_no_pirates_means_certain_port_arrival(self):
        state = _make_state()
        beliefs = infer_beliefs(state, "p0")
        self.assertEqual(_p_port_if_caught_on_13(state, beliefs, Ware.JADE), 1)

    def test_captain_with_more_shares_than_the_viewer_assumes_port(self):
        # Per the user: sending it to port raises the ware's black-market
        # value, so assume the captain does that when it's believed to
        # benefit them more than the viewer.
        state = _make_state()
        state.pirate_boat.captain.occupant = "p1"
        state.players[1].shares = [Share(ware=Ware.JADE, encumbered=False)]
        beliefs = infer_beliefs(state, "p0", signals=[ShareSignal(player_id="p1", ware=Ware.JADE)])
        self.assertEqual(_p_port_if_caught_on_13(state, beliefs, Ware.JADE), 1)

    def test_captain_with_no_more_shares_than_the_viewer_assumes_shipyard(self):
        state = _make_state()
        state.pirate_boat.captain.occupant = "p1"
        state.players[0].shares = [Share(ware=Ware.JADE, encumbered=False)]  # the viewer holds it
        beliefs = infer_beliefs(state, "p0")
        self.assertEqual(_p_port_if_caught_on_13(state, beliefs, Ware.JADE), 0)

    def test_tied_share_counts_assume_shipyard(self):
        # Not strictly *more* -- a tie doesn't count, per the user's wording.
        state = _make_state()
        state.pirate_boat.captain.occupant = "p1"
        beliefs = infer_beliefs(state, "p0")  # both p0 and p1 hold 0 jade shares
        self.assertEqual(_p_port_if_caught_on_13(state, beliefs, Ware.JADE), 0)

    def test_lone_second_pirate_is_the_decider_when_no_captain_is_aboard(self):
        state = _make_state()
        state.pirate_boat.second.occupant = "p1"
        state.players[1].shares = [Share(ware=Ware.JADE, encumbered=False)]
        beliefs = infer_beliefs(state, "p0", signals=[ShareSignal(player_id="p1", ware=Ware.JADE)])
        self.assertEqual(_p_port_if_caught_on_13(state, beliefs, Ware.JADE), 1)


class TestPirateSlotEvIfTakenNow(unittest.TestCase):
    def test_empty_boat_values_taking_the_captain_role(self):
        state = _make_state()
        state.punts[0].ware = Ware.JADE
        state.punts[0].position = 6
        state.punts[0].status = PuntStatus.ON_ROUTE
        p_caught = position_outcomes(6, state.movement_rounds_total - state.movement_round_index)["caught_on_13"]
        expected = p_caught * PLUNDER_PAYOUTS[Ware.JADE] - PIRATE_PRICE
        self.assertEqual(pirate_slot_ev_if_taken_now(state), expected)

    def test_captain_already_aboard_values_taking_the_second_role_as_a_two_way_split(self):
        state = _make_state()
        state.punts[0].ware = Ware.JADE
        state.punts[0].position = 6
        state.punts[0].status = PuntStatus.ON_ROUTE
        state.pirate_boat.captain.occupant = "p0"

        p_caught = position_outcomes(6, state.movement_rounds_total - state.movement_round_index)["caught_on_13"]
        expected = p_caught * (PLUNDER_PAYOUTS[Ware.JADE] // 2) - PIRATE_PRICE
        self.assertEqual(pirate_slot_ev_if_taken_now(state), expected)

    def test_none_when_the_boat_is_already_fully_crewed(self):
        state = _make_state()
        state.pirate_boat.captain.occupant = "p0"
        state.pirate_boat.second.occupant = "p1"
        self.assertIsNone(pirate_slot_ev_if_taken_now(state))

    def test_ignores_punts_that_are_not_on_route(self):
        state = _make_state()
        state.punts[0].ware = Ware.JADE
        state.punts[0].status = PuntStatus.IN_PORT
        self.assertEqual(pirate_slot_ev_if_taken_now(state), -PIRATE_PRICE)

    def test_captain_role_includes_the_free_boarding_bonus(self):
        # Regression test: this is exactly what the user's own bug report
        # traced -- the board's displayed pirate EV looked much smaller
        # than what the bot actually acted on, because
        # pirate_captain_boarding_bonus (often the *larger* of the two
        # terms early in a voyage) was missing from it entirely.
        state = _make_state()
        state.movement_round_index = 0  # boarding opportunity still ahead
        punt = state.punts[0]
        punt.ware = Ware.JADE
        punt.ware_slots = Punt.new(punt.id, Ware.JADE).ware_slots  # real vacant slots
        punt.status = PuntStatus.ON_ROUTE
        punt.position = 6

        p_caught = position_outcomes(6, state.movement_rounds_total - state.movement_round_index)["caught_on_13"]
        plunder_ev = p_caught * PLUNDER_PAYOUTS[Ware.JADE] - PIRATE_PRICE
        boarding = pirate_captain_boarding_bonus(state)
        self.assertGreater(boarding, 0)  # sanity: this scenario has a real boarding chance
        self.assertEqual(pirate_slot_ev_if_taken_now(state), plunder_ev + boarding)

    def test_second_role_does_not_include_the_boarding_bonus(self):
        # Only the captain gets the free-boarding privilege -- see
        # pirate_captain_boarding_bonus's docstring.
        state = _make_state()
        state.movement_round_index = 0
        punt = state.punts[0]
        punt.ware = Ware.JADE
        punt.ware_slots = Punt.new(punt.id, Ware.JADE).ware_slots
        punt.status = PuntStatus.ON_ROUTE
        punt.position = 6
        state.pirate_boat.captain.occupant = "p0"  # only second is left to take

        p_caught = position_outcomes(6, state.movement_rounds_total - state.movement_round_index)["caught_on_13"]
        expected = p_caught * (PLUNDER_PAYOUTS[Ware.JADE] // 2) - PIRATE_PRICE
        self.assertEqual(pirate_slot_ev_if_taken_now(state), expected)


class TestPirateCaptainBoardingBonus(unittest.TestCase):
    def _punt_with_vacancy(self, state, index, ware, position, occupied_slots, total_slots):
        punt = state.punts[index]
        punt.ware = ware
        punt.position = position
        punt.status = PuntStatus.ON_ROUTE
        punt.ware_slots = [AccompliceSlot(price=1, occupant="pX") for _ in range(occupied_slots)] + [
            AccompliceSlot(price=1, occupant=None) for _ in range(total_slots - occupied_slots)
        ]
        return punt

    def test_zero_after_the_second_dice_throw_has_happened(self):
        state = _make_state()
        self._punt_with_vacancy(state, 0, Ware.GINSENG, 8, occupied_slots=1, total_slots=3)
        state.movement_round_index = 2
        self.assertEqual(pirate_captain_boarding_bonus(state), 0)

    def test_nonzero_in_the_first_accomplice_round(self):
        state = _make_state()
        self._punt_with_vacancy(state, 0, Ware.GINSENG, 8, occupied_slots=1, total_slots=3)
        state.movement_round_index = 0
        self.assertGreater(pirate_captain_boarding_bonus(state), 0)

    def test_matches_a_hand_assembled_single_punt_computation(self):
        state = _make_state()
        self._punt_with_vacancy(state, 0, Ware.GINSENG, 8, occupied_slots=1, total_slots=3)
        state.movement_round_index = 1  # 1 round until the boarding roll
        p_boardable = position_outcomes(8, 1)["caught_on_13"]
        expected = p_boardable * (PLUNDER_PAYOUTS[Ware.GINSENG] // 2)  # occupied(1) + captain
        self.assertEqual(pirate_captain_boarding_bonus(state), expected)

    def test_sums_across_multiple_boardable_punts(self):
        state = _make_state()
        self._punt_with_vacancy(state, 0, Ware.GINSENG, 8, occupied_slots=1, total_slots=3)
        self._punt_with_vacancy(state, 1, Ware.NUTMEG, 5, occupied_slots=0, total_slots=3)
        state.movement_round_index = 0
        one = pirate_captain_boarding_bonus(state)

        solo_state = _make_state()
        self._punt_with_vacancy(solo_state, 0, Ware.GINSENG, 8, occupied_slots=1, total_slots=3)
        solo_state.movement_round_index = 0
        solo = pirate_captain_boarding_bonus(solo_state)

        other_state = _make_state()
        self._punt_with_vacancy(other_state, 1, Ware.NUTMEG, 5, occupied_slots=0, total_slots=3)
        other_state.movement_round_index = 0
        other = pirate_captain_boarding_bonus(other_state)

        self.assertEqual(one, solo + other)

    def test_skips_a_fully_occupied_punt(self):
        state = _make_state()
        self._punt_with_vacancy(state, 0, Ware.GINSENG, 8, occupied_slots=3, total_slots=3)
        state.movement_round_index = 0
        self.assertEqual(pirate_captain_boarding_bonus(state), 0)

    def test_assumes_occupied_plus_one_not_full_capacity(self):
        # A ginseng punt with 1 of 3 slots taken: the bonus should value
        # the captain's free seat as splitting 2 ways (occupied + captain),
        # not 3 ways as project_final_occupancy's "assume full" would.
        state = _make_state()
        self._punt_with_vacancy(state, 0, Ware.GINSENG, 8, occupied_slots=1, total_slots=3)
        state.movement_round_index = 1
        p_boardable = position_outcomes(8, 1)["caught_on_13"]
        two_way_share = p_boardable * (PLUNDER_PAYOUTS[Ware.GINSENG] // 2)
        three_way_share = p_boardable * (PLUNDER_PAYOUTS[Ware.GINSENG] // 3)
        self.assertEqual(pirate_captain_boarding_bonus(state), two_way_share)
        self.assertNotEqual(pirate_captain_boarding_bonus(state), three_way_share)


class TestBoardingBonusOnlyAppliesToTheCaptain(unittest.TestCase):
    def test_captain_gets_the_bonus_second_does_not(self):
        state = _make_state()
        state.punts[0].ware = Ware.GINSENG
        state.punts[0].position = 8
        state.punts[0].status = PuntStatus.ON_ROUTE
        state.punts[0].ware_slots = [AccompliceSlot(price=1, occupant=None)] * 3
        state.movement_round_index = 0
        state.pirate_boat.captain.occupant = "p0"
        state.pirate_boat.second.occupant = "p1"
        beliefs = infer_beliefs(state, "p0")

        bonus = pirate_captain_boarding_bonus(state)
        self.assertGreater(bonus, 0)

        punts = [(Ware.GINSENG, 8, 3)]
        plain_pirate_part = pirate_expected_payout(punts, pirate_count=2)
        self.assertEqual(expected_accomplice_return(state, beliefs, "p0"), plain_pirate_part + bonus)
        self.assertEqual(expected_accomplice_return(state, beliefs, "p1"), plain_pirate_part)


class TestExpectedAccompliceReturnAutoDerivesPirateThreat(unittest.TestCase):
    def test_p_safe_if_caught_defaults_to_the_real_pirate_presence(self):
        state = _make_state()
        state.punts[0].ware = Ware.GINSENG
        state.punts[0].position = 8
        state.punts[0].status = PuntStatus.ON_ROUTE
        state.punts[0].ware_slots = [AccompliceSlot(price=1, occupant="p1")] * 1 + [
            AccompliceSlot(price=2),
            AccompliceSlot(price=3),
        ]
        beliefs = infer_beliefs(state, "p0")

        safe = expected_accomplice_return(state, beliefs, "p1")
        state.pirate_boat.captain.occupant = "p0"
        endangered = expected_accomplice_return(state, beliefs, "p1")
        self.assertLess(endangered, safe)
        self.assertEqual(endangered, ware_slot_expected_payout(Ware.GINSENG, 8, 3, 3, p_safe_if_caught=0))

    def test_explicit_override_still_wins_over_auto_derivation(self):
        state = _make_state()
        state.punts[0].ware = Ware.GINSENG
        state.punts[0].position = 8
        state.punts[0].status = PuntStatus.ON_ROUTE
        state.punts[0].ware_slots = [AccompliceSlot(price=1, occupant="p1"), AccompliceSlot(price=2), AccompliceSlot(price=3)]
        state.pirate_boat.captain.occupant = "p0"  # real threat exists...
        beliefs = infer_beliefs(state, "p0")

        # ...but an explicit override is still honored, e.g. for a
        # counterfactual "what if I ignore pirate risk" comparison.
        forced_safe = expected_accomplice_return(state, beliefs, "p1", p_safe_if_caught=1)
        self.assertEqual(forced_safe, ware_slot_expected_payout(Ware.GINSENG, 8, 3, 3, p_safe_if_caught=1))


class TestActionImpact(unittest.TestCase):
    def _state_with_rival(self):
        state = _make_state()
        state.punts[0].ware = Ware.GINSENG
        state.punts[0].position = 8
        state.punts[0].status = PuntStatus.ON_ROUTE
        state.punts[0].ware_slots = [
            AccompliceSlot(price=1, occupant="p1"),
            AccompliceSlot(price=2),
            AccompliceSlot(price=3),
        ]
        state.players[1].cash = 500  # guarantee p1 counts as a rival
        return state

    def test_does_not_mutate_the_original_state(self):
        state = self._state_with_rival()
        beliefs = infer_beliefs(state, "p0")
        action_impact(state, beliefs, "p0", apply_pirate_placement("captain", "p0"))
        self.assertIsNone(state.pirate_boat.captain.occupant)
        self.assertEqual(state.players[0].cash, 30)

    def test_no_op_action_gives_zero_gains(self):
        # p1 is a rival on cash alone (500); p2 also counts as a rival now
        # that DEFENSIVE_WEALTH_MARGIN pads every opponent's estimate --
        # p2's padded wealth (30 + 15) exceeds p0's own exact 30.
        state = self._state_with_rival()
        beliefs = infer_beliefs(state, "p0")
        impact = action_impact(state, beliefs, "p0", lambda s: None)
        self.assertEqual(impact.my_gain, 0)
        self.assertEqual(impact.rival_gains, {"p1": Fraction(0), "p2": Fraction(0)})
        expected_total = -sum(rev(state, beliefs, "p0", r) for r in ("p1", "p2"))
        self.assertEqual(impact.total_rev_after, expected_total)

    def test_taking_a_pirate_slot_hurts_a_rivals_endangered_ware_accomplice(self):
        # p1's ginseng accomplice goes from "safe if caught" to "certain
        # plunder" the moment p0 becomes captain -- a real, negative effect
        # on a rival caused by an action that isn't p1's own.
        state = self._state_with_rival()
        beliefs = infer_beliefs(state, "p0")
        impact = action_impact(state, beliefs, "p0", apply_pirate_placement("captain", "p0"))
        self.assertLess(impact.rival_gains["p1"], 0)

    def test_total_rev_after_matches_the_users_formula(self):
        state = self._state_with_rival()
        state.players[2].cash = 400  # p2 is a rival too
        beliefs = infer_beliefs(state, "p0")
        impact = action_impact(state, beliefs, "p0", apply_pirate_placement("captain", "p0"))

        after = GameState.from_dict(state.to_dict())
        apply_pirate_placement("captain", "p0")(after)
        expected_total = sum(
            wealth_estimate(after, beliefs, "p0") - wealth_estimate(after, beliefs, r) for r in ("p1", "p2")
        )
        self.assertEqual(impact.total_rev_after, expected_total)

    def test_rivals_are_fixed_from_before_the_action(self):
        # p1 stops being a rival once the action lands (their punt's value
        # craters), but they were a rival *before* the action, so they
        # still count in total_rev_after.
        state = self._state_with_rival()
        beliefs = infer_beliefs(state, "p0")
        rivals_before = identify_rivals(state, beliefs, "p0")
        self.assertIn("p1", rivals_before)

        impact = action_impact(state, beliefs, "p0", apply_pirate_placement("captain", "p0"))
        self.assertIn("p1", impact.rival_gains)


class TestApplyWareSlotPlacement(unittest.TestCase):
    def test_occupies_the_cheapest_vacant_slot_and_deducts_its_price(self):
        state = _make_state()
        state.punts[0].ware = Ware.GINSENG
        state.punts[0].status = PuntStatus.ON_ROUTE
        state.punts[0].ware_slots = [
            AccompliceSlot(price=3, occupant=None),
            AccompliceSlot(price=1, occupant=None),
            AccompliceSlot(price=2, occupant=None),
        ]
        apply_ware_slot_placement(0, "p0")(state)
        self.assertEqual(state.punts[0].ware_slots[1].occupant, "p0")  # the price=1 slot
        self.assertEqual(state.players[0].cash, 30 - 1)

    def test_ignores_an_unknown_punt(self):
        state = _make_state()
        apply_ware_slot_placement(99, "p0")(state)  # should not raise

    def test_ignores_a_fully_occupied_punt(self):
        state = _make_state()
        state.punts[0].ware = Ware.GINSENG
        state.punts[0].status = PuntStatus.ON_ROUTE
        state.punts[0].ware_slots = [AccompliceSlot(price=1, occupant="p1")]
        apply_ware_slot_placement(0, "p0")(state)
        self.assertEqual(state.players[0].cash, 30)  # nothing charged


class TestJoiningARivalsPuntDoesNotAppearToDenyThemPesos(unittest.TestCase):
    def _rival_on_punt_state(self):
        state = _make_state()
        state.punts[0].ware = Ware.GINSENG
        state.punts[0].position = 8
        state.punts[0].status = PuntStatus.ON_ROUTE
        state.punts[0].ware_slots = [
            AccompliceSlot(price=1, occupant="p1"),
            AccompliceSlot(price=2, occupant=None),
            AccompliceSlot(price=3, occupant=None),
        ]
        state.players[1].cash = 500  # p1 is a rival
        return state

    def test_early_in_the_voyage_joining_does_not_reduce_a_rivals_wealth(self):
        # Plenty of placement turns remain, so the punt was already valued
        # as if it fills up regardless (project_final_occupancy) -- taking
        # the second slot doesn't change what p1 was already assumed to
        # get, so it shouldn't look like "denying" them anything.
        state = self._rival_on_punt_state()
        state.movement_round_index = 0
        beliefs = infer_beliefs(state, "p0")
        impact = action_impact(state, beliefs, "p0", apply_ware_slot_placement(0, "p0"))
        self.assertEqual(impact.rival_gains["p1"], 0)

    def test_in_the_final_turns_joining_genuinely_dilutes_a_rival(self):
        # No placement time remains for "it would have filled anyway" to
        # still be a safe assumption -- an actual join here really is a
        # real, correctly-modeled cost to p1.
        state = self._rival_on_punt_state()
        state.players[0].is_harbor_master = False
        state.players[1].is_harbor_master = True  # p1 goes first this round
        state.movement_round_index = 2
        state.current_turn_player_id = "p0"  # p0 is the last to act this round
        beliefs = infer_beliefs(state, "p0")
        impact = action_impact(state, beliefs, "p0", apply_ware_slot_placement(0, "p0"))
        self.assertLess(impact.rival_gains["p1"], 0)


class TestApplyPiratePlacement(unittest.TestCase):
    def test_occupies_the_slot_and_deducts_the_price(self):
        state = _make_state()
        apply_pirate_placement("captain", "p0")(state)
        self.assertEqual(state.pirate_boat.captain.occupant, "p0")
        self.assertEqual(state.players[0].cash, 30 - 5)

    def test_second_slot(self):
        state = _make_state()
        apply_pirate_placement("second", "p1")(state)
        self.assertEqual(state.pirate_boat.second.occupant, "p1")
        self.assertIsNone(state.pirate_boat.captain.occupant)

    def test_rejects_unknown_role(self):
        with self.assertRaises(ValueError):
            apply_pirate_placement("first-mate", "p0")


class TestApplyDockSlotPlacement(unittest.TestCase):
    def test_occupies_slot_a_first_and_deducts_its_price(self):
        state = _make_state()
        apply_dock_slot_placement("port", "p0")(state)
        self.assertEqual(state.port.slots["A"].occupant, "p0")
        self.assertEqual(state.players[0].cash, 30 - state.port.slots["A"].price)

    def test_falls_through_to_b_then_c(self):
        state = _make_state()
        state.port.slots["A"].occupant = "p1"
        apply_dock_slot_placement("port", "p0")(state)
        self.assertEqual(state.port.slots["B"].occupant, "p0")

    def test_shipyard_uses_the_shipyard_slots(self):
        state = _make_state()
        apply_dock_slot_placement("shipyard", "p0")(state)
        self.assertEqual(state.shipyard.slots["A"].occupant, "p0")
        self.assertIsNone(state.port.slots["A"].occupant)

    def test_noop_when_every_slot_is_occupied(self):
        state = _make_state()
        for key in ("A", "B", "C"):
            state.port.slots[key].occupant = "p1"
        apply_dock_slot_placement("port", "p0")(state)
        self.assertEqual(state.players[0].cash, 30)

    def test_rejects_unknown_dock(self):
        with self.assertRaises(ValueError):
            apply_dock_slot_placement("harbor", "p0")


class TestApplyPilotPlacement(unittest.TestCase):
    def test_occupies_the_slot_and_deducts_its_price(self):
        state = _make_state()
        apply_pilot_placement("small", "p0")(state)
        self.assertEqual(state.pilot_island.small.occupant, "p0")
        self.assertEqual(state.players[0].cash, 30 - state.pilot_island.small.price)

    def test_large_slot(self):
        state = _make_state()
        apply_pilot_placement("large", "p1")(state)
        self.assertEqual(state.pilot_island.large.occupant, "p1")
        self.assertEqual(state.players[1].cash, 30 - state.pilot_island.large.price)

    def test_rejects_unknown_size(self):
        with self.assertRaises(ValueError):
            apply_pilot_placement("medium", "p0")


class TestApplyInsurancePlacement(unittest.TestCase):
    def test_occupies_the_office_and_pays_immediately(self):
        state = _make_state()
        apply_insurance_placement("p0")(state)
        self.assertEqual(state.insurance.occupant, "p0")
        self.assertEqual(state.players[0].cash, 30 + state.insurance.payment)


class TestInsuranceEV(unittest.TestCase):
    def test_matches_a_hand_assembled_computation(self):
        state = _make_state()
        state.punts[0].ware = Ware.GINSENG
        state.punts[0].position = 3
        state.punts[0].status = PuntStatus.ON_ROUTE
        state.punts[1].ware = None
        state.punts[2].ware = None

        p_wreck = punt_shipyard_probability(3, 3)
        expected = Fraction(10) - p_wreck * 6  # 1 wreck costs 6
        self.assertEqual(insurance_ev(state), expected)

    def test_a_punt_already_in_the_shipyard_counts_as_a_certain_wreck(self):
        state = _make_state()
        state.punts[0].ware = Ware.GINSENG
        state.punts[0].status = PuntStatus.IN_SHIPYARD
        state.punts[1].ware = None
        state.punts[2].ware = None
        self.assertEqual(insurance_ev(state), Fraction(10) - 6)

    def test_a_punt_in_port_or_captured_never_contributes_wreck_risk(self):
        state = _make_state()
        state.punts[0].ware = Ware.GINSENG
        state.punts[0].position = 13
        state.punts[0].status = PuntStatus.IN_PORT
        state.punts[1].ware = None
        state.punts[2].ware = None
        self.assertEqual(insurance_ev(state), Fraction(10))

    def test_no_loaded_punts_gives_the_full_payment(self):
        state = _make_state()
        for punt in state.punts:
            punt.ware = None
        self.assertEqual(insurance_ev(state), Fraction(10))

    def test_multiple_at_risk_punts_use_the_poisson_binomial_distribution(self):
        state = _make_state()
        state.punts[0].ware = Ware.GINSENG
        state.punts[0].position = 3
        state.punts[0].status = PuntStatus.ON_ROUTE
        state.punts[1].ware = Ware.NUTMEG
        state.punts[1].position = 2
        state.punts[1].status = PuntStatus.ON_ROUTE
        state.punts[2].ware = None

        p1 = punt_shipyard_probability(3, 3)
        p2 = punt_shipyard_probability(2, 3)
        dist = dock_fill_distribution([p1, p2])
        expected_cost = sum(dist.get(k, Fraction(0)) * INSURANCE_SHIPYARD_COST.get(k, 0) for k in (1, 2))
        self.assertEqual(insurance_ev(state), Fraction(10) - expected_cost)


class TestApplyPilotMove(unittest.TestCase):
    def test_moves_an_on_route_punt(self):
        state = _make_state()
        state.punts[0].ware = Ware.GINSENG
        state.punts[0].position = 5
        state.punts[0].status = PuntStatus.ON_ROUTE
        apply_pilot_move(0, 2)(state)
        self.assertEqual(state.punts[0].position, 7)

    def test_clamps_at_zero(self):
        state = _make_state()
        state.punts[0].ware = Ware.GINSENG
        state.punts[0].position = 1
        state.punts[0].status = PuntStatus.ON_ROUTE
        apply_pilot_move(0, -5)(state)
        self.assertEqual(state.punts[0].position, 0)

    def test_overshoot_docks_immediately(self):
        state = _make_state()
        state.punts[0].ware = Ware.GINSENG
        state.punts[0].position = 12
        state.punts[0].status = PuntStatus.ON_ROUTE
        apply_pilot_move(0, 2)(state)
        self.assertEqual(state.punts[0].status, PuntStatus.IN_PORT)
        self.assertEqual(state.punts[0].position, 13)

    def test_ignores_a_punt_not_on_route(self):
        state = _make_state()
        state.punts[0].ware = Ware.GINSENG
        state.punts[0].position = 5
        state.punts[0].status = PuntStatus.IN_SHIPYARD
        apply_pilot_move(0, 2)(state)
        self.assertEqual(state.punts[0].position, 5)

    def test_ignores_an_unknown_punt_id(self):
        state = _make_state()
        apply_pilot_move(99, 2)(state)  # should not raise


class TestPilotMoveCandidates(unittest.TestCase):
    def _two_eligible_state(self):
        state = _make_state()
        state.punts[0].ware = Ware.GINSENG
        state.punts[0].status = PuntStatus.ON_ROUTE
        state.punts[1].ware = Ware.NUTMEG
        state.punts[1].status = PuntStatus.ON_ROUTE
        state.punts[2].ware = None
        return state

    def test_small_pilot_offers_plus_and_minus_one_per_eligible_punt_plus_skip(self):
        state = self._two_eligible_state()
        candidates = pilot_move_candidates(state, "small")
        self.assertEqual(len(candidates), 1 + 2 * 2)  # skip + (+1/-1) per punt

    def test_large_pilot_adds_single_punt_by_two_and_two_punt_combinations(self):
        state = self._two_eligible_state()
        candidates = pilot_move_candidates(state, "large")
        # skip + (+2/-2) per punt + 1 pair * 4 direction combinations
        self.assertEqual(len(candidates), 1 + 2 * 2 + 1 * 4)

    def test_ignores_unloaded_and_non_on_route_punts(self):
        state = _make_state()
        state.punts[0].ware = Ware.GINSENG
        state.punts[0].status = PuntStatus.ON_ROUTE
        state.punts[1].ware = Ware.NUTMEG
        state.punts[1].status = PuntStatus.IN_PORT
        state.punts[2].ware = None
        candidates = pilot_move_candidates(state, "small")
        self.assertEqual(len(candidates), 1 + 2)  # only punt 0 is eligible

    def test_rejects_unknown_pilot_size(self):
        state = _make_state()
        with self.assertRaises(ValueError):
            pilot_move_candidates(state, "medium")


class TestBestPilotMoveAndPilotSlotValue(unittest.TestCase):
    def _rival_punt_state(self, position):
        state = _make_state()
        state.punts[0].ware = Ware.GINSENG
        state.punts[0].position = position
        state.punts[0].status = PuntStatus.ON_ROUTE
        state.punts[0].ware_slots = [
            AccompliceSlot(price=1, occupant="p1"),
            AccompliceSlot(price=2),
            AccompliceSlot(price=3),
        ]
        state.punts[1].ware = None
        state.punts[2].ware = None
        state.movement_round_index = 2  # matches the pilot phase's real timing
        state.players[1].cash = 500  # guarantee p1 counts as a rival
        return state

    def test_picks_the_move_that_most_hurts_the_rival(self):
        # I hold no stake in p1's punt, so my own wealth is unaffected by
        # any of these choices -- the best move is whichever one most
        # lowers p1's arrival odds (moving their punt backward), since
        # that's what maximizes total_rev_after.
        state = self._rival_punt_state(position=11)
        beliefs = infer_beliefs(state, "p0")
        action, impact = best_pilot_move(state, beliefs, "p0", "small")

        forward = action_impact(state, beliefs, "p0", apply_pilot_move(0, 1))
        backward = action_impact(state, beliefs, "p0", apply_pilot_move(0, -1))
        skip = action_impact(state, beliefs, "p0", lambda s: None)

        self.assertEqual(
            impact.total_rev_after,
            max(forward.total_rev_after, backward.total_rev_after, skip.total_rev_after),
        )
        self.assertEqual(impact.total_rev_after, backward.total_rev_after)

    def test_pilot_slot_value_matches_best_pilot_move(self):
        state = self._rival_punt_state(position=11)
        beliefs = infer_beliefs(state, "p0")
        _, impact = best_pilot_move(state, beliefs, "p0", "small")
        value = pilot_slot_value(state, beliefs, "p0", "small")
        self.assertEqual(value, impact.total_rev_after)

    def test_reassessment_reflects_the_current_board_not_a_cached_answer(self):
        early = self._rival_punt_state(position=6)
        late = self._rival_punt_state(position=11)
        value_early = pilot_slot_value(early, infer_beliefs(early, "p0"), "p0", "small")
        value_late = pilot_slot_value(late, infer_beliefs(late, "p0"), "p0", "small")
        self.assertNotEqual(value_early, value_late)


class TestOccupiedPilotSlots(unittest.TestCase):
    def test_empty_when_nothing_occupied(self):
        state = _make_state()
        self.assertEqual(occupied_pilot_slots(state), [])

    def test_small_only(self):
        state = _make_state()
        state.pilot_island.small.occupant = "p1"
        self.assertEqual(occupied_pilot_slots(state), [("small", "p1")])

    def test_large_only(self):
        state = _make_state()
        state.pilot_island.large.occupant = "p2"
        self.assertEqual(occupied_pilot_slots(state), [("large", "p2")])

    def test_both_lists_small_before_large(self):
        state = _make_state()
        state.pilot_island.small.occupant = "p1"
        state.pilot_island.large.occupant = "p2"
        self.assertEqual(occupied_pilot_slots(state), [("small", "p1"), ("large", "p2")])


class TestPredictPilotMove(unittest.TestCase):
    def _state_where_p1_pilots_and_p2_is_p1s_rival(self, position=11):
        # p2 holds a ginseng accomplice; p2 is wealthy enough to be p1's
        # rival (not necessarily p0's -- p0 is a bystander here, cash 30).
        state = _make_state()
        state.punts[0].ware = Ware.GINSENG
        state.punts[0].position = position
        state.punts[0].status = PuntStatus.ON_ROUTE
        state.punts[0].ware_slots = [
            AccompliceSlot(price=1, occupant="p2"),
            AccompliceSlot(price=2),
            AccompliceSlot(price=3),
        ]
        state.punts[1].ware = None
        state.punts[2].ware = None
        state.movement_round_index = 2
        state.pilot_island.small.occupant = "p1"
        state.players[2].cash = 500  # p2 is p1's rival, not necessarily p0's
        return state

    def test_matches_best_pilot_move_run_as_the_pilot_holder(self):
        state = self._state_where_p1_pilots_and_p2_is_p1s_rival()
        predicted = predict_pilot_move(state, "p1", "small")

        pilot_beliefs = infer_beliefs(state, "p1")
        expected_action, _ = best_pilot_move(state, pilot_beliefs, "p1", "small")

        after_predicted = GameState.from_dict(state.to_dict())
        predicted(after_predicted)
        after_expected = GameState.from_dict(state.to_dict())
        expected_action(after_expected)
        self.assertEqual(after_predicted.punts[0].position, after_expected.punts[0].position)

    def test_reflects_the_pilots_own_rival_not_the_predictors(self):
        # p1 (the pilot) sees p2 as a rival and should push p2's punt
        # backward -- regardless of who's asking, since the prediction is
        # always computed from p1's own point of view.
        state = self._state_where_p1_pilots_and_p2_is_p1s_rival()
        action = predict_pilot_move(state, "p1", "small")
        after = GameState.from_dict(state.to_dict())
        action(after)
        self.assertLess(after.punts[0].position, state.punts[0].position)

    def test_forwards_signals_to_the_pilots_beliefs(self):
        state = self._state_where_p1_pilots_and_p2_is_p1s_rival()
        state.players[2].shares = [Share(ware=Ware.JADE), Share(ware=Ware.JADE)]
        signals = [ShareSignal(player_id="p2", ware=Ware.JADE, source="purchase")]

        with_signal = predict_pilot_move(state, "p1", "small", signals=signals)
        pilot_beliefs = infer_beliefs(state, "p1", signals)
        expected_action, _ = best_pilot_move(state, pilot_beliefs, "p1", "small")

        after_with = GameState.from_dict(state.to_dict())
        with_signal(after_with)
        after_expected = GameState.from_dict(state.to_dict())
        expected_action(after_expected)
        self.assertEqual(after_with.punts[0].position, after_expected.punts[0].position)


class TestWithPredictedPilotMoves(unittest.TestCase):
    def test_no_pilots_occupied_leaves_the_board_unchanged(self):
        state = _make_state()
        state.punts[0].ware = Ware.GINSENG
        state.punts[0].position = 8
        state.punts[0].status = PuntStatus.ON_ROUTE
        predicted = with_predicted_pilot_moves(state)
        self.assertEqual(predicted.punts[0].position, 8)

    def test_does_not_mutate_the_original_state(self):
        state = _make_state()
        state.punts[0].ware = Ware.GINSENG
        state.punts[0].position = 11
        state.punts[0].status = PuntStatus.ON_ROUTE
        state.punts[0].ware_slots = [AccompliceSlot(price=1, occupant="p2")]
        state.punts[1].ware = None
        state.punts[2].ware = None
        state.movement_round_index = 2
        state.pilot_island.small.occupant = "p1"
        state.players[2].cash = 500

        with_predicted_pilot_moves(state)
        self.assertEqual(state.punts[0].position, 11)

    def test_folds_in_a_single_occupied_pilots_prediction(self):
        state = _make_state()
        state.punts[0].ware = Ware.GINSENG
        state.punts[0].position = 11
        state.punts[0].status = PuntStatus.ON_ROUTE
        state.punts[0].ware_slots = [AccompliceSlot(price=1, occupant="p2")]
        state.punts[1].ware = None
        state.punts[2].ware = None
        state.movement_round_index = 2
        state.pilot_island.small.occupant = "p1"
        state.players[2].cash = 500

        predicted = with_predicted_pilot_moves(state)
        action = predict_pilot_move(state, "p1", "small")
        expected = GameState.from_dict(state.to_dict())
        action(expected)
        self.assertEqual(predicted.punts[0].position, expected.punts[0].position)

    def test_large_pilots_prediction_sees_the_small_pilots_result(self):
        # Both pilots occupied by different players. If small moves punt 0
        # first, large's own prediction (also evaluated against punt 0)
        # should be computed from that already-shifted position, not the
        # original -- confirmed by comparing against manually chaining the
        # two predictions in the same order.
        state = _make_state()
        state.punts[0].ware = Ware.GINSENG
        state.punts[0].position = 11
        state.punts[0].status = PuntStatus.ON_ROUTE
        state.punts[0].ware_slots = [AccompliceSlot(price=1, occupant="p2")]
        state.punts[1].ware = None
        state.punts[2].ware = None
        state.movement_round_index = 2
        state.pilot_island.small.occupant = "p1"
        state.pilot_island.large.occupant = "p2"
        state.players[1].cash = 500  # p1 is a rival too, so p2's large pilot has a reason to act

        predicted = with_predicted_pilot_moves(state)

        chained = GameState.from_dict(state.to_dict())
        small_action = predict_pilot_move(chained, "p1", "small")
        small_action(chained)
        large_action = predict_pilot_move(chained, "p2", "large")
        large_action(chained)

        self.assertEqual(predicted.punts[0].position, chained.punts[0].position)


if __name__ == "__main__":
    unittest.main()
