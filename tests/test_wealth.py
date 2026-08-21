import os
import sys
from fractions import Fraction
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from manilla.engine.models import (
    AccompliceSlot,
    GameState,
    Phase,
    PuntStatus,
    SHARE_REPAY_AMOUNT,
    Share,
    Ware,
)
from manilla.engine.beliefs import infer_beliefs
from manilla.engine.expected_value import (
    dock_slot_expected_payout,
    pirate_expected_payout,
    punt_port_probability,
    ware_slot_expected_payout,
)
from manilla.engine.wealth import (
    action_impact,
    apply_pilot_move,
    apply_pirate_placement,
    encumbered_penalty,
    expected_accomplice_return,
    identify_rivals,
    pirate_threat,
    project_final_occupancy,
    rev,
    rev_adjusted_score,
    wealth_estimate,
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
        expected = ware_slot_expected_payout(Ware.GINSENG, 8, 3, accomplices_on_punt=3, p_safe_if_caught=0)
        self.assertEqual(expected_accomplice_return(state, "p1"), expected)

    def test_docked_ware_slot_pays_out_with_certainty(self):
        state = self._rigged_state()
        # Silk punt already in port: certain payout, no dice math needed --
        # 30 (silk's plunder payout) split across its 1 occupied slot
        # (already resolved, not projected). p0's lone pirate captaincy is
        # valued using exactly the 1 pirate actually aboard -- no
        # projection, see test_pirate_valuation_never_projects_occupancy.
        pirate_part = pirate_expected_payout(
            [(Ware.GINSENG, 8, 3), (Ware.NUTMEG, 5, 3)], pirate_count=1
        )
        self.assertEqual(expected_accomplice_return(state, "p0"), Fraction(30) + pirate_part)

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

        for movement_round_index in (0, 1, 2):
            with self.subTest(movement_round_index=movement_round_index):
                state.movement_round_index = movement_round_index
                expected = pirate_expected_payout(
                    [(Ware.JADE, 6, 3 - movement_round_index)], pirate_count=1
                )
                self.assertEqual(expected_accomplice_return(state, "p0"), expected)

    def test_pirate_valuation_uses_current_count_when_two_are_aboard(self):
        # Two pirates actually aboard is a fact, not a projection -- the
        # split is valued as 2-way regardless of profitability or timing.
        state = _make_state()
        state.punts[0].ware = Ware.JADE
        state.punts[0].position = 6
        state.punts[0].status = PuntStatus.ON_ROUTE
        state.pirate_boat.captain.occupant = "p0"
        state.pirate_boat.second.occupant = "p1"

        expected = pirate_expected_payout([(Ware.JADE, 6, 3)], pirate_count=2)
        self.assertEqual(expected_accomplice_return(state, "p0"), expected)
        self.assertEqual(expected_accomplice_return(state, "p1"), expected)

    def test_last_two_turns_of_the_final_round_use_actual_occupancy(self):
        state = self._rigged_state()
        state.movement_round_index = 2  # about to trigger the third dice throw -- 1 round left
        state.current_turn_player_id = "p2"  # last of 3 players in this round's rotation
        # No more placement chances remain -- p1's slot is valued alone.
        expected = ware_slot_expected_payout(Ware.GINSENG, 8, 1, accomplices_on_punt=1, p_safe_if_caught=0)
        self.assertEqual(expected_accomplice_return(state, "p1"), expected)

    def test_early_in_the_final_round_still_projects_full_occupancy(self):
        state = self._rigged_state()
        state.movement_round_index = 2
        state.current_turn_player_id = "p0"  # harbor master, first to act -- 3 turns still remain
        expected = ware_slot_expected_payout(Ware.GINSENG, 8, 1, accomplices_on_punt=3, p_safe_if_caught=0)
        self.assertEqual(expected_accomplice_return(state, "p1"), expected)

    def test_exactly_two_turns_remaining_is_the_boundary(self):
        state = self._rigged_state()
        state.movement_round_index = 2
        state.current_turn_player_id = "p1"  # 2nd of 3 -- exactly 2 turns remain, including this one
        expected = ware_slot_expected_payout(Ware.GINSENG, 8, 1, accomplices_on_punt=1, p_safe_if_caught=0)
        self.assertEqual(expected_accomplice_return(state, "p1"), expected)

    def test_pirate_valuation_is_unaffected_by_the_last_two_turns_rule(self):
        # The last-two-turns exception is a ware-punt-only concept (see
        # project_final_occupancy) -- pirates don't have a growth
        # projection to fall back from in the first place.
        state = self._rigged_state()
        state.movement_round_index = 2
        state.current_turn_player_id = "p2"  # last turn of the round
        pirate_part = pirate_expected_payout(
            [(Ware.GINSENG, 8, 1), (Ware.NUTMEG, 5, 1)], pirate_count=1
        )
        self.assertEqual(expected_accomplice_return(state, "p0"), Fraction(30) + pirate_part)

    def test_dock_slot_matches_the_underlying_ev_function(self):
        state = self._rigged_state()
        arrival_probs = [punt_port_probability(8, 3), 1, punt_port_probability(5, 3)]
        expected = dock_slot_expected_payout("port", "A", arrival_probs)
        self.assertEqual(expected_accomplice_return(state, "p2"), expected)

    def test_settled_phases_return_zero_even_with_slots_occupied(self):
        state = self._rigged_state()
        state.phase = Phase.PROFIT_DISTRIBUTION
        self.assertEqual(expected_accomplice_return(state, "p1"), 0)
        self.assertEqual(expected_accomplice_return(state, "p0"), 0)

    def test_shipwrecked_ware_slot_pays_nothing(self):
        state = self._rigged_state()
        state.punts[0].status = PuntStatus.IN_SHIPYARD
        self.assertEqual(expected_accomplice_return(state, "p1"), 0)

    def test_player_with_no_slots_gets_zero(self):
        state = self._rigged_state()
        self.assertEqual(expected_accomplice_return(state, "nobody"), 0)


class TestProjectFinalOccupancy(unittest.TestCase):
    def test_defaults_to_full_capacity_outside_the_final_round(self):
        state = _make_state()
        state.movement_round_index = 0
        self.assertEqual(project_final_occupancy(state, current_occupied=1, max_slots=4), 4)

    def test_defaults_to_full_capacity_early_in_the_final_round(self):
        state = _make_state()
        state.movement_round_index = 2
        state.current_turn_player_id = "p0"  # harbor master, 3 turns remain
        self.assertEqual(project_final_occupancy(state, current_occupied=1, max_slots=4), 4)

    def test_uses_actual_occupancy_in_the_last_two_turns(self):
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
        state = _make_state()
        state.players[0].cash = 50
        state.players[1].cash = 53
        beliefs = infer_beliefs(state, "p0")
        self.assertEqual(rev(state, beliefs, "p0", "p1"), 3)

    def test_rev_can_go_negative_for_a_former_rival(self):
        state = _make_state()
        state.players[0].cash = 100
        state.players[1].cash = 53
        beliefs = infer_beliefs(state, "p0")
        self.assertEqual(rev(state, beliefs, "p0", "p1"), -47)


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

        safe = expected_accomplice_return(state, "p1")
        state.pirate_boat.captain.occupant = "p0"
        endangered = expected_accomplice_return(state, "p1")
        self.assertLess(endangered, safe)
        self.assertEqual(endangered, ware_slot_expected_payout(Ware.GINSENG, 8, 3, 3, p_safe_if_caught=0))

    def test_explicit_override_still_wins_over_auto_derivation(self):
        state = _make_state()
        state.punts[0].ware = Ware.GINSENG
        state.punts[0].position = 8
        state.punts[0].status = PuntStatus.ON_ROUTE
        state.punts[0].ware_slots = [AccompliceSlot(price=1, occupant="p1"), AccompliceSlot(price=2), AccompliceSlot(price=3)]
        state.pirate_boat.captain.occupant = "p0"  # real threat exists...

        # ...but an explicit override is still honored, e.g. for a
        # counterfactual "what if I ignore pirate risk" comparison.
        forced_safe = expected_accomplice_return(state, "p1", p_safe_if_caught=1)
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
        state = self._state_with_rival()
        beliefs = infer_beliefs(state, "p0")
        impact = action_impact(state, beliefs, "p0", lambda s: None)
        self.assertEqual(impact.my_gain, 0)
        self.assertEqual(impact.rival_gains, {"p1": Fraction(0)})
        self.assertEqual(impact.total_rev_after, -rev(state, beliefs, "p0", "p1"))

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


if __name__ == "__main__":
    unittest.main()
