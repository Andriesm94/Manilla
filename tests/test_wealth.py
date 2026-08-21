import os
import sys
from fractions import Fraction
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from manilla.engine.models import (
    AccompliceSlot,
    GameState,
    Phase,
    Punt,
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
    encumbered_penalty,
    expected_accomplice_return,
    identify_rivals,
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
        # if split 3 ways, not kept whole for p1 alone.
        state = self._rigged_state()
        expected = ware_slot_expected_payout(Ware.GINSENG, 8, 3, accomplices_on_punt=3)
        self.assertEqual(expected_accomplice_return(state, "p1"), expected)

    def test_docked_ware_slot_pays_out_with_certainty(self):
        state = self._rigged_state()
        # Silk punt already in port: certain payout, no dice math needed --
        # 30 (silk's plunder payout) split across its 1 occupied slot
        # (already resolved, not projected). p0's pirate captaincy is still
        # projected to end up shared with a second pirate, though.
        pirate_part = pirate_expected_payout(
            [(Ware.GINSENG, 8, 3), (Ware.NUTMEG, 5, 3)], pirate_count=2
        )
        self.assertEqual(expected_accomplice_return(state, "p0"), Fraction(30) + pirate_part)

    def test_last_two_turns_of_the_final_round_use_actual_occupancy(self):
        state = self._rigged_state()
        state.movement_round_index = 2  # about to trigger the third dice throw -- 1 round left
        state.current_turn_player_id = "p2"  # last of 3 players in this round's rotation
        # No more placement chances remain -- p1's slot is valued alone.
        expected = ware_slot_expected_payout(Ware.GINSENG, 8, 1, accomplices_on_punt=1)
        self.assertEqual(expected_accomplice_return(state, "p1"), expected)

    def test_early_in_the_final_round_still_projects_full_occupancy(self):
        state = self._rigged_state()
        state.movement_round_index = 2
        state.current_turn_player_id = "p0"  # harbor master, first to act -- 3 turns still remain
        expected = ware_slot_expected_payout(Ware.GINSENG, 8, 1, accomplices_on_punt=3)
        self.assertEqual(expected_accomplice_return(state, "p1"), expected)

    def test_exactly_two_turns_remaining_is_the_boundary(self):
        state = self._rigged_state()
        state.movement_round_index = 2
        state.current_turn_player_id = "p1"  # 2nd of 3 -- exactly 2 turns remain, including this one
        expected = ware_slot_expected_payout(Ware.GINSENG, 8, 1, accomplices_on_punt=1)
        self.assertEqual(expected_accomplice_return(state, "p1"), expected)

    def test_pirate_boat_projection_also_respects_the_last_two_turns(self):
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


if __name__ == "__main__":
    unittest.main()
