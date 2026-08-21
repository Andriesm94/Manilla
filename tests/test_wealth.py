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
    rev,
    rev_adjusted_score,
    wealth_estimate,
)


def _make_state():
    state = GameState.new_default_game(["Me", "P1", "P2"])
    for p in state.players:
        p.shares = []
    state.phase = Phase.ACCOMPLICE_ROUND
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

    def test_on_route_ware_slot_matches_the_underlying_ev_function(self):
        state = self._rigged_state()
        expected = ware_slot_expected_payout(Ware.GINSENG, 8, 3, accomplices_on_punt=1)
        self.assertEqual(expected_accomplice_return(state, "p1"), expected)

    def test_docked_ware_slot_pays_out_with_certainty(self):
        state = self._rigged_state()
        # Silk punt already in port: certain payout, no dice math needed --
        # 30 (silk's plunder payout) split across its 1 occupied slot, plus
        # p0's pirate captaincy over the two punts still at sea.
        pirate_part = pirate_expected_payout(
            [(Ware.GINSENG, 8, 3), (Ware.NUTMEG, 5, 3)], pirate_count=1
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
