import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from manilla.engine.models import AccompliceSlot, GameState, Phase, Punt, PuntStatus, Ware
from manilla.engine.beliefs import infer_beliefs
from manilla.engine.policy import AccompliceChoice, choose_accomplice_action
from manilla.engine.wealth import (
    action_impact,
    apply_dock_slot_placement,
    apply_pirate_placement,
    apply_ware_slot_placement,
    insurance_ev,
    pilot_slot_value,
)


def _make_state():
    state = GameState.new_default_game(["Me", "P1", "P2"])
    for p in state.players:
        p.shares = []
    state.players[0].is_harbor_master = True
    state.phase = Phase.ACCOMPLICE_ROUND
    state.current_turn_player_id = "p0"
    state.movement_round_index = 0
    return state


class TestChooseAccompliceAction(unittest.TestCase):
    def test_none_when_nothing_is_available(self):
        state = _make_state()
        for punt in state.punts:
            punt.ware = None
        state.pirate_boat.captain.occupant = "p1"
        state.pirate_boat.second.occupant = "p2"
        for key in ("A", "B", "C"):
            state.port.slots[key].occupant = "p1"
            state.shipyard.slots[key].occupant = "p1"
        state.pilot_island.small.occupant = "p1"
        state.pilot_island.large.occupant = "p1"
        state.insurance.occupant = "p1"
        beliefs = infer_beliefs(state, "p0")
        self.assertIsNone(choose_accomplice_action(state, beliefs, "p0"))

    def test_picks_whichever_candidate_scores_highest(self):
        state = _make_state()
        state.punts[0].ware = Ware.JADE
        state.punts[0].position = 6
        state.punts[0].status = PuntStatus.ON_ROUTE
        state.punts[1].ware = Ware.NUTMEG
        state.punts[1].position = 3
        state.punts[1].status = PuntStatus.ON_ROUTE
        state.punts[2].ware = None
        beliefs = infer_beliefs(state, "p0")

        choice = choose_accomplice_action(state, beliefs, "p0")

        # Independently compute every candidate's score the same way
        # choose_accomplice_action does, and confirm the chosen one is a
        # genuine max, not just "a" candidate. REV-based candidates
        # (everything but insurance) are baselined against a no-op's
        # total_rev_after -- see policy.py's module docstring on why that
        # baseline has to match for this to be a fair comparison.
        no_op_baseline = action_impact(state, beliefs, "p0", lambda s: None).total_rev_after
        scores = {}
        for punt in state.punts:
            if punt.ware is None or punt.status != PuntStatus.ON_ROUTE:
                continue
            if any(s.occupant is None for s in punt.ware_slots):
                mutator = apply_ware_slot_placement(punt.id, "p0")
                scores[("ware", punt.id)] = (
                    action_impact(state, beliefs, "p0", mutator).total_rev_after - no_op_baseline
                )
        for dock_name, dock in (("port", state.port), ("shipyard", state.shipyard)):
            mutator = apply_dock_slot_placement(dock_name, "p0")
            scores[("dock", dock_name)] = (
                action_impact(state, beliefs, "p0", mutator).total_rev_after - no_op_baseline
            )
        scores[("pirate", "captain")] = (
            action_impact(state, beliefs, "p0", apply_pirate_placement("captain", "p0")).total_rev_after
            - no_op_baseline
        )
        scores[("pilot", "small")] = (
            pilot_slot_value(state, beliefs, "p0", "small") - no_op_baseline - state.pilot_island.small.price
        )
        scores[("pilot", "large")] = (
            pilot_slot_value(state, beliefs, "p0", "large") - no_op_baseline - state.pilot_island.large.price
        )
        scores[("insurance", None)] = insurance_ev(state)

        best_key = max(scores, key=lambda k: scores[k])
        best_kind, best_id = best_key

        self.assertEqual(choice.kind, best_kind)
        if best_kind == "ware":
            self.assertEqual(choice.punt_id, best_id)
        elif best_kind == "dock":
            self.assertEqual(choice.dock, best_id)
        elif best_kind == "pirate":
            self.assertEqual(choice.pirate_role, best_id)
        elif best_kind == "pilot":
            self.assertEqual(choice.pilot_size, best_id)

    def test_many_padded_rivals_do_not_bias_everything_toward_insurance(self):
        # Regression test for the bug this baseline-subtraction fixes:
        # DEFENSIVE_WEALTH_MARGIN pads every rival's estimate, so with
        # several rivals present, action_impact's total_rev_after for any
        # ware/dock/pirate/pilot candidate is dragged deeply negative by a
        # constant that has nothing to do with the action -- insurance_ev
        # carries no such offset, so before the fix it would win by
        # default any time 2+ rivals existed, regardless of whether it
        # was actually the better choice. A near-certain, uncontested,
        # well-paying ware placement should still be able to beat a
        # merely-decent insurance option even with three rivals present.
        state = GameState.new_default_game(["Me", "P1", "P2", "P3"])
        for p in state.players:
            p.shares = []
        state.players[0].is_harbor_master = True
        state.phase = Phase.ACCOMPLICE_ROUND
        state.current_turn_player_id = "p0"
        state.punts[0].ware = Ware.JADE
        state.punts[0].ware_slots = Punt.new(0, Ware.JADE).ware_slots
        state.punts[0].position = 12  # 1 round left, near-certain arrival
        state.punts[0].status = PuntStatus.ON_ROUTE
        state.movement_round_index = 2
        state.punts[1].ware = None
        state.punts[2].ware = None
        for p in state.players[1:]:
            p.cash = 500  # everyone else is a rival

        beliefs = infer_beliefs(state, "p0")
        choice = choose_accomplice_action(state, beliefs, "p0")
        self.assertEqual(choice.kind, "ware")
        self.assertEqual(choice.punt_id, 0)

    def test_skips_a_fully_occupied_ware_punt(self):
        state = _make_state()
        state.punts[0].ware = Ware.GINSENG
        state.punts[0].position = 8
        state.punts[0].status = PuntStatus.ON_ROUTE
        state.punts[0].ware_slots = [AccompliceSlot(price=1, occupant="p1")] * 3
        state.punts[1].ware = None
        state.punts[2].ware = None
        # Remove every other option so a full ware punt would be the only
        # thing wrong if it were (incorrectly) offered.
        state.pirate_boat.captain.occupant = "p1"
        state.pirate_boat.second.occupant = "p1"
        state.pilot_island.small.occupant = "p1"
        state.pilot_island.large.occupant = "p1"
        state.insurance.occupant = "p1"
        beliefs = infer_beliefs(state, "p0")
        choice = choose_accomplice_action(state, beliefs, "p0")
        self.assertIn(choice.kind, ("dock",))  # only port/shipyard remain

    def test_offers_captain_before_second_never_both(self):
        state = _make_state()
        for punt in state.punts:
            punt.ware = None
        for key in ("A", "B", "C"):
            state.port.slots[key].occupant = "p1"
            state.shipyard.slots[key].occupant = "p1"
        state.pilot_island.small.occupant = "p1"
        state.pilot_island.large.occupant = "p1"
        state.insurance.occupant = "p1"
        beliefs = infer_beliefs(state, "p0")

        choice = choose_accomplice_action(state, beliefs, "p0")
        self.assertEqual(choice.kind, "pirate")
        self.assertEqual(choice.pirate_role, "captain")

        state.pirate_boat.captain.occupant = "p1"
        choice2 = choose_accomplice_action(state, beliefs, "p0")
        self.assertEqual(choice2.kind, "pirate")
        self.assertEqual(choice2.pirate_role, "second")

        state.pirate_boat.second.occupant = "p1"
        self.assertIsNone(choose_accomplice_action(state, beliefs, "p0"))


class TestAccompliceChoice(unittest.TestCase):
    def test_default_fields_are_none(self):
        choice = AccompliceChoice(kind="insurance")
        self.assertIsNone(choice.punt_id)
        self.assertIsNone(choice.dock)
        self.assertIsNone(choice.pirate_role)
        self.assertIsNone(choice.pilot_size)


if __name__ == "__main__":
    unittest.main()
