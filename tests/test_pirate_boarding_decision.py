"""The user asked for the pirate captain's (and, if room remains, the
second pirate's) free mid-voyage boarding decision to compare the REV of
boarding for free against the REV of staying a pirate, rather than the
random policy's flat 60% chance -- captain decides first, second only
gets to decide afterward if a boardable slot is still actually open.

These drive the real BoardSetupApp._handle_pirate_boarding chain end to
end (not just the underlying wealth.best_pirate_boarding_move, which
tests/test_wealth.py already covers directly) to confirm the wiring
itself -- policy routing, turn order, and the "captain can use up the
only remaining slot before second gets a say" sequencing -- actually
works live.
"""

import os
import sys
import time
import tkinter as tk
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from manilla.ui.board_setup import BoardSetupApp
from manilla.engine.models import GameState, Punt, PuntStatus, SEA_ROUTE_LENGTH, Ware


def _pump(root, seconds):
    deadline = time.time() + seconds
    while time.time() < deadline:
        root.update()
        time.sleep(0.005)


class PirateBoardingDecisionTestCase(unittest.TestCase):
    def setUp(self):
        self.root = tk.Tk()
        self.root.withdraw()
        self.app = BoardSetupApp(self.root)
        self.app.pack(fill=tk.BOTH, expand=True)
        self.state = self.app.state_obj
        self.state.game_setup_confirmed = True
        self.state.movement_round_index = 2  # right after the second dice throw
        self.root.update_idletasks()

    def tearDown(self):
        self.root.destroy()

    def make_punt(self, idx, ware, position, ware_slots=None):
        punt = self.state.punts[idx]
        punt.ware = ware
        punt.ware_slots = ware_slots if ware_slots is not None else Punt.new(punt.id, ware).ware_slots
        punt.status = PuntStatus.ON_ROUTE
        punt.position = position
        return punt


class TestRevCaptainBoardingDecision(PirateBoardingDecisionTestCase):
    def test_rev_captain_boards_when_nothing_else_is_worth_staying_for(self):
        captain = self.state.players[0]
        captain.is_bot = True
        captain.policy = "rev"
        self.state.pirate_boat.captain.occupant = captain.id

        punt = self.make_punt(0, Ware.GINSENG, SEA_ROUTE_LENGTH)

        self.app._handle_pirate_boarding()
        _pump(self.root, 3)

        self.assertEqual(punt.ware_slots[0].occupant, captain.id)
        self.assertIsNone(self.state.pirate_boat.captain.occupant)

    def test_rev_captain_stays_when_another_on_route_punt_is_worth_more(self):
        captain = self.state.players[0]
        captain.is_bot = True
        captain.policy = "rev"
        self.state.pirate_boat.captain.occupant = captain.id

        # Boardable ginseng punt already 2/3 full -- boarding only nets a
        # 3-way split. Two richer, still-catchable-next-round punts (jade,
        # silk) are worth more to a pirate who stays put.
        boardable = self.make_punt(0, Ware.GINSENG, SEA_ROUTE_LENGTH)
        boardable.ware_slots[0].occupant = "p1"
        boardable.ware_slots[1].occupant = "p1"
        self.make_punt(1, Ware.JADE, 12)
        self.make_punt(2, Ware.SILK, 12)

        self.app._handle_pirate_boarding()
        _pump(self.root, 3)

        self.assertIsNone(boardable.ware_slots[2].occupant)
        self.assertEqual(self.state.pirate_boat.captain.occupant, captain.id)


class TestRevCaptainThenSecondSequencing(PirateBoardingDecisionTestCase):
    def test_second_is_skipped_and_promoted_once_the_captain_takes_the_only_slot(self):
        # Only one vacant ware slot exists across every boardable punt --
        # per the user, the captain decides first. By the time the second
        # pirate's turn comes, _show_boarding_dialog recomputes `boardable`
        # and finds nothing left, so the second is never offered a choice
        # at all -- it just gets promoted into the now-vacant captain slot
        # (existing board_setup.py behavior, unrelated to REV).
        captain = self.state.players[0]
        second = self.state.players[1]
        captain.is_bot = True
        captain.policy = "rev"
        second.is_bot = True
        second.policy = "rev"
        self.state.pirate_boat.captain.occupant = captain.id
        self.state.pirate_boat.second.occupant = second.id

        punt = self.make_punt(0, Ware.GINSENG, SEA_ROUTE_LENGTH)
        punt.ware_slots[1].occupant = "p2"
        punt.ware_slots[2].occupant = "p2"
        # ware_slots[0] is the only vacancy anywhere on the board (GINSENG
        # has exactly 3 ware slots).

        self.app._handle_pirate_boarding()
        _pump(self.root, 3)

        self.assertEqual(punt.ware_slots[0].occupant, captain.id)
        self.assertEqual(self.state.pirate_boat.captain.occupant, second.id)  # promoted
        self.assertIsNone(self.state.pirate_boat.second.occupant)

    def test_second_also_boards_when_room_remains_after_the_captain(self):
        # A single boardable punt with capacity for both pirates, and
        # nothing else on route worth staying for -- so both the captain
        # and, afterward, the second (with its own freshly-recomputed
        # vacancy) should each independently choose to board.
        captain = self.state.players[0]
        second = self.state.players[1]
        captain.is_bot = True
        captain.policy = "rev"
        second.is_bot = True
        second.policy = "rev"
        self.state.pirate_boat.captain.occupant = captain.id
        self.state.pirate_boat.second.occupant = second.id

        punt = self.make_punt(0, Ware.GINSENG, SEA_ROUTE_LENGTH)

        self.app._handle_pirate_boarding()
        _pump(self.root, 3)

        self.assertEqual(punt.ware_slots[0].occupant, captain.id)
        self.assertEqual(punt.ware_slots[1].occupant, second.id)
        self.assertIsNone(self.state.pirate_boat.captain.occupant)
        self.assertIsNone(self.state.pirate_boat.second.occupant)


if __name__ == "__main__":
    unittest.main()
