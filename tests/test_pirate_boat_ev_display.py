"""The user asked to see the plain EV (not REV) of taking the next pirate
slot, displayed on the board underneath the pirate boat -- so they could
tell why a bot always seems to grab the pirate slot first. Checks that
BoardSetupApp._draw_pirate_boat actually renders that number on the
canvas, and that it disappears once the boat is fully crewed.
"""

import os
import sys
import tkinter as tk
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from manilla.ui.board_setup import BoardSetupApp
from manilla.engine.models import GameState, Punt, PuntStatus, Ware


class TestPirateBoatEvDisplay(unittest.TestCase):
    def setUp(self):
        self.root = tk.Tk()
        self.app = BoardSetupApp(self.root)
        self.app.pack(fill=tk.BOTH, expand=True)
        self.state = self.app.state_obj
        self.state.game_setup_confirmed = True
        self.root.update_idletasks()

    def tearDown(self):
        self.root.destroy()

    def _canvas_texts(self):
        texts = []
        for item in self.app.canvas.find_all():
            try:
                texts.append(self.app.canvas.itemcget(item, "text"))
            except tk.TclError:
                pass
        return texts

    def test_shows_an_ev_line_when_a_pirate_slot_is_open(self):
        self.state.punts[0].ware = Ware.JADE
        self.state.punts[0].position = 6
        self.state.punts[0].status = PuntStatus.ON_ROUTE
        self.app.refresh()

        self.assertTrue(any(t.startswith("EV:") for t in self._canvas_texts()))

    def test_ev_line_breaks_out_the_captains_boarding_bonus(self):
        # Regression test for the user's bug report: the displayed EV used
        # to omit pirate_captain_boarding_bonus entirely, making the
        # number on the board look much smaller than what actually drove
        # a bot to take the captain's seat first.
        punt = self.state.punts[0]
        punt.ware = Ware.JADE
        punt.ware_slots = Punt.new(punt.id, Ware.JADE).ware_slots
        punt.status = PuntStatus.ON_ROUTE
        punt.position = 6
        self.state.movement_round_index = 0
        self.app.refresh()

        self.assertTrue(any("boarding +" in t for t in self._canvas_texts()))

    def test_no_ev_line_once_the_boat_is_fully_crewed(self):
        self.state.pirate_boat.captain.occupant = self.state.players[0].id
        self.state.pirate_boat.second.occupant = self.state.players[1].id
        self.app.refresh()

        self.assertFalse(any(t.startswith("EV:") for t in self._canvas_texts()))


if __name__ == "__main__":
    unittest.main()
