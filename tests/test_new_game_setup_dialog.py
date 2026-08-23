"""Regression test for a setup bug: each player row used to carry two
independent checkboxes, "Computer" and "REV policy (vs. random)". Ticking
"REV policy" without also ticking "Computer" left that seat's is_bot False
-- it looked configured for a REV computer but was actually still a human
seat, silently requiring a manual Bid/Pass click on every single turn for
every player, since none of the intended "computers" were ever recognized
as bots at all. The fix collapses both checkboxes into one seat selector
per row (Human / Computer (random) / Computer (REV)) so that combination
can't be produced any more.
"""

import os
import sys
import time
import tkinter as tk
import unittest
from tkinter import ttk

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from manilla.ui.board_setup import BoardSetupApp
from manilla.engine.models import GameState


def _pump(root, seconds):
    deadline = time.time() + seconds
    while time.time() < deadline:
        root.update()
        time.sleep(0.005)


class TestNewGameSetupDialog(unittest.TestCase):
    def setUp(self):
        state = GameState.new_default_game(["P1", "P2", "P3", "P4"])
        state.game_setup_confirmed = False

        self.root = tk.Tk()
        self.app = BoardSetupApp(self.root, state)
        self.app.pack(fill=tk.BOTH, expand=True)
        self.root.update_idletasks()
        _pump(self.root, 0.3)  # let the __init__-scheduled auto-popup fire

    def tearDown(self):
        self.root.destroy()

    def _dialog(self):
        return [w for w in self.root.winfo_children() if isinstance(w, tk.Toplevel)][0]

    def _rows_and_button_frame(self, dialog):
        frames = [c for c in dialog.winfo_children() if isinstance(c, ttk.Frame)]
        return frames[0].winfo_children(), frames[-1]

    def _seat_combo(self, row):
        # [0] is the color combobox, [1] is the seat selector.
        return [c for c in row.winfo_children() if isinstance(c, ttk.Combobox)][1]

    def _confirm(self, button_frame):
        confirm_btn = next(
            c for c in button_frame.winfo_children() if isinstance(c, ttk.Button) and c.cget("text") == "Confirm"
        )
        confirm_btn.invoke()
        _pump(self.root, 0.3)

    def test_three_computer_rev_seats_and_one_human_are_all_recognized_correctly(self):
        dialog = self._dialog()
        rows, button_frame = self._rows_and_button_frame(dialog)
        for row in rows[:3]:
            self._seat_combo(row).set("Computer (REV)")
        # rows[3] is left at the "Human" default.

        self._confirm(button_frame)

        players = self.app.state_obj.players
        for player in players[:3]:
            self.assertTrue(player.is_bot)
            self.assertEqual(player.policy, "rev")
        self.assertFalse(players[3].is_bot)

    def test_a_computer_random_seat_is_a_bot_on_the_random_policy(self):
        dialog = self._dialog()
        rows, button_frame = self._rows_and_button_frame(dialog)
        self._seat_combo(rows[0]).set("Computer (random)")

        self._confirm(button_frame)

        player = self.app.state_obj.players[0]
        self.assertTrue(player.is_bot)
        self.assertEqual(player.policy, "random")

    def test_seat_options_offer_no_way_to_produce_rev_without_a_computer(self):
        dialog = self._dialog()
        rows, _ = self._rows_and_button_frame(dialog)
        options = set(self._seat_combo(rows[0]).cget("values"))
        self.assertEqual(options, {"Human", "Computer (random)", "Computer (REV)"})

    def test_simulate_all_rev_computers_sets_every_seat_to_bot_rev(self):
        dialog = self._dialog()
        _, button_frame = self._rows_and_button_frame(dialog)
        simulate_btn = next(
            c
            for c in button_frame.winfo_children()
            if isinstance(c, ttk.Button) and c.cget("text") == "Simulate (all REV computers)"
        )
        simulate_btn.invoke()
        _pump(self.root, 0.3)

        for player in self.app.state_obj.players:
            self.assertTrue(player.is_bot)
            self.assertEqual(player.policy, "rev")


if __name__ == "__main__":
    unittest.main()
