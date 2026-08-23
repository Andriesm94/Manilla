"""Regression test for a UI bug: the harbor-master auction's Bid/Pass
controls stayed enabled during a computer's own turn, so an impatient
click while a bot was still deciding (a REV bot's first decision of the
auction runs best_punt_setup, which visibly takes a moment) could land on
whichever player was current once the bot's own turn actually resolved --
letting a human accidentally "bid for" the computer instead of it bidding
for itself. The fix disables Bid/Pass/the bid spinbox, and shows a
"thinking..." status, for the whole time it's a bot's turn.
"""

import os
import sys
import time
import tkinter as tk
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from manilla.ui.board_setup import BoardSetupApp
from manilla.engine.models import GameState


def _pump(root, seconds):
    deadline = time.time() + seconds
    while time.time() < deadline:
        root.update()
        time.sleep(0.005)


class TestAuctionDialogDisablesControlsDuringABotsTurn(unittest.TestCase):
    def setUp(self):
        # Human is last in player-list order so bots go first when nobody
        # is harbor master yet -- exercising an actual bot's own turn.
        state = GameState.new_default_game(["Bot1", "Bot2", "Bot3", "Human"])
        for player in state.players[:3]:
            player.is_bot = True
            player.policy = "rev"
        state.players[3].is_bot = False
        state.game_setup_confirmed = True

        self.root = tk.Tk()
        self.app = BoardSetupApp(self.root, state)
        self.app.pack(fill=tk.BOTH, expand=True)
        self.root.update_idletasks()

    def tearDown(self):
        self.root.destroy()

    def _dialog(self):
        tops = [w for w in self.root.winfo_children() if isinstance(w, tk.Toplevel)]
        return tops[0] if tops else None

    def _controls(self, dialog):
        import tkinter.ttk as ttk

        for frame in dialog.winfo_children():
            if isinstance(frame, ttk.Frame):
                buttons = [c for c in frame.winfo_children() if isinstance(c, (ttk.Button, ttk.Spinbox))]
                if buttons:
                    return buttons
        return []

    def _status_text(self, dialog):
        labels = [c for c in dialog.winfo_children() if isinstance(c, tk.Label)]
        return labels[0].cget("text")

    def test_controls_are_disabled_and_status_says_thinking_on_a_bots_turn(self):
        # Bot1 goes first (no harbor master yet, player-list order) -- its
        # turn should immediately disable every control, before the (slow,
        # uncached) REV bid computation even starts.
        self.app._show_auction_dialog()
        dialog = self._dialog()
        self.assertIn("thinking", self._status_text(dialog))
        for control in self._controls(dialog):
            self.assertEqual(control.state(), ("disabled",))

    def test_controls_re_enable_once_its_the_humans_turn(self):
        self.app._show_auction_dialog()
        deadline = time.time() + 20
        while time.time() < deadline:
            dialog = self._dialog()
            if "Human" in self._status_text(dialog) and "thinking" not in self._status_text(dialog):
                break
            _pump(self.root, 0.05)
        else:
            self.fail("auction never reached the human's turn")

        for control in self._controls(dialog):
            self.assertEqual(control.state(), ())

    def test_random_policy_bots_are_also_disabled_during_their_turn(self):
        # The fix is policy-agnostic -- it keys off is_bot, not the "rev"
        # policy specifically.
        for player in self.app.state_obj.players[1:]:
            player.policy = "random"
        self.app._show_auction_dialog()
        dialog = self._dialog()
        self.assertIn("thinking", self._status_text(dialog))
        for control in self._controls(dialog):
            self.assertEqual(control.state(), ("disabled",))


if __name__ == "__main__":
    unittest.main()
