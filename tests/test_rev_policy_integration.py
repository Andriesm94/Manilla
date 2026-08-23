"""End-to-end check that the "rev" policy can actually drive a live game
through `BoardSetupApp`, not just pass its own engine-layer unit tests.

Uses the project's established headless-Tkinter pattern: build a real
(never-shown) BoardSetupApp, kick off the auction, and pump the Tk event
loop so every self.after()-scheduled bot decision fires, exactly as it
would in a real, running window.
"""

import os
import sys
import time
import tkinter as tk
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from manilla.ui.board_setup import BoardSetupApp
from manilla.engine.models import GameState, Phase


class TestRevPolicyDrivesALiveGame(unittest.TestCase):
    def setUp(self):
        # game_setup_confirmed must already be True *before* the app is
        # constructed: BoardSetupApp.__init__ checks it synchronously and,
        # if it's False at that moment, schedules an after(50, ...) call
        # to pop up the "New Game Setup" dialog -- setting it True on the
        # resulting object afterward doesn't cancel that already-scheduled
        # callback, which then fires mid-test and collides with whatever
        # dialog the auction/placement flow has open by then.
        state = GameState.new_default_game(["A", "B", "C", "D"])
        for player in state.players:
            player.is_bot = True
            player.policy = "rev"
        state.game_setup_confirmed = True

        self.root = tk.Tk()
        self.app = BoardSetupApp(self.root, state)
        self.app.pack(fill=tk.BOTH, expand=True)
        self.root.update_idletasks()

    def tearDown(self):
        self.root.destroy()

    def _pump_until(self, condition, timeout_seconds=240):
        # root.update() alone, called in a tight loop, does not reliably
        # let self.after()-scheduled callbacks (BOT_DELAY_MS pacing
        # between bot decisions) become due -- Tkinter only fires timers
        # that have *already* elapsed real wall-clock time, and a fast
        # Python loop can spin through many update() calls without ever
        # accumulating enough real time for a pending timer to fire. A
        # small sleep between updates is what actually lets the clock
        # advance.
        deadline = time.time() + timeout_seconds
        while time.time() < deadline:
            if condition():
                return True
            self.root.update()
            time.sleep(0.005)
        return condition()

    def test_all_rev_bots_complete_a_full_voyage_without_stalling(self):
        self.app._show_auction_dialog()

        reached = self._pump_until(lambda: self.app.state_obj.phase == Phase.PROFIT_DISTRIBUTION)
        self.assertTrue(reached, "the voyage never reached PROFIT_DISTRIBUTION")

        # Money never goes negative -- the game's own invariant, backed by
        # the credit/blind-passenger mechanism; a REV bot stalling mid-
        # payment or double-charging would show up here.
        for player in self.app.state_obj.players:
            self.assertGreaterEqual(player.cash, 0)

        # A harbor master was actually chosen and the voyage was loaded --
        # confirms the auction and load/place decisions both resolved,
        # not just that *some* phase transition eventually happened.
        self.assertTrue(any(p.is_harbor_master for p in self.app.state_obj.players))
        self.assertIsNotNone(self.app.state_obj.unloaded_ware)
        self.assertEqual(sum(1 for p in self.app.state_obj.punts if p.ware is not None), 3)


if __name__ == "__main__":
    unittest.main()
