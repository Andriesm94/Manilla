"""Regression tests for the "Voyage profit summary" shown in the Profit
Distribution notification.

It compares each player's cash now against their cash at the very start
of this voyage's accomplice-placement phase (`_cash_at_round_start`,
snapshotted in `_apply_load_and_place`) -- covering every cash movement
all voyage (accomplice placement costs, the insurance bonus, every
round-end payout) rather than an itemized list that has to remember to
include each source. `_distribute_profits` (see `_show_profit_
distribution_gate`) is where this summary gets built, only once
"Continue to Profit Distribution" is explicitly clicked.
"""

import os
import sys
import tkinter as tk
import unittest
from unittest import mock

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from manilla.ui.board_setup import BoardSetupApp
from manilla.engine.models import Punt, PuntStatus, Ware


class ProfitSummaryTestCase(unittest.TestCase):
    def setUp(self):
        self.root = tk.Tk()
        self.root.withdraw()  # never actually show a window during tests
        self.app = BoardSetupApp(self.root)
        self.app.pack(fill=tk.BOTH, expand=True)
        self.state = self.app.state_obj
        self.state.game_setup_confirmed = True
        self.root.update_idletasks()

    def tearDown(self):
        self.root.destroy()

    def make_punt(self, idx, ware, position):
        punt = self.state.punts[idx]
        punt.ware = ware
        punt.ware_slots = Punt.new(punt.id, ware).ware_slots
        punt.status = PuntStatus.ON_ROUTE
        punt.position = position
        return punt


class TestCashSnapshotAtRoundStart(ProfitSummaryTestCase):
    def test_load_and_place_snapshots_every_players_cash(self):
        for i, player in enumerate(self.state.players):
            player.cash = 30 + i
        harbor_master = self.state.players[0]
        loaded = [Ware.NUTMEG, Ware.SILK, Ware.GINSENG]
        positions = {Ware.NUTMEG: 3, Ware.SILK: 3, Ware.GINSENG: 3}

        self.app._apply_load_and_place(harbor_master, loaded, positions)

        self.assertEqual(self.app._cash_at_round_start, {p.id: p.cash for p in self.state.players})


class TestVoyageProfitSummarySignFormatting(ProfitSummaryTestCase):
    def test_positive_negative_and_zero_deltas_are_formatted_correctly(self):
        players = self.state.players
        self.app._cash_at_round_start = {p.id: p.cash for p in players}
        players[0].cash += 10  # positive delta
        players[1].cash -= 7  # negative delta
        # players[2] (and beyond) stay untouched -- zero delta

        with mock.patch("manilla.ui.board_setup.messagebox.showinfo") as showinfo:
            self.app._distribute_profits([], then=None)
            message = showinfo.call_args[0][1]

        self.assertIn(f"{players[0].color}: +10", message)
        self.assertIn(f"{players[1].color}: -7", message)
        self.assertIn(f"{players[2].color}: +0", message)


class TestVoyageProfitSummaryReflectsNetChange(ProfitSummaryTestCase):
    def test_summary_captures_placement_costs_ware_profit_and_port_reward_together(self):
        """The old itemized "what got paid" list never included accomplice
        placement costs at all -- this diffs total cash instead, so it
        can't miss anything, whatever the source."""
        players = self.state.players
        harbor_master = players[0]
        loaded = [Ware.NUTMEG, Ware.SILK, Ware.GINSENG]
        positions = {Ware.NUTMEG: 3, Ware.SILK: 3, Ware.GINSENG: 3}
        with mock.patch("manilla.ui.board_setup.messagebox.showinfo"):
            self.app._apply_load_and_place(harbor_master, loaded, positions)  # snapshots cash here

        # A placement cost between the snapshot and distribution -- the
        # kind of cash movement an itemized payout list never captured.
        players[0].cash -= 4

        punt = self.state.punts[0]  # loaded with NUTMEG (payout 24)
        punt.status = PuntStatus.IN_PORT
        punt.dock_slot = "A"
        punt.ware_slots[0].occupant = players[1].id  # sole ware accomplice -> full 24
        self.state.port.slots["A"].occupant = players[2].id  # port slot A pays 6
        self.state.movement_round_index = 3

        with mock.patch("manilla.ui.board_setup.messagebox.showinfo") as showinfo:
            self.app._distribute_profits([], then=None)
            message = showinfo.call_args[0][1]

        self.assertIn(f"{players[0].color}: -4", message)
        self.assertIn(f"{players[1].color}: +24", message)
        self.assertIn(f"{players[2].color}: +6", message)


if __name__ == "__main__":
    unittest.main()
