"""Regression tests for a bug: pirate plunder payouts and the insurance
occupant's upfront payment were real (players did receive the cash) but
never appeared in the round-end "Voyage profit summary" notification --
only port/shipyard rewards, ware-cargo splits, and insurance repair costs
were included. Fixed by having `_pay_pirate_plunder` report what it paid
out and folding it, plus a new `_voyage_bonus_payouts` tracker for the
insurance payment (paid at placement time, well before round-end), into
the same summary.

All of this -- pirate plunder, ware/port/shipyard profits, insurance
repairs, and the ware-value rise -- is computed together by
`_distribute_profits`, only once "Continue to Profit Distribution" is
explicitly clicked (see the "Voyage complete" gate in
`_show_profit_distribution_gate`), not automatically the instant the
third roll resolves.
"""

import os
import sys
import tkinter as tk
import unittest
from unittest import mock

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from manilla.ui.board_setup import BoardSetupApp
from manilla.engine.models import GameState, Punt, PuntStatus, SEA_ROUTE_LENGTH, Ware


class ProfitSummaryTestCase(unittest.TestCase):
    def setUp(self):
        self.root = tk.Tk()
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

    def find_toplevels(self, title_substr):
        return [
            t for t in self.root.winfo_children() if isinstance(t, tk.Toplevel) and title_substr in t.title()
        ]

    def all_widgets(self, widget):
        acc = [widget]
        for child in widget.winfo_children():
            acc.extend(self.all_widgets(child))
        return acc

    def button_in(self, widget, text_contains):
        return next(
            w for w in self.all_widgets(widget) if w.winfo_class() == "TButton" and text_contains in w.cget("text")
        )

    def click_continue_to_profit_distribution(self):
        gate = self.find_toplevels("Voyage complete")
        self.assertEqual(len(gate), 1, "expected the profit-distribution gate to be showing")
        self.button_in(gate[0], "Continue to Profit Distribution").invoke()


class TestPayPiratePlunderReportsItsPayouts(ProfitSummaryTestCase):
    def test_returns_the_color_and_share_paid_to_each_pirate(self):
        punt = self.make_punt(0, Ware.NUTMEG, SEA_ROUTE_LENGTH)
        players = self.state.players
        self.state.pirate_boat.captain.occupant = players[0].id
        self.state.pirate_boat.second.occupant = players[1].id

        paid = self.app._pay_pirate_plunder([punt])

        expected_share = 24 // 2  # NUTMEG plunder payout split two ways
        self.assertEqual(set(paid), {(players[0].color, expected_share), (players[1].color, expected_share)})

    def test_empty_when_no_pirates_are_aboard(self):
        punt = self.make_punt(0, Ware.NUTMEG, SEA_ROUTE_LENGTH)
        paid = self.app._pay_pirate_plunder([punt])
        self.assertEqual(paid, [])


class TestVoyageBonusPayoutsTracksInsurance(ProfitSummaryTestCase):
    def test_placing_insurance_records_a_positive_bonus_payout(self):
        player = self.state.players[0]
        self.state.current_turn_player_id = player.id
        before = player.cash

        self.app._place_or_remove_insurance()

        payment = self.state.insurance.payment
        self.assertEqual(player.cash, before + payment)
        self.assertIn((player.color, payment), self.app._voyage_bonus_payouts)

    def test_removing_insurance_records_a_matching_negative_entry(self):
        player = self.state.players[0]
        self.state.current_turn_player_id = player.id
        self.app._place_or_remove_insurance()  # place
        self.app._place_or_remove_insurance()  # remove (click again)

        payment = self.state.insurance.payment
        total = sum(amount for color, amount in self.app._voyage_bonus_payouts if color == player.color)
        self.assertEqual(total, 0)


class TestRoundEndSummaryIncludesEverything(ProfitSummaryTestCase):
    def test_plunder_and_insurance_payouts_appear_in_the_voyage_profit_summary(self):
        players = self.state.players
        pirate_player = players[0]
        insurance_player = players[1]

        # An insurance payment from earlier in the voyage (placement-time,
        # well before this round's payouts get computed).
        self.state.insurance.occupant = insurance_player.id
        self.app._voyage_bonus_payouts = [(insurance_player.color, self.state.insurance.payment)]

        # A punt one roll away from space 13, with the third movement
        # round about to happen, and a pirate captain aboard to plunder it.
        self.make_punt(0, Ware.NUTMEG, SEA_ROUTE_LENGTH - 1)
        for punt in self.state.punts[1:]:
            punt.ware = None
        self.state.pirate_boat.captain.occupant = pirate_player.id
        self.state.movement_round_index = 2
        pirate_player.is_bot = True  # skip the human port-vs-shipyard dialog

        with mock.patch("manilla.ui.board_setup.random.randint", return_value=1), mock.patch(
            "manilla.ui.board_setup.random.random", return_value=0.0
        ), mock.patch("manilla.ui.board_setup.messagebox.showinfo") as showinfo:
            self.app._roll_dice_and_move()
            showinfo.assert_called_once()  # just the roll summary so far -- nobody's been paid yet
            self.click_continue_to_profit_distribution()

        message = showinfo.call_args_list[-1].args[1]
        self.assertIn("Voyage profit summary", message)
        self.assertIn(pirate_player.color, message)
        self.assertIn(insurance_player.color, message)

    def test_voyage_bonus_payouts_are_cleared_after_being_folded_in(self):
        players = self.state.players
        self.app._voyage_bonus_payouts = [(players[0].color, 10)]

        self.make_punt(0, Ware.NUTMEG, SEA_ROUTE_LENGTH)
        for punt in self.state.punts[1:]:
            punt.ware = None
        self.state.movement_round_index = 2

        with mock.patch("manilla.ui.board_setup.random.randint", return_value=1), mock.patch(
            "manilla.ui.board_setup.messagebox.showinfo"
        ):
            self.app._roll_dice_and_move()
            self.assertEqual(
                self.app._voyage_bonus_payouts, [(players[0].color, 10)], "still pending -- gate not clicked yet"
            )
            self.click_continue_to_profit_distribution()

        self.assertEqual(self.app._voyage_bonus_payouts, [])


if __name__ == "__main__":
    unittest.main()
