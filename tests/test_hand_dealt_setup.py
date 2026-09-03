"""Tests for the hand-dealt opening deal.

Normally the opening deal is shuffled (`GameState.new_default_game`). This
lets you state it instead, for reproducing a position from a real table --
along with how many of each ware are for sale, which the rules would
otherwise pin at "everything that wasn't dealt" (rules p.2, Preparation).

Two things the hand-dealt path must not change: every player still holds
exactly two shares, and setting the hands here doesn't *reveal* them -- the
computers go on inferring opponents' hands from public signals.
"""

import os
import sys
import tkinter as tk
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from manilla.engine import beliefs
from manilla.engine.models import (
    DEAL_POOL_PER_WARE,
    SHARES_PER_WARE,
    STARTING_SHARES_PER_PLAYER,
    GameState,
    Ware,
)
from manilla.ui.board_setup import BoardSetupApp

NAMES = ["P1", "P2", "P3", "P4"]
# A legal four-player deal: 3 nutmeg, 2 silk, 2 ginseng, 1 jade.
HANDS = [
    [Ware.NUTMEG, Ware.NUTMEG],
    [Ware.NUTMEG, Ware.SILK],
    [Ware.SILK, Ware.GINSENG],
    [Ware.GINSENG, Ware.JADE],
]


class TestHandDealtGame(unittest.TestCase):
    def test_each_player_gets_exactly_the_hand_given(self):
        state = GameState.new_hand_dealt_game(NAMES, HANDS)
        for player, hand in zip(state.players, HANDS):
            self.assertEqual([s.ware for s in player.shares], hand)

    def test_availability_defaults_to_everything_undealt(self):
        """With no stock given, the rules apply: undealt shares go on the
        table, so availability is just what wasn't dealt."""
        state = GameState.new_hand_dealt_game(NAMES, HANDS)
        self.assertEqual(state.shares_available(Ware.NUTMEG), SHARES_PER_WARE - 3)
        self.assertEqual(state.shares_available(Ware.SILK), SHARES_PER_WARE - 2)
        self.assertEqual(state.shares_available(Ware.JADE), SHARES_PER_WARE - 1)
        self.assertEqual(state.shares_withheld, {})

    def test_a_smaller_stock_is_recorded_as_withheld(self):
        state = GameState.new_hand_dealt_game(NAMES, HANDS, available={Ware.JADE: 1})
        self.assertEqual(state.shares_available(Ware.JADE), 1)
        # 5 exist, 1 dealt, 1 for sale -- the other 3 are out of the game.
        self.assertEqual(state.shares_withheld[Ware.JADE], 3)

    def test_buying_still_eats_into_the_reduced_stock(self):
        """Availability has to stay a live count, not a frozen number: a
        withheld ware still shrinks as the harbor master buys from it."""
        state = GameState.new_hand_dealt_game(NAMES, HANDS, available={Ware.JADE: 2})
        self.assertEqual(state.shares_available(Ware.JADE), 2)
        state.players[0].shares.append(state.players[3].shares.pop())  # a jade changes hands
        self.assertEqual(state.shares_available(Ware.JADE), 2)  # still 1 dealt overall
        state.players[0].shares.append(
            type(state.players[0].shares[0])(ware=Ware.JADE)
        )  # ...now one is bought
        self.assertEqual(state.shares_available(Ware.JADE), 1)

    def test_withheld_shares_survive_a_save_and_load(self):
        state = GameState.new_hand_dealt_game(NAMES, HANDS, available={Ware.SILK: 0})
        restored = GameState.from_dict(state.to_dict())
        self.assertEqual(restored.shares_withheld[Ware.SILK], 3)
        self.assertEqual(restored.shares_available(Ware.SILK), 0)

    def test_a_default_game_stores_no_withheld_shares(self):
        state = GameState.new_default_game(NAMES)
        self.assertEqual(state.shares_withheld, {})
        for ware in Ware:
            self.assertEqual(
                state.shares_available(ware), SHARES_PER_WARE - state.shares_owned(ware)
            )


class TestHandDealValidation(unittest.TestCase):
    def test_a_legal_deal_has_no_problems(self):
        self.assertEqual(GameState.hand_deal_problems(HANDS), [])

    def test_every_player_must_hold_two(self):
        hands = [list(h) for h in HANDS]
        hands[1] = [Ware.SILK]
        problems = GameState.hand_deal_problems(hands)
        self.assertTrue(any("Player 2" in p for p in problems), problems)

    def test_no_ware_can_be_dealt_more_than_the_pool_holds(self):
        """The deal comes off a pile of 3 of each ware, not all 5 -- so a
        hand-dealt 4 of one ware is a position that can't arise."""
        hands = [[Ware.JADE, Ware.JADE] for _ in NAMES]
        problems = GameState.hand_deal_problems(hands)
        self.assertTrue(any(str(DEAL_POOL_PER_WARE) in p for p in problems), problems)

    def test_stock_cannot_exceed_what_is_left_after_dealing(self):
        problems = GameState.hand_deal_problems(HANDS, {Ware.NUTMEG: 3})
        self.assertTrue(any("nutmeg" in p for p in problems), problems)
        self.assertEqual(GameState.hand_deal_problems(HANDS, {Ware.NUTMEG: 2}), [])

    def test_the_factory_refuses_what_the_check_rejects(self):
        with self.assertRaises(ValueError):
            GameState.new_hand_dealt_game(NAMES, [[Ware.JADE, Ware.JADE] for _ in NAMES])
        with self.assertRaises(ValueError):
            GameState.new_hand_dealt_game(NAMES, HANDS[:3])

    def test_over_withheld_shares_are_flagged_by_validate(self):
        state = GameState.new_hand_dealt_game(NAMES, HANDS)
        state.shares_withheld[Ware.NUTMEG] = 4  # 3 dealt + 4 withheld > 5
        self.assertTrue(any("withheld" in w for w in state.validate()))


class TestHandsStayHidden(unittest.TestCase):
    def test_a_computer_cannot_read_a_hand_dealt_opponents_shares(self):
        """Filling the hands in sets them, it doesn't reveal them: at the
        start of the game a viewer has nothing confirmed about anyone but
        themselves, hand-dealt or shuffled."""
        state = GameState.new_hand_dealt_game(NAMES, HANDS)
        view = beliefs.infer_beliefs(state, "p0")
        for other in ("p1", "p2", "p3"):
            self.assertEqual(view.known_total(other), 0)
        self.assertEqual(view.known_total("p0"), STARTING_SHARES_PER_PLAYER)


class HandDealDialogTestCase(unittest.TestCase):
    def setUp(self):
        self.root = tk.Tk()
        self.root.withdraw()  # never actually show a window during tests
        self.app = BoardSetupApp(self.root, GameState.new_default_game(NAMES))
        self.app.pack()
        self.app._show_new_game_setup_dialog()
        self.root.update_idletasks()
        self.dialog = next(
            w for w in self.root.winfo_children() if isinstance(w, tk.Toplevel)
        )

    def tearDown(self):
        self.root.destroy()

    def _of_class(self, cls, parent=None):
        found = []

        def walk(p):
            for child in p.winfo_children():
                if child.winfo_class() == cls:
                    found.append(child)
                walk(child)

        walk(parent or self.dialog)
        return found

    def _toggle_hand_deal(self):
        self._of_class("TCheckbutton")[0].invoke()
        self.root.update_idletasks()

    def _seat_rows(self):
        rows_frame = [w for w in self.dialog.pack_slaves() if w.winfo_class() == "TFrame"][0]
        return rows_frame.winfo_children()

    def _ware_combos(self, row):
        # Each row is: color, seat type, then one combobox per starting share.
        return [c for c in row.winfo_children() if c.winfo_class() == "TCombobox"][2:]

    def _error(self):
        return [w for w in self.dialog.pack_slaves() if w.winfo_class() == "Label"][0].cget(
            "text"
        )

    def _set_hand(self, row, wares):
        for combo, ware in zip(self._ware_combos(row), wares):
            combo.set(ware.value.title())
        self._ware_combos(row)[0].event_generate("<<ComboboxSelected>>")
        self.root.update_idletasks()


class TestDialogHandDealToggle(HandDealDialogTestCase):
    def test_share_pickers_are_hidden_until_hand_dealing_is_on(self):
        row = self._seat_rows()[0]
        packed = [c for c in row.pack_slaves() if c.winfo_class() == "TCombobox"]
        self.assertEqual(len(packed), 2)  # color and seat type only

        self._toggle_hand_deal()
        row = self._seat_rows()[0]
        packed = [c for c in row.pack_slaves() if c.winfo_class() == "TCombobox"]
        self.assertEqual(len(packed), 2 + STARTING_SHARES_PER_PLAYER)

    def test_the_stock_panel_follows_the_same_switch(self):
        self.assertNotIn("TLabelframe", [w.winfo_class() for w in self.dialog.pack_slaves()])
        self._toggle_hand_deal()
        self.assertIn("TLabelframe", [w.winfo_class() for w in self.dialog.pack_slaves()])
        self._toggle_hand_deal()
        self.assertNotIn("TLabelframe", [w.winfo_class() for w in self.dialog.pack_slaves()])

    def test_the_dialog_opens_on_a_deal_it_would_accept(self):
        self._toggle_hand_deal()
        self.assertEqual(self._error(), "")


class TestDialogStockTracksTheDeal(HandDealDialogTestCase):
    def _spinboxes(self):
        frame = [w for w in self.dialog.pack_slaves() if w.winfo_class() == "TLabelframe"][0]
        boxes = {}
        for ware, box in zip(Ware, self._of_class("TSpinbox", frame)):
            boxes[ware] = box
        return boxes

    def test_untouched_stock_follows_what_is_left_undealt(self):
        self._toggle_hand_deal()
        self._set_hand(self._seat_rows()[0], [Ware.JADE, Ware.JADE])
        self.assertEqual(int(self._spinboxes()[Ware.JADE].get()), SHARES_PER_WARE - 2)

    def test_a_hand_set_number_is_not_overwritten_by_the_next_pick(self):
        """Otherwise saying "only 1 jade is for sale" would be silently
        undone the moment you corrected somebody's hand."""
        self._toggle_hand_deal()
        boxes = self._spinboxes()
        boxes[Ware.JADE].set(1)  # typed straight into the box
        self.root.update_idletasks()
        self._set_hand(self._seat_rows()[1], [Ware.JADE, Ware.SILK])
        # Untouched it would have followed the deal down to 4; set by hand it
        # stays at 1, which is still a legal stock with one jade dealt out.
        self.assertEqual(int(self._spinboxes()[Ware.JADE].get()), 1)

    def test_an_impossible_deal_is_reported_while_you_type(self):
        self._toggle_hand_deal()
        for row in self._seat_rows():
            self._set_hand(row, [Ware.JADE, Ware.JADE])
        self.assertIn("jade", self._error())
        self.assertNotIn("negative", self._error())


class TestDialogBuildsTheGame(HandDealDialogTestCase):
    def _confirm(self):
        # Confirm normally rolls straight into the auction; stub that out so
        # the test only exercises the setup it is about.
        self.app._show_auction_dialog = lambda: None
        [b for b in self._of_class("TButton") if b.cget("text") == "Confirm"][0].invoke()
        self.root.update_idletasks()
        return self.app.state_obj

    def test_confirm_deals_the_hands_that_were_filled_in(self):
        self._toggle_hand_deal()
        for row, hand in zip(self._seat_rows(), HANDS):
            self._set_hand(row, hand)
        state = self._confirm()
        for player, hand in zip(state.players, HANDS):
            self.assertEqual(sorted(s.ware.value for s in player.shares),
                             sorted(w.value for w in hand))

    def test_confirm_applies_the_stock_that_was_filled_in(self):
        self._toggle_hand_deal()
        for row, hand in zip(self._seat_rows(), HANDS):
            self._set_hand(row, hand)
        boxes = [w for w in self.dialog.pack_slaves() if w.winfo_class() == "TLabelframe"][0]
        jade_box = self._of_class("TSpinbox", boxes)[3]
        jade_box.set(2)
        self.root.update_idletasks()
        state = self._confirm()
        self.assertEqual(state.shares_available(Ware.JADE), 2)
        self.assertEqual(state.shares_withheld[Ware.JADE], 2)  # 5 - 1 dealt - 2 for sale

    def test_leaving_the_option_off_still_deals_at_random(self):
        state = self._confirm()
        self.assertEqual(state.shares_withheld, {})
        for player in state.players:
            self.assertEqual(len(player.shares), STARTING_SHARES_PER_PLAYER)

    def test_an_invalid_deal_blocks_confirm_rather_than_building_a_bad_game(self):
        self._toggle_hand_deal()
        for row in self._seat_rows():
            self._set_hand(row, [Ware.JADE, Ware.JADE])
        before = self.app.state_obj
        self.app._show_auction_dialog = lambda: None
        [b for b in self._of_class("TButton") if b.cget("text") == "Confirm"][0].invoke()
        self.root.update_idletasks()
        self.assertIs(self.app.state_obj, before)  # nothing was replaced
        self.assertTrue(self.dialog.winfo_exists())  # and the dialog stayed open


if __name__ == "__main__":
    unittest.main()
