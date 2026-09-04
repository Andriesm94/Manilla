"""Tests for the hand-dealt opening deal.

Normally the opening deal is shuffled (`GameState.new_default_game`). This
lets you state it instead, for reproducing a position from a real table.

A computer seat says what it was dealt. A human seat doesn't -- they would
rather not tell the game state their hand -- so their two shares are held
*unrecorded*: counted, but with no ware attached. What makes such a position
add up is the stock: whatever isn't dealt out and isn't for sale has to be
exactly the unrecorded hands.

The rules keep those identities hidden right up to the end (p.8: even an
encumbered share is set aside face-down), and then require them, since at
game end each player "counts his cash and adds to this the value of his
shares". So the one place they surface is scoring.
"""

import os
import sys
import tkinter as tk
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from manilla.engine import beliefs
from manilla.engine.models import (
    DEAL_POOL_PER_WARE,
    SHARE_LOAN_AMOUNT,
    SHARES_PER_WARE,
    STARTING_SHARES_PER_PLAYER,
    GameState,
    Share,
    Ware,
)
from manilla.ui.board_setup import BoardSetupApp

NAMES = ["P1", "P2", "P3", "P4"]

# Every hand recorded: 3 nutmeg, 2 silk, 2 ginseng, 1 jade dealt out.
HANDS = [
    [Ware.NUTMEG, Ware.NUTMEG],
    [Ware.NUTMEG, Ware.SILK],
    [Ware.SILK, Ware.GINSENG],
    [Ware.GINSENG, Ware.JADE],
]

# The last seat is a human who didn't say. Recorded: 3 nutmeg, 2 silk,
# 1 ginseng -- so the stock has to be 2 short of the 2/3/4/5 left undealt,
# and here that shortfall sits on ginseng and jade.
MIXED_HANDS = [
    [Ware.NUTMEG, Ware.NUTMEG],
    [Ware.NUTMEG, Ware.SILK],
    [Ware.SILK, Ware.GINSENG],
    None,
]
MIXED_STOCK = {Ware.NUTMEG: 2, Ware.SILK: 3, Ware.GINSENG: 3, Ware.JADE: 4}


class TestRecordedHands(unittest.TestCase):
    def test_each_player_gets_exactly_the_hand_given(self):
        state = GameState.new_hand_dealt_game(NAMES, HANDS)
        for player, hand in zip(state.players, HANDS):
            self.assertEqual([s.ware for s in player.shares], hand)

    def test_availability_follows_the_deal_when_no_stock_is_given(self):
        """With every hand recorded the rules settle it: undealt shares go
        on the table, so availability is just what wasn't dealt."""
        state = GameState.new_hand_dealt_game(NAMES, HANDS)
        self.assertEqual(state.shares_available(Ware.NUTMEG), SHARES_PER_WARE - 3)
        self.assertEqual(state.shares_available(Ware.SILK), SHARES_PER_WARE - 2)
        self.assertEqual(state.shares_available(Ware.JADE), SHARES_PER_WARE - 1)
        self.assertEqual(state.unrecorded_holdings, {})

    def test_a_default_game_holds_nothing_unrecorded(self):
        state = GameState.new_default_game(NAMES)
        self.assertEqual(state.unrecorded_holdings, {})
        self.assertEqual(state.unrecorded_share_count(), 0)
        for ware in Ware:
            self.assertEqual(
                state.shares_available(ware), SHARES_PER_WARE - state.shares_owned(ware)
            )


class TestUnrecordedHands(unittest.TestCase):
    def test_a_human_seat_holds_two_shares_with_no_ware(self):
        state = GameState.new_hand_dealt_game(NAMES, MIXED_HANDS, MIXED_STOCK)
        human = state.players[3]
        self.assertEqual(len(human.shares), STARTING_SHARES_PER_PLAYER)
        self.assertTrue(all(not s.is_recorded for s in human.shares))
        self.assertEqual(state.unrecorded_share_count(), STARTING_SHARES_PER_PLAYER)

    def test_the_stock_comes_out_exactly_as_entered(self):
        state = GameState.new_hand_dealt_game(NAMES, MIXED_HANDS, MIXED_STOCK)
        for ware, expected in MIXED_STOCK.items():
            self.assertEqual(state.shares_available(ware), expected)

    def test_the_shortfall_is_booked_against_the_wares_it_came_from(self):
        state = GameState.new_hand_dealt_game(NAMES, MIXED_HANDS, MIXED_STOCK)
        # 4 ginseng and 5 jade were undealt; 3 and 4 are for sale.
        self.assertEqual(state.unrecorded_holdings, {Ware.GINSENG: 1, Ware.JADE: 1})

    def test_buying_still_eats_into_the_stock(self):
        """Availability has to stay a live count, not a frozen number."""
        state = GameState.new_hand_dealt_game(NAMES, MIXED_HANDS, MIXED_STOCK)
        state.players[0].shares.append(Share(ware=Ware.JADE))
        self.assertEqual(state.shares_available(Ware.JADE), MIXED_STOCK[Ware.JADE] - 1)

    def test_it_all_survives_a_save_and_load(self):
        state = GameState.new_hand_dealt_game(NAMES, MIXED_HANDS, MIXED_STOCK)
        restored = GameState.from_dict(state.to_dict())
        self.assertEqual(restored.unrecorded_holdings, state.unrecorded_holdings)
        self.assertEqual(restored.unrecorded_share_count(), STARTING_SHARES_PER_PLAYER)
        for ware, expected in MIXED_STOCK.items():
            self.assertEqual(restored.shares_available(ware), expected)


class TestNamingAShareLater(unittest.TestCase):
    def setUp(self):
        self.state = GameState.new_hand_dealt_game(NAMES, MIXED_HANDS, MIXED_STOCK)
        self.human = self.state.players[3]

    def test_naming_a_share_does_not_change_what_is_for_sale(self):
        """Revealing a share you already held doesn't put anything back on
        the table -- it just moves it out of the unrecorded column."""
        before = {w: self.state.shares_available(w) for w in Ware}
        self.state.record_share_identity(self.human.shares[0], Ware.GINSENG)
        after = {w: self.state.shares_available(w) for w in Ware}
        self.assertEqual(before, after)
        self.assertEqual(self.state.shares_owned(Ware.GINSENG), 2)

    def test_a_named_share_leaves_the_unrecorded_count(self):
        self.state.record_share_identity(self.human.shares[0], Ware.JADE)
        self.assertEqual(self.state.unrecorded_share_count(), 1)
        self.assertEqual(self.state.unrecorded_holdings, {Ware.GINSENG: 1})

    def test_a_ware_with_nothing_outstanding_cannot_be_named(self):
        """Only a hand the stock said was possible can be revealed."""
        with self.assertRaises(ValueError):
            self.state.record_share_identity(self.human.shares[0], Ware.NUTMEG)

    def test_an_already_named_share_cannot_be_named_again(self):
        self.state.record_share_identity(self.human.shares[0], Ware.JADE)
        with self.assertRaises(ValueError):
            self.state.record_share_identity(self.human.shares[0], Ware.GINSENG)

    def test_validate_flags_a_tally_that_drifted_from_the_hands(self):
        self.state.unrecorded_holdings[Ware.JADE] = 3  # more than anyone holds
        self.assertTrue(any("unnamed" in w for w in self.state.validate()))


class TestHandDealValidation(unittest.TestCase):
    def test_a_legal_all_recorded_deal_has_no_problems(self):
        self.assertEqual(GameState.hand_deal_problems(HANDS), [])

    def test_a_legal_mixed_deal_has_no_problems(self):
        self.assertEqual(GameState.hand_deal_problems(MIXED_HANDS, MIXED_STOCK), [])

    def test_every_recorded_player_must_hold_two(self):
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

    def test_the_pool_bound_counts_hidden_shares_too(self):
        """3 nutmeg are already dealt, so a stock of 0 nutmeg would put a
        4th in a hidden hand -- more than the deal pile holds."""
        problems = GameState.hand_deal_problems(
            MIXED_HANDS, {**MIXED_STOCK, Ware.NUTMEG: 0, Ware.GINSENG: 4, Ware.JADE: 5}
        )
        self.assertTrue(any(str(DEAL_POOL_PER_WARE) in p for p in problems), problems)

    def test_stock_cannot_exceed_what_is_left_after_dealing(self):
        problems = GameState.hand_deal_problems(HANDS, {Ware.NUTMEG: 3})
        self.assertTrue(any("nutmeg" in p for p in problems), problems)

    def test_an_unrecorded_hand_needs_the_stock_filled_in(self):
        problems = GameState.hand_deal_problems(MIXED_HANDS)
        self.assertTrue(any("for sale" in p for p in problems), problems)

    def test_shares_that_are_neither_dealt_nor_for_sale_need_a_hand_to_be_in(self):
        # Every hand recorded, but one jade is neither dealt nor for sale.
        problems = GameState.hand_deal_problems(
            HANDS, {Ware.NUTMEG: 2, Ware.SILK: 3, Ware.GINSENG: 3, Ware.JADE: 3}
        )
        self.assertTrue(any("nowhere for them to be" in p for p in problems), problems)

    def test_a_stock_that_leaves_too_little_hidden_says_how_much_to_take_off(self):
        problems = GameState.hand_deal_problems(
            MIXED_HANDS, {Ware.NUTMEG: 2, Ware.SILK: 3, Ware.GINSENG: 4, Ware.JADE: 5}
        )
        self.assertTrue(any("take 2 more off" in p for p in problems), problems)

    def test_a_stock_that_hides_too_much_says_to_put_some_back(self):
        problems = GameState.hand_deal_problems(
            MIXED_HANDS, {Ware.NUTMEG: 2, Ware.SILK: 3, Ware.GINSENG: 2, Ware.JADE: 3}
        )
        self.assertTrue(any("put 2 back" in p for p in problems), problems)

    def test_the_factory_refuses_what_the_check_rejects(self):
        with self.assertRaises(ValueError):
            GameState.new_hand_dealt_game(NAMES, [[Ware.JADE, Ware.JADE] for _ in NAMES])
        with self.assertRaises(ValueError):
            GameState.new_hand_dealt_game(NAMES, MIXED_HANDS)  # no stock given
        with self.assertRaises(ValueError):
            GameState.new_hand_dealt_game(NAMES, HANDS[:3])


class TestComputersLearnNothing(unittest.TestCase):
    def test_a_recorded_hand_is_set_but_not_revealed(self):
        """Typing the computers' hands in doesn't tell them about each
        other: a viewer starts with nothing confirmed but their own."""
        state = GameState.new_hand_dealt_game(NAMES, HANDS)
        view = beliefs.infer_beliefs(state, "p0")
        for other in ("p1", "p2", "p3"):
            self.assertEqual(view.known_total(other), 0)
        self.assertEqual(view.known_total("p0"), STARTING_SHARES_PER_PLAYER)

    def test_an_unrecorded_hand_is_not_confirmed_even_to_its_own_holder(self):
        state = GameState.new_hand_dealt_game(NAMES, MIXED_HANDS, MIXED_STOCK)
        view = beliefs.infer_beliefs(state, "p3")
        self.assertEqual(view.known_total("p3"), 0)

    def test_hidden_shares_stay_in_the_secret_pool(self):
        """The pool has to cover every unknown slot. Leaving the unrecorded
        shares out of it would shrink the pool below the slots it prices,
        and every unidentified share would come out too cheap."""
        state = GameState.new_hand_dealt_game(NAMES, MIXED_HANDS, MIXED_STOCK)
        for viewer in ("p0", "p3"):
            view = beliefs.infer_beliefs(state, viewer)
            pool = beliefs.secret_pool(state, view)
            self.assertEqual(
                sum(pool.values()), beliefs.total_secret_slots(state, view), viewer
            )

    def test_the_pool_matches_a_shuffled_game_the_same_way(self):
        state = GameState.new_default_game(NAMES)
        view = beliefs.infer_beliefs(state, "p0")
        pool = beliefs.secret_pool(state, view)
        self.assertEqual(sum(pool.values()), beliefs.total_secret_slots(state, view))


class DialogTestCase(unittest.TestCase):
    """Drives the real setup dialog rather than a stand-in for it."""

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

    def _rows(self):
        rows_frame = [w for w in self.dialog.pack_slaves() if w.winfo_class() == "TFrame"][0]
        return rows_frame.winfo_children()

    def _combos(self, row):
        return [c for c in row.winfo_children() if c.winfo_class() == "TCombobox"]

    def _set_seat(self, row, kind):
        combo = self._combos(row)[1]
        combo.set(kind)
        combo.event_generate("<<ComboboxSelected>>")
        self.root.update_idletasks()

    def _set_hand(self, row, wares):
        for combo, ware in zip(self._combos(row)[2:], wares):
            combo.set(ware.value.title())
        self._combos(row)[2].event_generate("<<ComboboxSelected>>")
        self.root.update_idletasks()

    def _stock_boxes(self):
        frame = [w for w in self.dialog.pack_slaves() if w.winfo_class() == "TLabelframe"][0]
        return dict(zip(Ware, self._of_class("TSpinbox", frame)))

    def _set_stock(self, stock):
        boxes = self._stock_boxes()
        for ware, value in stock.items():
            boxes[ware].set(value)
        self.root.update_idletasks()

    def _error(self):
        return [w for w in self.dialog.pack_slaves() if w.winfo_class() == "Label"][0].cget(
            "text"
        )

    def _visible_texts(self, row):
        return [
            c.cget("text") for c in row.pack_slaves() if c.winfo_class() == "TLabel"
        ]

    def _lay_out_mixed_table(self):
        """Three REV computers with recorded hands, one human without."""
        self._toggle_hand_deal()
        for row, hand in zip(self._rows(), MIXED_HANDS):
            if hand is None:
                continue
            self._set_seat(row, "Computer (REV)")
            self._set_hand(row, hand)
        self._set_stock(MIXED_STOCK)


class TestDialogHidesHumanHands(DialogTestCase):
    def test_a_human_seat_gets_no_share_pickers(self):
        self._toggle_hand_deal()
        row = self._rows()[0]  # seats default to Human
        packed = [c for c in row.pack_slaves() if c.winfo_class() == "TCombobox"]
        self.assertEqual(len(packed), 2)  # color and seat type only
        self.assertIn("shares not recorded", self._visible_texts(row))

    def test_a_computer_seat_gets_them(self):
        self._toggle_hand_deal()
        row = self._rows()[0]
        self._set_seat(row, "Computer (REV)")
        packed = [c for c in row.pack_slaves() if c.winfo_class() == "TCombobox"]
        self.assertEqual(len(packed), 2 + STARTING_SHARES_PER_PLAYER)
        self.assertNotIn("shares not recorded", self._visible_texts(row))

    def test_switching_a_seat_back_to_human_takes_them_away_again(self):
        self._toggle_hand_deal()
        row = self._rows()[0]
        self._set_seat(row, "Computer (REV)")
        self._set_seat(row, "Human")
        packed = [c for c in row.pack_slaves() if c.winfo_class() == "TCombobox"]
        self.assertEqual(len(packed), 2)
        self.assertIn("shares not recorded", self._visible_texts(row))

    def test_nothing_shows_at_all_with_hand_dealing_off(self):
        row = self._rows()[0]
        self.assertEqual(len(self._combos(row)) - 2, STARTING_SHARES_PER_PLAYER)
        packed = [c for c in row.pack_slaves() if c.winfo_class() == "TCombobox"]
        self.assertEqual(len(packed), 2)
        self.assertNotIn("shares not recorded", self._visible_texts(row))


class TestDialogStockGuidance(DialogTestCase):
    def test_it_says_how_many_shares_are_still_unaccounted_for(self):
        self._toggle_hand_deal()
        for row in self._rows()[1:]:
            self._set_seat(row, "Computer (REV)")
        # One human left, and the stock still defaults to everything undealt.
        self.assertIn("take 2 more off", self._error())

    def test_a_stock_that_adds_up_clears_the_error(self):
        self._lay_out_mixed_table()
        self.assertEqual(self._error(), "")

    def test_a_hand_set_number_is_not_overwritten_by_the_next_pick(self):
        """Otherwise correcting somebody's hand would silently undo the
        stock you just entered."""
        self._lay_out_mixed_table()
        self._set_hand(self._rows()[0], [Ware.NUTMEG, Ware.NUTMEG])
        self.assertEqual(int(self._stock_boxes()[Ware.JADE].get()), MIXED_STOCK[Ware.JADE])


class TestDialogBuildsTheGame(DialogTestCase):
    def _confirm(self):
        # Confirm normally rolls straight into the auction; stub that out so
        # the test only exercises the setup it is about.
        self.app._show_auction_dialog = lambda: None
        [b for b in self._of_class("TButton") if b.cget("text") == "Confirm"][0].invoke()
        self.root.update_idletasks()
        return self.app.state_obj

    def test_the_computers_get_their_hands_and_the_human_stays_hidden(self):
        self._lay_out_mixed_table()
        state = self._confirm()
        for player, hand in zip(state.players, MIXED_HANDS):
            if hand is None:
                self.assertTrue(all(not s.is_recorded for s in player.shares))
            else:
                self.assertEqual(
                    sorted(s.ware.value for s in player.shares),
                    sorted(w.value for w in hand),
                )
        self.assertEqual(state.unrecorded_share_count(), STARTING_SHARES_PER_PLAYER)

    def test_the_stock_comes_out_as_entered(self):
        self._lay_out_mixed_table()
        state = self._confirm()
        for ware, expected in MIXED_STOCK.items():
            self.assertEqual(state.shares_available(ware), expected)

    def test_leaving_the_option_off_still_deals_at_random(self):
        state = self._confirm()
        self.assertEqual(state.unrecorded_holdings, {})
        for player in state.players:
            self.assertEqual(len(player.shares), STARTING_SHARES_PER_PLAYER)
            self.assertTrue(all(s.is_recorded for s in player.shares))

    def test_a_stock_that_does_not_add_up_blocks_confirm(self):
        self._toggle_hand_deal()
        for row in self._rows()[1:]:
            self._set_seat(row, "Computer (REV)")
        before = self.app.state_obj
        self.app._show_auction_dialog = lambda: None
        [b for b in self._of_class("TButton") if b.cget("text") == "Confirm"][0].invoke()
        self.root.update_idletasks()
        self.assertIs(self.app.state_obj, before)  # nothing was replaced
        self.assertTrue(self.dialog.winfo_exists())  # and the dialog stayed open

    def test_simulate_all_computers_still_works_from_a_human_table(self):
        """It flips every seat to a computer, which makes hands that were
        hidden a moment ago recorded -- the stock has to follow."""
        self._toggle_hand_deal()
        self.app._show_auction_dialog = lambda: None
        [b for b in self._of_class("TButton") if "all REV" in b.cget("text")][0].invoke()
        self.root.update_idletasks()
        state = self.app.state_obj
        self.assertEqual(state.unrecorded_share_count(), 0)
        self.assertTrue(all(p.is_bot for p in state.players))


class UnrecordedSharesInPlayTestCase(unittest.TestCase):
    def setUp(self):
        self.root = tk.Tk()
        self.root.withdraw()
        state = GameState.new_hand_dealt_game(NAMES, MIXED_HANDS, MIXED_STOCK)
        state.game_setup_confirmed = True
        self.app = BoardSetupApp(self.root, state)
        self.app.pack()
        self.state = self.app.state_obj
        self.human = self.state.players[3]
        self.root.update_idletasks()

    def tearDown(self):
        self.root.destroy()

    def _widgets(self, parent):
        found = []

        def walk(p):
            for child in p.winfo_children():
                found.append(child)
                walk(child)

        walk(parent)
        return found

    def _texts(self, widget):
        texts = []
        for child in self._widgets(widget):
            try:
                texts.append(child.cget("text"))
            except Exception:
                pass
        return texts

    def _invoke(self, dialog, cls, text=None):
        for widget in self._widgets(dialog):
            if widget.winfo_class() == cls and (text is None or widget.cget("text") == text):
                widget.invoke()
                return

    def _drive_dialog(self, fill, open_dialog):
        """Run `fill(dialog, captured)` inside a modal dialog's nested event
        loop, then return what it captured plus `open_dialog`'s result.

        The dialog is closed no matter what `fill` does. Tk swallows an
        exception raised inside an `after` callback, so a dialog left open
        by a mistake in the test would hang the whole suite instead of
        failing one case -- which is exactly what it did once.
        """
        captured = {}

        def run():
            dialog = next(
                w for w in self.root.winfo_children() if isinstance(w, tk.Toplevel)
            )
            try:
                fill(dialog, captured)
            finally:
                if dialog.winfo_exists():
                    dialog.destroy()

        self.root.after(50, run)
        captured["result"] = open_dialog()
        return captured


class TestUnrecordedSharesInTheUI(UnrecordedSharesInPlayTestCase):
    def test_the_player_panel_shows_them_instead_of_four_zeros(self):
        """Splitting an unrecorded hand by ware would read as "holds
        nothing", which is the opposite of what it means."""
        self.app.refresh()
        self.root.update_idletasks()
        texts = " ".join(t for t in self._texts(self.app) if isinstance(t, str))
        self.assertIn("Unrec.", texts)

    def test_the_buy_dialog_says_what_it_could_not_count(self):
        self.app._show_buy_share_dialog(self.human, lambda: None)
        self.root.update_idletasks()
        dialog = next(w for w in self.root.winfo_children() if isinstance(w, tk.Toplevel))
        texts = " ".join(t for t in self._texts(dialog) if isinstance(t, str))
        self.assertIn("never recorded", texts)
        dialog.destroy()

    def test_the_buy_dialog_leaves_availability_alone(self):
        """The stock is public whatever the hands say, so it must still
        read exactly as it was set up."""
        self.app._show_buy_share_dialog(self.human, lambda: None)
        self.root.update_idletasks()
        dialog = next(w for w in self.root.winfo_children() if isinstance(w, tk.Toplevel))
        rows = {}
        for child in dialog.winfo_children():
            for cell in child.winfo_children():
                info = cell.grid_info()
                if info:
                    rows.setdefault(info["row"], {})[int(info["column"])] = cell.cget("text")
        jade = next(r for r in rows.values() if 0 in r and r[0].strip() == "Jade")
        self.assertEqual(jade[3], str(MIXED_STOCK[Ware.JADE]))
        dialog.destroy()

    def test_taking_credit_can_encumber_one_without_naming_it(self):
        """Rules p.8: an encumbered share is set aside face-down, so this
        is not a moment that reveals anything."""
        self.human.is_bot = False
        captured = self._drive_dialog(
            lambda dialog, out: out.update(
                labels=[
                    w.cget("text") for w in self._widgets(dialog) if w.winfo_class() == "TButton"
                ]
            )
            or self._invoke(dialog, "TButton"),
            lambda: self.app._prompt_encumber_share(self.human),
        )
        share = captured["result"]
        self.assertIsNotNone(share)
        self.assertFalse(share.is_recorded)
        self.assertIn(share, self.human.shares)
        self.assertTrue(
            all("unrecorded share" in text for text in captured["labels"]), captured["labels"]
        )


class TestRevealingAtGameEnd(UnrecordedSharesInPlayTestCase):
    def _reveal(self, picks=()):
        def fill(dialog, out):
            combos = [w for w in self._widgets(dialog) if w.winfo_class() == "TCombobox"]
            out["offered"] = list(combos[0].cget("values")) if combos else []
            for combo, ware in zip(combos, picks):
                combo.set(ware.value.title())
            if picks:
                self._invoke(dialog, "TButton")

        return self._drive_dialog(fill, lambda: self.app._reveal_unrecorded_shares(self.human))

    def test_revealing_names_the_shares(self):
        self._reveal([Ware.GINSENG, Ware.JADE])
        self.assertEqual(
            sorted(s.ware.value for s in self.human.shares), ["ginseng", "jade"]
        )
        self.assertEqual(self.state.unrecorded_share_count(), 0)

    def test_revealing_does_not_change_what_is_for_sale(self):
        before = {w: self.state.shares_available(w) for w in Ware}
        self._reveal([Ware.GINSENG, Ware.JADE])
        after = {w: self.state.shares_available(w) for w in Ware}
        self.assertEqual(before, after)

    def test_only_the_wares_the_stock_allows_are_offered(self):
        """3 nutmeg are already dealt out and 2 nutmeg are for sale, so no
        unrecorded share can be a nutmeg."""
        captured = self._reveal()  # look, don't answer
        self.assertEqual(sorted(captured["offered"]), ["Ginseng", "Jade"])

    def test_naming_more_of_a_ware_than_the_stock_allows_is_refused(self):
        """Both shares can't be ginseng -- only one is unaccounted for."""
        self._reveal([Ware.GINSENG, Ware.GINSENG])
        self.assertEqual(self.state.unrecorded_share_count(), 2)  # nothing was applied

    def test_a_revealed_share_counts_toward_the_final_score(self):
        self.state.black_market.values[Ware.JADE] = 20
        self.state.black_market.values[Ware.GINSENG] = 10
        *_, shares_value, _total, _forfeited = self.app._compute_fortune(self.human)
        self.assertEqual(shares_value, 0)  # nothing scoreable while unnamed
        self._reveal([Ware.GINSENG, Ware.JADE])
        *_, shares_value, _total, _forfeited = self.app._compute_fortune(self.human)
        self.assertEqual(shares_value, 30)

    def test_a_player_with_nothing_hidden_is_not_asked(self):
        self.app._reveal_unrecorded_shares(self.state.players[0])  # returns, no dialog
        self.assertEqual(
            [w for w in self.root.winfo_children() if isinstance(w, tk.Toplevel)], []
        )


if __name__ == "__main__":
    unittest.main()
