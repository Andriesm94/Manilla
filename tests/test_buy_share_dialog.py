"""Tests for the harbor master's "Buy a Share?" dialog.

The decision needs three things the dialog used to omit: what the player
already holds, how many of each ware are left, and the black-market level
itself. That last one is not recoverable from the price -- `share_price`
floors at 5, so a ware at level 0 and one at level 5 both cost 5 while
being worth very different amounts later.
"""

import os
import sys
import tkinter as tk
import unittest
from unittest import mock

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from manilla.ui.board_setup import BoardSetupApp
from manilla.engine.models import SHARES_PER_WARE, GameState, Share, Ware


class BuyShareDialogTestCase(unittest.TestCase):
    def setUp(self):
        state = GameState.new_default_game(["Ann", "Bo", "Cid", "Di"])
        state.game_setup_confirmed = True
        for player in state.players:
            player.shares = []
        self.root = tk.Tk()
        self.root.withdraw()  # never actually show a window during tests
        self.app = BoardSetupApp(self.root, state)
        self.app.pack(fill=tk.BOTH, expand=True)
        self.state = self.app.state_obj
        self.me = self.state.players[0]
        self.root.update_idletasks()

    def tearDown(self):
        self.root.destroy()

    def _open(self):
        self.app._show_buy_share_dialog(self.me, lambda: None)
        self.root.update_idletasks()
        return next(w for w in self.root.winfo_children() if isinstance(w, tk.Toplevel))

    def _widgets(self, dialog):
        """Every widget in the dialog, flattened, as (class, text) pairs."""
        found = []

        def walk(parent):
            for child in parent.winfo_children():
                try:
                    found.append((child.winfo_class(), child.cget("text")))
                except Exception:
                    pass
                walk(child)

        walk(dialog)
        return found

    def _texts(self, dialog):
        return [text for _cls, text in self._widgets(dialog)]


class TestEveryWareIsListed(BuyShareDialogTestCase):
    def test_shows_all_four_wares_even_when_one_is_sold_out(self):
        # Corner ginseng so nobody can buy it -- it should still be listed,
        # because its market value is what already-held shares are worth.
        self.state.players[1].shares = [Share(ware=Ware.GINSENG) for _ in range(SHARES_PER_WARE)]
        dialog = self._open()
        texts = " ".join(self._texts(dialog))
        for ware in Ware:
            self.assertIn(ware.value.title(), texts)

    def test_a_sold_out_ware_is_not_selectable(self):
        self.state.players[1].shares = [Share(ware=Ware.GINSENG) for _ in range(SHARES_PER_WARE)]
        dialog = self._open()
        radios = [text for cls, text in self._widgets(dialog) if cls == "TRadiobutton"]
        self.assertNotIn("Ginseng", [t.strip() for t in radios])
        # ...while the others still are.
        self.assertIn("Jade", [t.strip() for t in radios])


class TestMarketLevelIsVisible(BuyShareDialogTestCase):
    def test_level_and_price_are_shown_separately(self):
        """A ware at 0 and a ware at 5 both cost 5, so price alone can't
        tell them apart -- the level has to be shown in its own right."""
        self.state.black_market.values[Ware.NUTMEG] = 0
        self.state.black_market.values[Ware.SILK] = 5
        dialog = self._open()

        rows = {}
        for child in dialog.winfo_children():
            for cell in child.winfo_children():
                info = cell.grid_info()
                if info:
                    rows.setdefault(info["row"], {})[int(info["column"])] = cell.cget("text")

        by_ware = {r[0].strip(): r for r in rows.values() if 0 in r}
        self.assertEqual(by_ware["Nutmeg"][1], "0")  # market level
        self.assertEqual(by_ware["Nutmeg"][2], "5")  # price, floored
        self.assertEqual(by_ware["Silk"][1], "5")
        self.assertEqual(by_ware["Silk"][2], "5")


class TestPlayersOwnHoldings(BuyShareDialogTestCase):
    def test_counts_the_players_shares_and_flags_encumbered_ones(self):
        self.me.shares = [
            Share(ware=Ware.JADE),
            Share(ware=Ware.JADE, encumbered=True),
            Share(ware=Ware.SILK),
        ]
        dialog = self._open()
        texts = self._texts(dialog)
        self.assertIn("2 (1 encumbered)", texts)  # jade
        self.assertIn("1", texts)  # silk

    def test_shows_the_players_cash(self):
        self.me.cash = 23
        dialog = self._open()
        self.assertTrue(any("23 PESOS" in t for t in self._texts(dialog)))

    def test_another_players_shares_are_not_counted_as_mine(self):
        self.state.players[1].shares = [Share(ware=Ware.JADE) for _ in range(3)]
        dialog = self._open()
        rows = {}
        for child in dialog.winfo_children():
            for cell in child.winfo_children():
                info = cell.grid_info()
                if info:
                    rows.setdefault(info["row"], {})[int(info["column"])] = cell.cget("text")
        jade = next(r for r in rows.values() if 0 in r and r[0].strip() == "Jade")
        self.assertEqual(jade[3], "2")  # 5 exist, 3 are held by someone else
        self.assertEqual(jade[4], "0")  # ...but none of them are mine


class TestBuyingStillWorks(BuyShareDialogTestCase):
    def test_buy_button_grants_the_selected_share_and_charges_for_it(self):
        self.state.black_market.values[Ware.NUTMEG] = 10
        self.me.cash = 30
        dialog = self._open()

        buttons = {}

        def collect(parent):
            for child in parent.winfo_children():
                if child.winfo_class() == "TButton":
                    buttons[child.cget("text")] = child
                collect(child)

        collect(dialog)
        # Select nutmeg, then buy.
        for child in dialog.winfo_children():
            for cell in child.winfo_children():
                if cell.winfo_class() == "TRadiobutton" and cell.cget("text") == "Nutmeg":
                    cell.invoke()
        buttons["Buy"].invoke()

        self.assertEqual([s.ware for s in self.me.shares], [Ware.NUTMEG])
        self.assertEqual(self.me.cash, 20)

    def test_skip_buys_nothing(self):
        dialog = self._open()
        for child in dialog.winfo_children():
            for cell in child.winfo_children():
                if cell.winfo_class() == "TButton" and cell.cget("text") == "Skip":
                    cell.invoke()
        self.assertEqual(self.me.shares, [])


if __name__ == "__main__":
    unittest.main()
