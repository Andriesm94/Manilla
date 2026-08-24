"""Regression tests for the pirate boarding/plunder timing rule:

Pirates may only act as a direct consequence of an actual dice roll --
boarding right after the second movement round's roll, plunder right
after the third. The pilot phase (which runs *before* the third roll)
must never trigger either on its own, even if it leaves a punt sitting
exactly on space 13 -- per the rulebook, "when a pilot moves a punt to
space 13, nothing happens (the pirates only attack immediately after
the movement round)".
"""

import os
import sys
import time
import tkinter as tk
import unittest
from unittest import mock

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from manilla.ui.board_setup import BoardSetupApp
from manilla.engine.models import Punt, PuntStatus, SEA_ROUTE_LENGTH, Ware


class TkTestCase(unittest.TestCase):
    """Builds a real (never-shown) BoardSetupApp per test."""

    def setUp(self):
        self.root = tk.Tk()
        self.app = BoardSetupApp(self.root)
        self.app.pack(fill=tk.BOTH, expand=True)
        self.state = self.app.state_obj
        self.state.game_setup_confirmed = True
        self.root.update_idletasks()

    def tearDown(self):
        self.root.destroy()

    def find_toplevels(self, title_substr):
        return [
            t for t in self.root.winfo_children() if isinstance(t, tk.Toplevel) and title_substr in t.title()
        ]

    def make_punt(self, idx, ware, position):
        punt = self.state.punts[idx]
        punt.ware = ware
        punt.ware_slots = Punt.new(punt.id, ware).ware_slots
        punt.status = PuntStatus.ON_ROUTE
        punt.position = position
        return punt

    def set_pirates(self, captain=True, second=False):
        players = self.state.players
        pb = self.state.pirate_boat
        pb.captain.occupant = players[0].id if captain else None
        pb.second.occupant = players[1].id if second else None

    def all_widgets(self, widget):
        acc = [widget]
        for child in widget.winfo_children():
            acc.extend(self.all_widgets(child))
        return acc

    def button_in(self, widget, text_contains):
        return next(
            w for w in self.all_widgets(widget) if w.winfo_class() == "TButton" and text_contains in w.cget("text")
        )


class TestPilotAloneNeverTriggersPirates(TkTestCase):
    """The core regression: moving a punt onto space 13 via a pilot must
    never, by itself, open a boarding/plunder dialog or pay anyone out."""

    def test_pilot_move_onto_13_does_not_board(self):
        punt = self.make_punt(0, Ware.NUTMEG, 12)
        self.set_pirates(captain=True, second=True)
        cash_before = [p.cash for p in self.state.players]

        self.app._apply_pilot_move(punt, 1)  # exactly reaches space 13
        self.root.update()

        self.assertEqual(punt.position, SEA_ROUTE_LENGTH)
        self.assertEqual(punt.status, PuntStatus.ON_ROUTE, "pilot alone must not dock/capture the punt")
        self.assertEqual([p.cash for p in self.state.players], cash_before, "no pirate payout from a pilot move")
        self.assertEqual(self.find_toplevels("Pirate captain"), [])
        self.assertEqual(self.find_toplevels("Second pirate"), [])

    def test_pilot_move_onto_13_does_not_plunder(self):
        punt = self.make_punt(0, Ware.JADE, 11)
        punt.ware_slots[0].occupant = self.state.players[2].id
        self.set_pirates(captain=True, second=False)

        with mock.patch("manilla.ui.board_setup.messagebox.askyesno") as askyesno:
            self.app._apply_pilot_move(punt, 2)  # large pilot, exactly reaches 13
            self.root.update()
            askyesno.assert_not_called()

        self.assertEqual(punt.position, SEA_ROUTE_LENGTH)
        self.assertEqual(punt.status, PuntStatus.ON_ROUTE)
        self.assertIsNotNone(punt.ware_slots[0].occupant, "accomplice must still be aboard -- nobody was plundered")

    def test_large_pilot_two_punt_move_onto_13_does_not_trigger_pirates(self):
        a = self.make_punt(0, Ware.SILK, 12)
        b = self.make_punt(1, Ware.GINSENG, 5)
        self.set_pirates(captain=True, second=True)

        self.app._apply_pilot_move(a, 1)
        self.app._apply_pilot_move(b, 1)
        self.root.update()

        self.assertEqual(a.position, SEA_ROUTE_LENGTH)
        self.assertEqual(a.status, PuntStatus.ON_ROUTE)
        self.assertEqual(self.find_toplevels("Pirate captain"), [])
        self.assertEqual(self.find_toplevels("Second pirate"), [])


class TestBoardingOnlyAfterRound2Roll(TkTestCase):
    def test_landing_on_13_after_round2_roll_offers_boarding(self):
        punt = self.make_punt(0, Ware.NUTMEG, 12)
        self.set_pirates(captain=True, second=False)
        self.state.movement_round_index = 1  # this roll becomes round 2

        with mock.patch("manilla.ui.board_setup.messagebox.showinfo"), mock.patch(
            "manilla.ui.board_setup.random.randint", return_value=1
        ):
            self.app._roll_dice_and_move()
        self.root.update()

        self.assertEqual(punt.position, SEA_ROUTE_LENGTH)
        self.assertEqual(punt.status, PuntStatus.ON_ROUTE, "boarding offer only -- captain hasn't chosen yet")
        self.assertEqual(len(self.find_toplevels("Pirate captain")), 1)

    def test_no_boarding_offer_after_round1_roll(self):
        punt = self.make_punt(0, Ware.NUTMEG, 12)
        self.set_pirates(captain=True, second=False)
        self.state.movement_round_index = 0  # this roll becomes round 1

        with mock.patch("manilla.ui.board_setup.messagebox.showinfo"), mock.patch(
            "manilla.ui.board_setup.random.randint", return_value=1
        ):
            self.app._roll_dice_and_move()
        self.root.update()

        self.assertEqual(punt.position, SEA_ROUTE_LENGTH)
        self.assertEqual(self.find_toplevels("Pirate captain"), [], "round 1 landing on 13 is not boardable")

    def test_no_boarding_offer_after_round3_roll_plunder_instead(self):
        punt = self.make_punt(0, Ware.NUTMEG, 12)
        self.set_pirates(captain=True, second=False)
        self.state.movement_round_index = 2  # this roll becomes round 3

        with mock.patch("manilla.ui.board_setup.messagebox.showinfo"), mock.patch(
            "manilla.ui.board_setup.messagebox.askyesno", return_value=True
        ), mock.patch("manilla.ui.board_setup.random.randint", return_value=1):
            self.app._roll_dice_and_move()
        self.root.update()

        self.assertEqual(self.find_toplevels("Pirate captain"), [], "round 3 is plunder, not the boarding dialog")
        self.assertEqual(punt.status, PuntStatus.IN_PORT, "plundered punt was resolved (sent to port)")


class TestPlunderOnlyAfterRound3Roll(TkTestCase):
    def test_pilot_leaves_punt_on_13_then_round3_roll_overshoots_to_port_not_plunder(self):
        """A punt the pilot leaves sitting on 13 still takes its mandatory
        round-3 die roll on top -- since the minimum roll is 1, it always
        moves past 13 and simply arrives, rather than being plundered.
        Confirms plunder resolution only ever happens post-roll, never as
        an immediate consequence of the pilot phase itself."""
        punt = self.make_punt(0, Ware.SILK, 11)
        self.set_pirates(captain=True, second=True)

        self.app._apply_pilot_move(punt, 2)  # large pilot: 11 -> 13
        self.assertEqual(punt.position, SEA_ROUTE_LENGTH)
        self.assertEqual(punt.status, PuntStatus.ON_ROUTE, "still just sitting on 13, unresolved")

        cash_before = [p.cash for p in self.state.players]
        self.state.movement_round_index = 2  # this roll becomes round 3
        with mock.patch("manilla.ui.board_setup.messagebox.showinfo"), mock.patch(
            "manilla.ui.board_setup.random.randint", return_value=3
        ):
            self.app._roll_dice_and_move()

        self.assertEqual(punt.status, PuntStatus.IN_PORT, "13 + a mandatory die roll always overshoots to port")
        self.assertEqual([p.cash for p in self.state.players], cash_before, "no plunder payout -- it wasn't caught")

    def test_landing_exactly_on_13_after_round3_roll_does_plunder(self):
        punt = self.make_punt(0, Ware.JADE, 12)
        punt.ware_slots[0].occupant = self.state.players[2].id
        self.set_pirates(captain=True, second=True)
        self.state.movement_round_index = 2

        cash_before = self.state.players[0].cash
        with mock.patch("manilla.ui.board_setup.messagebox.showinfo"), mock.patch(
            "manilla.ui.board_setup.messagebox.askyesno", return_value=True
        ), mock.patch("manilla.ui.board_setup.random.randint", return_value=1):
            self.app._roll_dice_and_move()

        self.assertGreater(self.state.players[0].cash, cash_before, "captain shares the plunder payout")
        self.assertIsNone(punt.ware_slots[0].occupant, "plundered punt's accomplices are returned empty-handed")
        self.assertEqual(punt.status, PuntStatus.IN_PORT)


class TestPlunderDestinationDecision(TkTestCase):
    """When a punt is plundered, the captain -- or the second pirate if
    they're aboard alone -- chooses port vs. shipyard for that punt,
    independently for each plundered punt."""

    def test_human_captain_sends_punt_to_port(self):
        punt = self.make_punt(0, Ware.NUTMEG, SEA_ROUTE_LENGTH)
        self.set_pirates(captain=True, second=False)

        with mock.patch("manilla.ui.board_setup.messagebox.askyesno", return_value=True) as askyesno:
            self.app._resolve_plunder(punt)

        askyesno.assert_called_once()
        self.assertIn(self.state.players[0].name, askyesno.call_args[0][1], "dialog names the captain")
        self.assertEqual(punt.status, PuntStatus.IN_PORT)

    def test_human_captain_sends_punt_to_shipyard(self):
        punt = self.make_punt(0, Ware.NUTMEG, SEA_ROUTE_LENGTH)
        self.set_pirates(captain=True, second=False)

        with mock.patch("manilla.ui.board_setup.messagebox.askyesno", return_value=False):
            self.app._resolve_plunder(punt)

        self.assertEqual(punt.status, PuntStatus.IN_SHIPYARD)

    def test_bot_captain_decides_without_a_dialog(self):
        punt = self.make_punt(0, Ware.NUTMEG, SEA_ROUTE_LENGTH)
        self.set_pirates(captain=True, second=False)
        self.state.players[0].is_bot = True

        with mock.patch("manilla.ui.board_setup.messagebox.askyesno") as askyesno, mock.patch(
            "manilla.ui.board_setup.random.random", return_value=0.0
        ):
            self.app._resolve_plunder(punt)

        askyesno.assert_not_called()
        self.assertEqual(punt.status, PuntStatus.IN_PORT)

    def test_lone_second_pirate_decides_when_captain_absent(self):
        """No captain aboard -- the decision falls to the second pirate,
        by name, not a generic 'The pirates' placeholder."""
        punt = self.make_punt(0, Ware.NUTMEG, SEA_ROUTE_LENGTH)
        self.set_pirates(captain=False, second=True)

        with mock.patch("manilla.ui.board_setup.messagebox.askyesno", return_value=True) as askyesno:
            self.app._resolve_plunder(punt)

        askyesno.assert_called_once()
        self.assertIn(self.state.players[1].name, askyesno.call_args[0][1], "dialog names the second pirate")
        self.assertEqual(punt.status, PuntStatus.IN_PORT)

    def test_lone_second_pirate_bot_decides_without_blocking_a_dialog(self):
        """The bug this guards against: with no captain aboard, a bot
        second pirate must not fall through to a blocking human dialog --
        that would hang an all-bot simulation."""
        punt = self.make_punt(0, Ware.NUTMEG, SEA_ROUTE_LENGTH)
        self.set_pirates(captain=False, second=True)
        self.state.players[1].is_bot = True

        with mock.patch("manilla.ui.board_setup.messagebox.askyesno") as askyesno, mock.patch(
            "manilla.ui.board_setup.random.random", return_value=0.99
        ):
            self.app._resolve_plunder(punt)

        askyesno.assert_not_called()
        self.assertEqual(punt.status, PuntStatus.IN_SHIPYARD)

    def test_each_plundered_punt_decided_independently(self):
        a = self.make_punt(0, Ware.NUTMEG, SEA_ROUTE_LENGTH)
        b = self.make_punt(1, Ware.SILK, SEA_ROUTE_LENGTH)
        self.set_pirates(captain=True, second=False)

        with mock.patch("manilla.ui.board_setup.messagebox.askyesno", side_effect=[True, False]):
            self.app._resolve_plunder(a)
            self.app._resolve_plunder(b)

        self.assertEqual(a.status, PuntStatus.IN_PORT)
        self.assertEqual(b.status, PuntStatus.IN_SHIPYARD)


class TestPilotPhaseIntegration(TkTestCase):
    """Drives the real turn -> pilot dialog -> round-3 roll chain (not just
    the individual helpers) to confirm nothing pirate-related fires while
    the pilot dialog is still open, only after it resolves into a roll."""

    def test_no_pirate_dialog_appears_during_the_pilot_phase_itself(self):
        punt = self.make_punt(0, Ware.NUTMEG, 12)
        self.state.pilot_island.small.occupant = self.state.players[3].id
        self.set_pirates(captain=True, second=False)

        # Fast-forward to "one placement away from completing round 2",
        # about to trigger the pilot phase ahead of round 3's roll.
        self.state.movement_round_index = 2
        self.app._round_placements = len(self.state.players) - 1

        vacant_slot = self.state.port.slots["A"]
        self.app._place_or_remove_accomplice(vacant_slot)
        self.root.update()

        small_dialog = self.find_toplevels("Small Pilot")
        self.assertEqual(len(small_dialog), 1, "pilot phase should be showing, dice not rolled yet")
        self.assertEqual(punt.position, SEA_ROUTE_LENGTH - 1, "punt untouched so far")
        self.assertEqual(self.find_toplevels("Pirate captain"), [], "no pirate action before the roll")

        # Move the punt onto 13 via the small pilot. Resolving the dialog
        # runs the whole rest of the chain synchronously (pilot move, then
        # the automatic round-3 roll) -- there's no observable gap between
        # them from outside, which is itself part of the guarantee: nothing
        # can act on "punt sitting on 13" in between. Mock the roll so the
        # outcome is deterministic and we can check what actually resolved
        # it wasn't a stray pirate action fired off the pilot move.
        dlg = small_dialog[0]
        combos = [w for w in self.all_widgets(dlg) if w.winfo_class() == "TCombobox"]
        combos[0].set(f"Punt {punt.id} ({punt.ware.value}, at space {punt.position})")
        forward_btn = self.button_in(dlg, "forward")
        with mock.patch("manilla.ui.board_setup.messagebox.showinfo"), mock.patch(
            "manilla.ui.board_setup.random.randint", return_value=4
        ):
            forward_btn.invoke()
        self.root.update()

        self.assertEqual(self.state.movement_round_index, 3, "resolved by the real round-3 roll, not mid-pilot-phase")
        self.assertEqual(punt.status, PuntStatus.IN_PORT, "13 + the mandatory die roll (4) overshoots to port")
        self.assertEqual(self.find_toplevels("Pirate captain"), [], "boarding never applies outside round 2")
        self.assertEqual(self.find_toplevels("Second pirate"), [])


class TestSecondPirateBecomesCaptain(TkTestCase):
    """When the captain boards a punt, their piece physically leaves the
    boat. If the second pirate stays behind, they immediately move up to
    the now-vacant captain spot ("the forward position")."""

    def test_second_promotes_when_captain_boards_and_second_skips(self):
        self.make_punt(0, Ware.JADE, SEA_ROUTE_LENGTH)
        self.set_pirates(captain=True, second=True)
        captain_id, second_id = self.state.pirate_boat.captain.occupant, self.state.pirate_boat.second.occupant

        self.app._handle_pirate_boarding()
        self.root.update()

        captain_dlg = self.find_toplevels("Pirate captain")
        self.assertEqual(len(captain_dlg), 1)
        self.button_in(captain_dlg[0], "Board Punt").invoke()
        self.root.update()

        second_dlg = self.find_toplevels("Second pirate")
        self.assertEqual(len(second_dlg), 1)
        self.button_in(second_dlg[0], "Skip").invoke()
        self.root.update()

        pb = self.state.pirate_boat
        self.assertEqual(pb.captain.occupant, second_id, "second pirate moved up to the captain spot")
        self.assertIsNone(pb.second.occupant, "second slot is now empty")
        self.assertNotEqual(pb.captain.occupant, captain_id, "the old captain is gone -- they boarded the punt")

    def test_no_promotion_when_both_board(self):
        self.make_punt(0, Ware.JADE, SEA_ROUTE_LENGTH)  # 4 ware slots -- room for both
        self.set_pirates(captain=True, second=True)

        self.app._handle_pirate_boarding()
        self.root.update()
        self.button_in(self.find_toplevels("Pirate captain")[0], "Board Punt").invoke()
        self.root.update()
        self.button_in(self.find_toplevels("Second pirate")[0], "Board Punt").invoke()
        self.root.update()

        pb = self.state.pirate_boat
        self.assertIsNone(pb.captain.occupant, "nobody left in the boat to be captain")
        self.assertIsNone(pb.second.occupant)

    def test_no_promotion_when_captain_skips_even_if_second_boards(self):
        self.make_punt(0, Ware.NUTMEG, SEA_ROUTE_LENGTH)
        self.set_pirates(captain=True, second=True)
        captain_id = self.state.pirate_boat.captain.occupant

        self.app._handle_pirate_boarding()
        self.root.update()
        self.button_in(self.find_toplevels("Pirate captain")[0], "Skip").invoke()
        self.root.update()
        self.button_in(self.find_toplevels("Second pirate")[0], "Board Punt").invoke()
        self.root.update()

        pb = self.state.pirate_boat
        self.assertEqual(pb.captain.occupant, captain_id, "captain stayed put -- still captain, nothing to promote")
        self.assertIsNone(pb.second.occupant, "second boarded and left")

    def test_nothing_changes_when_both_skip(self):
        self.make_punt(0, Ware.SILK, SEA_ROUTE_LENGTH)
        self.set_pirates(captain=True, second=True)
        captain_id, second_id = self.state.pirate_boat.captain.occupant, self.state.pirate_boat.second.occupant

        self.app._handle_pirate_boarding()
        self.root.update()
        self.button_in(self.find_toplevels("Pirate captain")[0], "Skip").invoke()
        self.root.update()
        self.button_in(self.find_toplevels("Second pirate")[0], "Skip").invoke()
        self.root.update()

        pb = self.state.pirate_boat
        self.assertEqual(pb.captain.occupant, captain_id)
        self.assertEqual(pb.second.occupant, second_id)

    def test_solo_captain_boarding_does_not_crash_with_no_second(self):
        self.make_punt(0, Ware.GINSENG, SEA_ROUTE_LENGTH)
        self.set_pirates(captain=True, second=False)

        self.app._handle_pirate_boarding()
        self.root.update()
        self.button_in(self.find_toplevels("Pirate captain")[0], "Board Punt").invoke()
        self.root.update()

        pb = self.state.pirate_boat
        self.assertIsNone(pb.captain.occupant, "captain left, and there's no second pirate to promote")
        self.assertIsNone(pb.second.occupant)

    def test_bot_captain_boarding_also_promotes_a_skipping_bot_second(self):
        self.make_punt(0, Ware.JADE, SEA_ROUTE_LENGTH)
        self.set_pirates(captain=True, second=True)
        self.state.players[0].is_bot = True  # captain
        self.state.players[1].is_bot = True  # second
        second_id = self.state.pirate_boat.second.occupant

        # Force "always board" for the captain's roll, then "always skip"
        # for the second's roll, by controlling random.random()'s sequence.
        with mock.patch("manilla.ui.board_setup.random.random", side_effect=[0.0, 0.99]), mock.patch(
            "manilla.ui.board_setup.random.choice", side_effect=lambda seq: seq[0]
        ):
            self.app._handle_pirate_boarding()
            deadline = time.time() + 3
            while self.state.pirate_boat.second.occupant is not None and time.time() < deadline:
                self.root.update()
                time.sleep(0.01)

        pb = self.state.pirate_boat
        self.assertEqual(pb.captain.occupant, second_id, "bot second promoted after the bot captain boarded")
        self.assertIsNone(pb.second.occupant)


class TestPirateSlotPlacementInvariant(TkTestCase):
    """It must never be possible for the pirate boat to hold only a second
    pirate with no captain -- placement always fills the captain slot
    first, and removing the captain promotes a present second pirate."""

    def test_clicking_vacant_second_circle_places_as_captain(self):
        pb = self.state.pirate_boat

        self.app._place_or_remove_pirate_slot(pb.second, True)

        self.assertIsNotNone(pb.captain.occupant, "placement redirected to the captain slot")
        self.assertIsNone(pb.second.occupant, "second stays vacant until a captain is aboard")

    def test_clicking_vacant_captain_circle_places_as_captain(self):
        pb = self.state.pirate_boat

        self.app._place_or_remove_pirate_slot(pb.captain, False)

        self.assertIsNotNone(pb.captain.occupant)
        self.assertIsNone(pb.second.occupant)

    def test_second_only_fills_once_captain_is_present(self):
        self.set_pirates(captain=True, second=False)
        pb = self.state.pirate_boat
        self.state.current_turn_player_id = self.state.players[1].id

        self.app._place_or_remove_pirate_slot(pb.second, True)

        self.assertEqual(pb.second.occupant, self.state.players[1].id)
        self.assertEqual(pb.captain.occupant, self.state.players[0].id, "captain untouched")

    def test_removing_captain_while_second_present_promotes_second_and_refunds_captain(self):
        self.set_pirates(captain=True, second=True)
        pb = self.state.pirate_boat
        captain_id, second_id = pb.captain.occupant, pb.second.occupant
        captain_player = self.state.player_by_id(captain_id)
        cash_before = captain_player.cash

        self.app._place_or_remove_pirate_slot(pb.captain, False)

        self.assertEqual(pb.captain.occupant, second_id, "second promoted into the vacated captain slot")
        self.assertIsNone(pb.second.occupant)
        self.assertEqual(captain_player.cash, cash_before + pb.captain.price, "old captain refunded on removal")

    def test_removing_second_alone_leaves_captain_untouched(self):
        self.set_pirates(captain=True, second=True)
        pb = self.state.pirate_boat
        captain_id, second_id = pb.captain.occupant, pb.second.occupant
        second_player = self.state.player_by_id(second_id)
        cash_before = second_player.cash

        self.app._place_or_remove_pirate_slot(pb.second, True)

        self.assertEqual(pb.captain.occupant, captain_id, "removing the second must not disturb the captain")
        self.assertIsNone(pb.second.occupant)
        self.assertEqual(second_player.cash, cash_before + pb.second.price)


class TestPuntStandingsNotification(TkTestCase):
    """An explicit notification reports where every punt sits right after
    the pilot phase resolves, before the mandatory third and final roll."""

    def test_standings_message_lists_each_loaded_punt(self):
        """Only ON_ROUTE and IN_PORT are reachable states at this point --
        shipwrecks are decided by the final roll itself, so a punt can
        never already be in the shipyard here."""
        self.make_punt(0, Ware.NUTMEG, 9)
        self.make_punt(1, Ware.SILK, SEA_ROUTE_LENGTH)
        self.state.punts[1].status = PuntStatus.IN_PORT  # arrived early, overshot 13 in an earlier round
        self.make_punt(2, Ware.GINSENG, 4)

        with mock.patch("manilla.ui.board_setup.messagebox.showinfo") as showinfo, mock.patch(
            "manilla.ui.board_setup.random.randint", return_value=1
        ):
            self.app._show_punt_standings_before_final_roll()

        title, body = showinfo.call_args_list[0].args
        self.assertIn("Punt standings", title)
        self.assertIn("Nutmeg: space 9 of 13", body)
        self.assertIn("Silk: already in port", body)
        self.assertIn("Ginseng: space 4 of 13", body)

    def test_standings_shown_after_pilots_and_before_the_roll(self):
        punt = self.make_punt(0, Ware.NUTMEG, 12)
        self.state.pilot_island.small.occupant = self.state.players[3].id
        self.state.movement_round_index = 2
        self.app._round_placements = len(self.state.players) - 1

        calls = []
        with mock.patch(
            "manilla.ui.board_setup.messagebox.showinfo", side_effect=lambda title, body: calls.append(title)
        ):
            self.app._place_or_remove_accomplice(self.state.port.slots["A"])
            self.root.update()
            dlg = self.find_toplevels("Small Pilot")[0]
            combos = [w for w in self.all_widgets(dlg) if w.winfo_class() == "TCombobox"]
            combos[0].set(f"Punt {punt.id} ({punt.ware.value}, at space {punt.position})")
            with mock.patch("manilla.ui.board_setup.random.randint", return_value=1):
                self.button_in(dlg, "forward").invoke()
            self.root.update()

        self.assertEqual(calls, ["Punt standings before the final roll", "Punts move"], "standings, then the roll")

    def test_no_blocking_dialog_in_an_all_bot_game(self):
        for p in self.state.players:
            p.is_bot = True

        with mock.patch("manilla.ui.board_setup.messagebox.showinfo") as showinfo:
            self.app._show_punt_standings_before_final_roll()

        showinfo.assert_not_called()


if __name__ == "__main__":
    unittest.main()
