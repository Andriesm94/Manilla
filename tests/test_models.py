import json
import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from manilla.engine.models import (
    BLACK_MARKET_LEVELS,
    PIRATE_PRICE,
    PLUNDER_PAYOUTS,
    SHARES_PER_WARE,
    BlackMarket,
    GameState,
    Phase,
    Punt,
    PuntStatus,
    Share,
    Ware,
)


class TestDefaultGame(unittest.TestCase):
    def test_default_4_player_game(self):
        state = GameState.new_default_game(["A", "B", "C", "D"])
        self.assertEqual(state.player_count, 4)
        self.assertEqual(state.voyage_number, 1)
        self.assertEqual(state.phase, Phase.AUCTION)
        self.assertEqual(state.accomplice_rounds_total, 3)

        for player in state.players:
            self.assertEqual(player.cash, 30)
            self.assertEqual(player.accomplices_in_hand, 3)
            self.assertEqual(len(player.shares), 2)

        # No harbor master is pre-assigned -- that's decided by the auction.
        harbor_masters = [p for p in state.players if p.is_harbor_master]
        self.assertEqual(len(harbor_masters), 0)

    def test_default_3_player_game_has_four_accomplices_and_four_rounds(self):
        state = GameState.new_default_game(["A", "B", "C"])
        self.assertEqual(state.player_count, 3)
        self.assertEqual(state.accomplice_rounds_total, 4)
        for player in state.players:
            self.assertEqual(player.accomplices_in_hand, 4)

    def test_rejects_invalid_player_counts(self):
        with self.assertRaises(ValueError):
            GameState.new_default_game(["A", "B"])
        with self.assertRaises(ValueError):
            GameState.new_default_game(["A", "B", "C", "D", "E", "F"])


class TestBlackMarket(unittest.TestCase):
    def test_share_price_floor_is_five(self):
        market = BlackMarket()
        market.values[Ware.NUTMEG] = 0
        self.assertEqual(market.share_price(Ware.NUTMEG), 5)

    def test_share_price_matches_value_above_floor(self):
        market = BlackMarket()
        market.values[Ware.SILK] = 20
        self.assertEqual(market.share_price(Ware.SILK), 20)

    def test_game_over_when_any_ware_hits_thirty(self):
        market = BlackMarket()
        self.assertFalse(market.is_game_over())
        market.values[Ware.JADE] = 30
        self.assertTrue(market.is_game_over())

    def test_raise_value_follows_the_nonuniform_track(self):
        market = BlackMarket()
        self.assertEqual(BLACK_MARKET_LEVELS, [0, 5, 10, 20, 30])
        expected_next = {0: 5, 5: 10, 10: 20, 20: 30, 30: 30}
        for current, expected in expected_next.items():
            market.values[Ware.NUTMEG] = current
            market.raise_value(Ware.NUTMEG)
            self.assertEqual(market.values[Ware.NUTMEG], expected)


class TestValidation(unittest.TestCase):
    def test_flags_bad_start_position_sum(self):
        state = GameState.new_default_game(["A", "B", "C", "D"])
        for i, punt in enumerate(state.punts):
            punt.ware = list(Ware)[i]
            punt.status = PuntStatus.ON_ROUTE
            punt.position = 0
        warnings = state.validate()
        self.assertTrue(any("sum to" in w for w in warnings))

    def test_accepts_valid_start_position_sum(self):
        state = GameState.new_default_game(["A", "B", "C", "D"])
        positions = [4, 3, 2]
        for punt, pos in zip(state.punts, positions):
            punt.status = PuntStatus.ON_ROUTE
            punt.position = pos
        warnings = state.validate()
        self.assertFalse(any("sum to" in w for w in warnings))


class TestShares(unittest.TestCase):
    def test_default_game_never_over_allocates_a_ware(self):
        state = GameState.new_default_game(["A", "B", "C", "D"])
        for ware in Ware:
            self.assertLessEqual(state.shares_owned(ware), SHARES_PER_WARE)
            self.assertEqual(
                state.shares_available(ware), SHARES_PER_WARE - state.shares_owned(ware)
            )

    def test_shares_available_reflects_ownership(self):
        state = GameState.new_default_game(["A", "B", "C", "D"])
        for p in state.players:
            p.shares = []
        state.players[0].shares = [Share(ware=Ware.JADE), Share(ware=Ware.JADE)]
        state.players[1].shares = [Share(ware=Ware.JADE, encumbered=True)]
        self.assertEqual(state.shares_owned(Ware.JADE), 3)
        self.assertEqual(state.shares_available(Ware.JADE), 2)
        self.assertEqual(state.shares_available(Ware.NUTMEG), SHARES_PER_WARE)

    def test_validate_flags_over_allocated_shares(self):
        state = GameState.new_default_game(["A", "B", "C", "D"])
        state.players[0].shares = [Share(ware=Ware.SILK) for _ in range(SHARES_PER_WARE + 1)]
        for p in state.players[1:]:
            p.shares = []
        warnings = state.validate()
        self.assertTrue(any("silk" in w and "only 5 exist" in w for w in warnings))


class TestPirateBoatAndPlunder(unittest.TestCase):
    def test_pirate_slots_have_a_price(self):
        state = GameState.new_default_game(["A", "B", "C", "D"])
        self.assertEqual(state.pirate_boat.captain.price, PIRATE_PRICE)
        self.assertEqual(state.pirate_boat.second.price, PIRATE_PRICE)

    def test_plunder_payouts_match_the_rulebook(self):
        self.assertEqual(PLUNDER_PAYOUTS[Ware.NUTMEG], 24)
        self.assertEqual(PLUNDER_PAYOUTS[Ware.GINSENG], 18)
        self.assertEqual(PLUNDER_PAYOUTS[Ware.JADE], 36)
        self.assertEqual(PLUNDER_PAYOUTS[Ware.SILK], 30)


class TestSerialization(unittest.TestCase):
    def test_round_trip_through_json(self):
        state = GameState.new_default_game(["A", "B", "C", "D"])
        state.punts[0].ware = Ware.JADE
        state.punts[0].position = 4
        state.black_market.values[Ware.SILK] = 15
        state.pirate_boat.captain.occupant = state.players[0].id
        state.current_turn_player_id = state.players[1].id

        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "state.json")
            state.save(path)
            self.assertTrue(os.path.exists(path))

            with open(path, "r", encoding="utf-8") as f:
                raw = json.load(f)
            self.assertEqual(raw["black_market"]["silk"], 15)

            loaded = GameState.load(path)

        self.assertEqual(loaded.player_count, state.player_count)
        self.assertEqual(loaded.punts[0].ware, Ware.JADE)
        self.assertEqual(loaded.punts[0].position, 4)
        self.assertEqual(loaded.black_market.values[Ware.SILK], 15)
        self.assertEqual(loaded.players[0].name, state.players[0].name)
        self.assertEqual(loaded.pirate_boat.captain.occupant, state.players[0].id)
        self.assertEqual(loaded.pirate_boat.captain.price, PIRATE_PRICE)
        self.assertEqual(loaded.current_turn_player_id, state.players[1].id)


if __name__ == "__main__":
    unittest.main()
