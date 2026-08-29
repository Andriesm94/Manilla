"""Tests for `manilla.engine.selfplay_data`'s JSONL training/analysis
data collection. Random policy throughout -- REV self-play is
correctness-tested elsewhere (`tests/test_selfplay.py`) and is too slow
to spend on data-format checks.
"""

import json
import tempfile
import unittest
from pathlib import Path

from manilla.engine.models import GameState, Punt, PuntStatus, Ware
from manilla.engine.selfplay_data import (
    _favored_wares,
    _seat_offsets,
    record_self_play_games,
)


class TestSeatOffsets(unittest.TestCase):
    def test_harbor_master_is_offset_zero_and_order_wraps_around(self):
        state = GameState.new_default_game(["A", "B", "C", "D"])
        harbor_master = state.players[2]  # p2
        offsets = _seat_offsets(state, harbor_master)
        self.assertEqual(offsets["p2"], 0)
        self.assertEqual(offsets["p3"], 1)
        self.assertEqual(offsets["p0"], 2)  # wraps past the end of the list
        self.assertEqual(offsets["p1"], 3)


class TestFavoredWares(unittest.TestCase):
    def test_picks_the_two_highest_starting_positions(self):
        state = GameState.new_default_game(["A", "B", "C", "D"])
        state.punts[0] = Punt.new(0, Ware.GINSENG)
        state.punts[0].position = 1
        state.punts[0].status = PuntStatus.ON_ROUTE
        state.punts[1] = Punt.new(1, Ware.NUTMEG)
        state.punts[1].position = 5
        state.punts[1].status = PuntStatus.ON_ROUTE
        state.punts[2] = Punt.new(2, Ware.SILK)
        state.punts[2].position = 3
        state.punts[2].status = PuntStatus.ON_ROUTE

        self.assertEqual(_favored_wares(state), {"nutmeg", "silk"})

    def test_ignores_the_unloaded_punt(self):
        state = GameState.new_default_game(["A", "B", "C", "D"])
        state.punts[0] = Punt.new(0, Ware.GINSENG)
        state.punts[0].position = 4
        state.punts[1] = Punt.new(1, Ware.NUTMEG)
        state.punts[1].position = 2
        state.punts[2] = Punt.new(2, None)  # left ashore -- never loaded
        result = _favored_wares(state)
        self.assertEqual(len(result), 2)
        self.assertIn("ginseng", result)
        self.assertIn("nutmeg", result)


class TestRecordSelfPlayGames(unittest.TestCase):
    def test_writes_both_jsonl_files_with_expected_shape(self):
        with tempfile.TemporaryDirectory() as tmp:
            results = record_self_play_games(
                3, player_count=4, policy="random", seed=1, out_dir=tmp
            )
            self.assertEqual(len(results), 3)

            profit_rows = [
                json.loads(line)
                for line in Path(tmp, "harbor_master_profit.jsonl").read_text().splitlines()
            ]
            training_rows = [
                json.loads(line)
                for line in Path(tmp, "bid_buy_training.jsonl").read_text().splitlines()
            ]

            self.assertGreater(len(profit_rows), 0)
            self.assertEqual(len(profit_rows), len(training_rows))  # one voyage produces one of each

            game_ids = {r["game_id"] for r in profit_rows}
            self.assertEqual(len(game_ids), 3)  # one distinct id per game

            row = profit_rows[0]
            self.assertIn("voyage_number", row)
            self.assertIn("player_count", row)
            self.assertEqual(row["policy"], "random")
            # JSON object keys are always strings, even though the seat
            # offsets were ints on the Python side.
            self.assertEqual(set(row["pesos_by_seat_offset"].keys()), {"0", "1", "2", "3"})

            trow = training_rows[0]
            self.assertEqual(set(trow["shares_in_play"].keys()), {w.value for w in Ware})
            self.assertEqual(len(trow["favored_wares"]), 2)
            # Every voyage's row gets the same game's final standings,
            # filled in only after that game actually finished.
            self.assertIsNotNone(trow["final_black_market"])
            self.assertTrue(any(v >= 30 for v in trow["final_black_market"].values()))

    def test_every_voyage_in_a_game_shares_the_same_final_label(self):
        with tempfile.TemporaryDirectory() as tmp:
            record_self_play_games(2, player_count=4, policy="random", seed=2, out_dir=tmp)
            rows = [
                json.loads(line)
                for line in Path(tmp, "bid_buy_training.jsonl").read_text().splitlines()
            ]
            by_game: dict = {}
            for row in rows:
                by_game.setdefault(row["game_id"], []).append(row["final_black_market"])
            for game_id, labels in by_game.items():
                self.assertEqual(len(set(json.dumps(label, sort_keys=True) for label in labels)), 1)

    def test_appends_across_repeated_calls_rather_than_overwriting(self):
        with tempfile.TemporaryDirectory() as tmp:
            record_self_play_games(1, player_count=4, policy="random", seed=3, out_dir=tmp)
            record_self_play_games(1, player_count=4, policy="random", seed=4, out_dir=tmp)
            rows = Path(tmp, "harbor_master_profit.jsonl").read_text().splitlines()
            game_ids = {json.loads(line)["game_id"] for line in rows}
            self.assertEqual(len(game_ids), 2)


if __name__ == "__main__":
    unittest.main()
