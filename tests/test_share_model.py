"""Tests for `manilla.engine.share_model` -- the learned share-buying
model (Roadmap milestone 5)."""

import json
import os
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from manilla.engine.models import GameState, Ware
from manilla.engine.share_model import (
    DEFAULT_COEFFICIENTS,
    FEATURE_NAMES,
    ShareValueModel,
    best_share_to_buy,
    default_model,
    evaluate,
    extract_features,
    favored_wares_from_setup,
    fit_ridge,
    load_rows,
    train,
    train_test_split,
)


def _row(game_id, voyage, shares, favored, market, final, schema_version=2):
    return {
        "game_id": game_id,
        "voyage_number": voyage,
        "player_count": 4,
        "policy": "rev",
        "schema_version": schema_version,
        "shares_in_play": shares,
        "favored_wares": favored,
        "black_market": market,
        "final_black_market": final,
    }


FLAT = {w.value: 0 for w in Ware}


class TestExtractFeatures(unittest.TestCase):
    def test_row_length_matches_the_declared_feature_names(self):
        row = extract_features(Ware.JADE, FLAT, ["jade"], 1, FLAT)
        self.assertEqual(len(row), len(FEATURE_NAMES))

    def test_flags_the_favored_ware_and_reads_its_own_level(self):
        market = {**FLAT, "jade": 20, "silk": 5}
        jade = extract_features(Ware.JADE, FLAT, ["jade", "silk"], 3, market)
        nutmeg = extract_features(Ware.NUTMEG, FLAT, ["jade", "silk"], 3, market)
        by_name = dict(zip(FEATURE_NAMES, jade))
        self.assertEqual(by_name["is_favored"], 1.0)
        self.assertEqual(by_name["black_market"], 20.0)
        self.assertEqual(dict(zip(FEATURE_NAMES, nutmeg))["is_favored"], 0.0)

    def test_totals_span_every_ware_not_just_this_one(self):
        market = {**FLAT, "jade": 20, "silk": 10}
        by_name = dict(zip(FEATURE_NAMES, extract_features(Ware.JADE, FLAT, [], 1, market)))
        self.assertEqual(by_name["total_black_market"], 30.0)


class TestFitRidge(unittest.TestCase):
    def test_recovers_a_known_linear_relationship(self):
        # y = 3 + 2x, fitted with a bias and one feature.
        features = [[1.0, float(x)] for x in range(30)]
        targets = [3.0 + 2.0 * x for x in range(30)]
        bias, slope = fit_ridge(features, targets, alpha=1e-9)
        self.assertAlmostEqual(bias, 3.0, places=4)
        self.assertAlmostEqual(slope, 2.0, places=4)

    def test_penalty_shrinks_the_slope_but_not_the_bias(self):
        features = [[1.0, float(x)] for x in range(30)]
        targets = [3.0 + 2.0 * x for x in range(30)]
        _, loose = fit_ridge(features, targets, alpha=1e-9)
        _, tight = fit_ridge(features, targets, alpha=500.0)
        self.assertLess(abs(tight), abs(loose))

    def test_rejects_exactly_collinear_features(self):
        features = [[1.0, float(x), float(x)] for x in range(10)]
        with self.assertRaises(ValueError):
            fit_ridge(features, [float(x) for x in range(10)], alpha=0.0)

    def test_rejects_empty_training_data(self):
        with self.assertRaises(ValueError):
            fit_ridge([], [], alpha=1.0)


class TestTrainTestSplit(unittest.TestCase):
    def test_no_game_appears_on_both_sides(self):
        """The split must be by game, never by row -- every voyage of a
        game shares one label, so a row-wise split would leak it."""
        rows = [_row(f"g{g}", v, FLAT, [], FLAT, FLAT) for g in range(40) for v in range(1, 5)]
        train_rows, test_rows = train_test_split(rows, test_fraction=0.25, seed=1)
        self.assertTrue(train_rows and test_rows)
        overlap = {r["game_id"] for r in train_rows} & {r["game_id"] for r in test_rows}
        self.assertEqual(overlap, set())

    def test_keeps_every_row(self):
        rows = [_row(f"g{g}", v, FLAT, [], FLAT, FLAT) for g in range(20) for v in range(1, 4)]
        train_rows, test_rows = train_test_split(rows, test_fraction=0.3, seed=2)
        self.assertEqual(len(train_rows) + len(test_rows), len(rows))


class TestLoadRows(unittest.TestCase):
    def test_skips_rows_written_before_the_black_market_field(self):
        """Legacy rows must be dropped, not defaulted to zero -- zero would
        teach the model that every ware starts worthless. Real v1 rows carry
        no schema_version key at all, so that's what this writes."""
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp, "rows.jsonl")
            legacy = _row("g0", 1, FLAT, [], FLAT, FLAT)
            del legacy["black_market"]
            del legacy["schema_version"]
            path.write_text(
                "\n".join(json.dumps(r) for r in [legacy, _row("g1", 1, FLAT, [], FLAT, FLAT)]),
                encoding="utf-8",
            )
            loaded = load_rows(path)
            self.assertEqual([r["game_id"] for r in loaded], ["g1"])

    def test_rejects_an_outdated_schema_version_even_with_the_field_present(self):
        """The version gate is the real check -- a row stamped v1 is
        rejected whatever fields it happens to carry, since the meaning of
        its numbers, not just its shape, may have changed."""
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp, "rows.jsonl")
            path.write_text(json.dumps(_row("g0", 1, FLAT, [], FLAT, FLAT, schema_version=1)), encoding="utf-8")
            self.assertEqual(load_rows(path), [])

    def test_missing_file_is_empty_rather_than_an_error(self):
        with tempfile.TemporaryDirectory() as tmp:
            self.assertEqual(load_rows(Path(tmp, "nope.jsonl")), [])


class TestTrainedModelLearnsTheMarketSignal(unittest.TestCase):
    def _synthetic(self, n_games=60):
        """Games where the final level is driven by the current level --
        the relationship the real data shows."""
        rows = []
        for g in range(n_games):
            level = (g % 4) * 5
            market = {**FLAT, "jade": level}
            final = {**FLAT, "jade": min(30, level + 10)}
            rows.append(_row(f"g{g}", 1 + (g % 3), FLAT, ["jade"], market, final))
        return rows

    def test_gives_the_current_level_a_positive_coefficient(self):
        model = train(self._synthetic(), alpha=0.01)
        by_name = dict(zip(FEATURE_NAMES, model.coefficients))
        self.assertGreater(by_name["black_market"], 0.0)

    def test_predicts_a_higher_finish_for_a_higher_current_level(self):
        model = train(self._synthetic(), alpha=0.01)
        low = model.predict(Ware.JADE, FLAT, ["jade"], 1, {**FLAT, "jade": 0})
        high = model.predict(Ware.JADE, FLAT, ["jade"], 1, {**FLAT, "jade": 20})
        self.assertGreater(high, low)


class TestEvaluate(unittest.TestCase):
    def test_reports_counts_and_the_comparison_baselines(self):
        rows = [
            _row(f"g{g}", 1, FLAT, ["jade"], {**FLAT, "jade": 10}, {**FLAT, "jade": 30})
            for g in range(12)
        ]
        result = evaluate(default_model(), rows)
        self.assertEqual(result.n_games, 12)
        self.assertEqual(result.n_examples, 12 * len(list(Ware)))
        for accuracy in (result.top_pick_accuracy, result.favored_pick_accuracy, result.random_pick_accuracy):
            self.assertGreaterEqual(accuracy, 0.0)
            self.assertLessEqual(accuracy, 1.0)


class TestSerialization(unittest.TestCase):
    def test_round_trips_through_a_dict(self):
        model = default_model()
        restored = ShareValueModel.from_dict(json.loads(json.dumps(model.to_dict())))
        self.assertEqual(restored.coefficients, model.coefficients)
        self.assertEqual(restored.feature_names, model.feature_names)

    def test_shipped_defaults_line_up_with_the_feature_list(self):
        self.assertEqual(len(DEFAULT_COEFFICIENTS), len(FEATURE_NAMES))


class TestFavoredWaresFromSetup(unittest.TestCase):
    def test_takes_the_two_most_advanced_loaded_wares(self):
        loaded = [Ware.GINSENG, Ware.NUTMEG, Ware.SILK]
        positions = {Ware.GINSENG: 1, Ware.NUTMEG: 5, Ware.SILK: 3}
        self.assertEqual(favored_wares_from_setup(loaded, positions), ["nutmeg", "silk"])


class TestBestShareToBuy(unittest.TestCase):
    def test_skips_wares_with_no_shares_left(self):
        state = GameState.new_default_game(["A", "B", "C", "D"])
        for player in state.players:
            player.shares = []
        # Corner every ware but jade, so only jade can be bought.
        from manilla.engine.models import SHARES_PER_WARE, Share

        for ware in Ware:
            if ware is not Ware.JADE:
                state.players[0].shares.extend(Share(ware=ware) for _ in range(SHARES_PER_WARE))

        choice = best_share_to_buy(default_model(), state, ["jade"])
        self.assertIn(choice, (Ware.JADE, None))
        if choice is not None:
            self.assertEqual(choice, Ware.JADE)

    def test_none_when_every_ware_is_sold_out(self):
        state = GameState.new_default_game(["A", "B", "C", "D"])
        from manilla.engine.models import SHARES_PER_WARE, Share

        for player in state.players:
            player.shares = []
        for ware in Ware:
            state.players[0].shares.extend(Share(ware=ware) for _ in range(SHARES_PER_WARE))
        self.assertIsNone(best_share_to_buy(default_model(), state, []))


if __name__ == "__main__":
    unittest.main()
