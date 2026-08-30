"""Tests for `manilla.engine.seat_value` -- the measured per-seat profit
table that replaced the hand-picked random first-mover coefficient."""

import json
import os
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from manilla.engine.seat_value import (
    MEASURED_SEAT_PROFIT,
    measure_seat_profit_means,
    seat_advantage,
    seat_profit_means,
)


class TestSeatProfitMeans(unittest.TestCase):
    def test_returns_the_measured_table_for_a_measured_player_count(self):
        self.assertEqual(seat_profit_means(4), MEASURED_SEAT_PROFIT[4])

    def test_one_entry_per_seat_for_every_supported_player_count(self):
        for count in (3, 4, 5):
            with self.subTest(player_count=count):
                self.assertEqual(len(seat_profit_means(count)), count)

    def test_harbor_master_leads_every_other_seat(self):
        # The whole point of the table -- if this stops holding, bidding
        # would start valuing the office at zero or negative.
        for count in (3, 4, 5):
            with self.subTest(player_count=count):
                means = seat_profit_means(count)
                self.assertTrue(all(means[0] > m for m in means[1:]))

    def test_unmeasured_counts_are_flat_behind_the_harbor_master(self):
        """Unmeasured player counts must not invent a per-seat slope --
        the measurement says the gradient behind the harbor master is
        essentially flat, so extrapolating one would be fiction."""
        means = seat_profit_means(5)
        self.assertEqual(len(set(means[1:])), 1)


class TestSeatAdvantage(unittest.TestCase):
    def test_zero_for_the_same_seat(self):
        self.assertEqual(seat_advantage((20.0, 6.0, 4.0), 0), 0.0)

    def test_is_the_gap_between_the_harbor_master_and_that_seat(self):
        self.assertEqual(seat_advantage((20.0, 6.0, 4.0), 1), 14.0)
        self.assertEqual(seat_advantage((20.0, 6.0, 4.0), 2), 16.0)

    def test_follows_the_table_rather_than_the_distance(self):
        # Seat 3 is worth more than seat 2 in the real measurement, so the
        # advantage over seat 3 is *smaller* -- a distance-scaled formula
        # would get the ordering backwards.
        seats = (20.0, 6.0, 4.0, 5.0)
        self.assertGreater(seat_advantage(seats, 2), seat_advantage(seats, 3))

    def test_clamps_past_the_end_of_the_table(self):
        seats = (20.0, 6.0, 4.0)
        self.assertEqual(seat_advantage(seats, 9), seat_advantage(seats, 2))

    def test_empty_table_is_zero_rather_than_an_error(self):
        self.assertEqual(seat_advantage((), 2), 0.0)


class TestMeasureSeatProfitMeans(unittest.TestCase):
    def _write(self, path, rows):
        path.write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")

    def test_averages_each_offset_across_rows(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp, "profit.jsonl")
            self._write(
                path,
                [
                    {"policy": "rev", "player_count": 2, "pesos_by_seat_offset": {"0": 10, "1": 2}},
                    {"policy": "rev", "player_count": 2, "pesos_by_seat_offset": {"0": 20, "1": 4}},
                ],
            )
            self.assertEqual(measure_seat_profit_means(path, player_count=2), (15.0, 3.0))

    def test_ignores_rows_from_another_policy_or_player_count(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp, "profit.jsonl")
            self._write(
                path,
                [
                    {"policy": "rev", "player_count": 2, "pesos_by_seat_offset": {"0": 10, "1": 2}},
                    {"policy": "random", "player_count": 2, "pesos_by_seat_offset": {"0": 999, "1": 999}},
                    {"policy": "rev", "player_count": 4, "pesos_by_seat_offset": {"0": 999, "1": 999}},
                ],
            )
            self.assertEqual(measure_seat_profit_means(path, player_count=2), (10.0, 2.0))

    def test_none_when_the_file_is_missing_or_has_no_matching_rows(self):
        with tempfile.TemporaryDirectory() as tmp:
            self.assertIsNone(measure_seat_profit_means(Path(tmp, "nope.jsonl")))
            path = Path(tmp, "profit.jsonl")
            self._write(path, [{"policy": "random", "player_count": 4, "pesos_by_seat_offset": {"0": 1}}])
            self.assertIsNone(measure_seat_profit_means(path, policy="rev", player_count=4))


if __name__ == "__main__":
    unittest.main()
