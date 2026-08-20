import os
import sys
from fractions import Fraction
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from manilla.engine.models import SEA_ROUTE_LENGTH, Ware
from manilla.engine.probability import (
    movement_distribution,
    movement_distribution_by_ware,
    position_outcomes,
)


class TestMovementDistribution(unittest.TestCase):
    def test_zero_rounds_is_certain_zero(self):
        self.assertEqual(movement_distribution(0), {0: Fraction(1)})

    def test_one_round_is_uniform_over_die_faces(self):
        dist = movement_distribution(1)
        self.assertEqual(set(dist), {1, 2, 3, 4, 5, 6})
        for total in range(1, 7):
            self.assertEqual(dist[total], Fraction(1, 6))
        self.assertEqual(sum(dist.values()), 1)

    def test_two_rounds_matches_classic_2d6_counts(self):
        dist = movement_distribution(2)
        expected_counts = {2: 1, 3: 2, 4: 3, 5: 4, 6: 5, 7: 6, 8: 5, 9: 4, 10: 3, 11: 2, 12: 1}
        for total, count in expected_counts.items():
            self.assertEqual(dist[total], Fraction(count, 36))
        self.assertEqual(sum(dist.values()), 1)

    def test_three_rounds_sums_to_one_and_spans_3_to_18(self):
        dist = movement_distribution(3)
        self.assertEqual(sum(dist.values()), 1)
        self.assertEqual(min(dist), 3)
        self.assertEqual(max(dist), 18)

    def test_negative_rounds_rejected(self):
        with self.assertRaises(ValueError):
            movement_distribution(-1)


class TestMovementDistributionByWare(unittest.TestCase):
    def test_every_ware_currently_shares_the_same_distribution(self):
        by_ware = movement_distribution_by_ware(2)
        self.assertEqual(set(by_ware), set(Ware))
        base = movement_distribution(2)
        for ware in Ware:
            self.assertEqual(by_ware[ware], base)

    def test_accepts_a_restricted_ware_subset(self):
        by_ware = movement_distribution_by_ware(1, wares=[Ware.JADE])
        self.assertEqual(set(by_ware), {Ware.JADE})


class TestPositionOutcomes(unittest.TestCase):
    def test_already_on_13_with_no_rounds_left_is_caught(self):
        outcomes = position_outcomes(SEA_ROUTE_LENGTH, 0)
        self.assertEqual(outcomes["caught_on_13"], 1)
        self.assertEqual(outcomes["arrived"], 0)
        self.assertEqual(outcomes["shipwrecked"], 0)

    def test_short_of_13_with_no_rounds_left_is_shipwrecked(self):
        outcomes = position_outcomes(10, 0)
        self.assertEqual(outcomes["shipwrecked"], 1)
        self.assertEqual(outcomes["arrived"], 0)
        self.assertEqual(outcomes["caught_on_13"], 0)

    def test_one_round_from_7_can_only_reach_or_undershoot_13(self):
        # 7 + (1..6) maxes out at exactly 13 -- never overshoots.
        outcomes = position_outcomes(7, 1)
        self.assertEqual(outcomes["arrived"], 0)
        self.assertEqual(outcomes["caught_on_13"], Fraction(1, 6))  # only a roll of 6
        self.assertEqual(outcomes["shipwrecked"], Fraction(5, 6))

    def test_one_round_from_8_can_overshoot(self):
        # 8 + 5 = 13 (caught), 8 + 6 = 14 (arrived), 8 + (1..4) stays short.
        outcomes = position_outcomes(8, 1)
        self.assertEqual(outcomes["arrived"], Fraction(1, 6))
        self.assertEqual(outcomes["caught_on_13"], Fraction(1, 6))
        self.assertEqual(outcomes["shipwrecked"], Fraction(4, 6))

    def test_outcomes_always_sum_to_one(self):
        for start in range(0, SEA_ROUTE_LENGTH + 1):
            for rounds_remaining in range(0, 4):
                outcomes = position_outcomes(start, rounds_remaining)
                self.assertEqual(sum(outcomes.values()), 1)

    def test_full_voyage_from_a_legal_start_position(self):
        # A punt starting at space 0 with all 3 movement rounds ahead of it.
        outcomes = position_outcomes(0, 3)
        self.assertEqual(sum(outcomes.values()), 1)
        # 3 dice sum to at most 18 and at least 3, so overshooting 13 is
        # possible but far from certain, and undershooting is also possible.
        self.assertGreater(outcomes["arrived"], 0)
        self.assertGreater(outcomes["shipwrecked"], 0)

    def test_more_rounds_remaining_can_only_raise_arrival_probability(self):
        # More chances to roll only ever helps a punt clear space 13.
        p_arrival_1_round = position_outcomes(9, 1)["arrived"]
        p_arrival_2_rounds = position_outcomes(9, 2)["arrived"]
        self.assertGreaterEqual(p_arrival_2_rounds, p_arrival_1_round)

    def test_start_out_of_range_rejected(self):
        with self.assertRaises(ValueError):
            position_outcomes(-1, 1)
        with self.assertRaises(ValueError):
            position_outcomes(SEA_ROUTE_LENGTH + 1, 1)

    def test_negative_rounds_remaining_rejected(self):
        with self.assertRaises(ValueError):
            position_outcomes(0, -1)


if __name__ == "__main__":
    unittest.main()
