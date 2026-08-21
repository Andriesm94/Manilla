import os
import sys
from fractions import Fraction
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from manilla.engine.models import (
    DEFAULT_PORT_PAYOUTS,
    DEFAULT_PORT_PRICES,
    DEFAULT_SHIPYARD_PAYOUTS,
    DEFAULT_SHIPYARD_PRICES,
    PIRATE_PRICE,
    PLUNDER_PAYOUTS,
    Ware,
)
from manilla.engine.expected_value import (
    dock_fill_distribution,
    dock_slot_ev,
    dock_slot_fill_probability,
    pirate_slot_ev,
    punt_port_probability,
    punt_shipyard_probability,
    ware_slot_ev,
)


class TestPuntProbabilities(unittest.TestCase):
    def test_port_probability_defaults_to_pirate_free(self):
        # start=7, 1 round left: roll of 6 lands exactly on 13 (caught),
        # nothing overshoots -- with the default p_safe_if_caught=1, that
        # counts fully toward "reaches port".
        self.assertEqual(punt_port_probability(7, 1), Fraction(1, 6))

    def test_port_probability_scales_by_pirate_safety_estimate(self):
        half_safe = punt_port_probability(7, 1, p_safe_if_caught=Fraction(1, 2))
        self.assertEqual(half_safe, Fraction(1, 12))

    def test_shipyard_probability_ignores_pirate_safety(self):
        # 8 + (1..4) falls short of 13 regardless of pirates.
        self.assertEqual(punt_shipyard_probability(8, 1), Fraction(4, 6))


class TestWareSlotEV(unittest.TestCase):
    def test_matches_hand_computation_for_a_simple_case(self):
        # start=8, 1 round left, ginseng slot 0 -- the cheapest vacant slot,
        # price 1 (DEFAULT_WARE_SLOT_PRICES lists ginseng as [3, 2, 1], but
        # placement always takes the cheapest available regardless of its
        # position in that list). Payout 18.
        # p_safe = P(arrived) + P(caught_on_13) = 1/6 + 1/6 = 1/3.
        ev = ware_slot_ev(Ware.GINSENG, 0, start=8, rounds_remaining=1, accomplices_on_punt=1)
        self.assertEqual(ev, Fraction(1, 3) * 18 - 1)

    def test_slot_index_follows_fill_order_not_raw_array_order(self):
        # Ginseng's raw price list is [3, 2, 1] (descending); slot_index
        # should walk it cheapest-first (1, 2, 3) to match real placement.
        cheapest = ware_slot_ev(Ware.GINSENG, 0, start=8, rounds_remaining=1, accomplices_on_punt=1)
        middle = ware_slot_ev(Ware.GINSENG, 1, start=8, rounds_remaining=1, accomplices_on_punt=1)
        priciest = ware_slot_ev(Ware.GINSENG, 2, start=8, rounds_remaining=1, accomplices_on_punt=1)
        self.assertEqual(cheapest - middle, 1)  # price 1 vs price 2
        self.assertEqual(middle - priciest, 1)  # price 2 vs price 3

    def test_more_accomplices_split_the_payout_further(self):
        solo = ware_slot_ev(Ware.GINSENG, 0, start=8, rounds_remaining=1, accomplices_on_punt=1)
        shared = ware_slot_ev(Ware.GINSENG, 0, start=8, rounds_remaining=1, accomplices_on_punt=3)
        self.assertGreater(solo, shared)

    def test_pirate_risk_lowers_the_expected_return(self):
        safe = ware_slot_ev(Ware.SILK, 0, start=7, rounds_remaining=1, accomplices_on_punt=1)
        risky = ware_slot_ev(
            Ware.SILK, 0, start=7, rounds_remaining=1, accomplices_on_punt=1, p_safe_if_caught=Fraction(0)
        )
        self.assertLess(risky, safe)

    def test_rejects_out_of_range_slot_index(self):
        with self.assertRaises(ValueError):
            ware_slot_ev(Ware.GINSENG, 3, start=0, rounds_remaining=3, accomplices_on_punt=1)

    def test_rejects_invalid_accomplice_count(self):
        with self.assertRaises(ValueError):
            ware_slot_ev(Ware.GINSENG, 0, start=0, rounds_remaining=3, accomplices_on_punt=0)
        with self.assertRaises(ValueError):
            ware_slot_ev(Ware.GINSENG, 0, start=0, rounds_remaining=3, accomplices_on_punt=4)


class TestPirateSlotEV(unittest.TestCase):
    def test_matches_hand_computation_for_one_punt(self):
        # start=7, 1 round left: only a roll of 6 lands exactly on 13.
        ev = pirate_slot_ev([(Ware.NUTMEG, 7, 1)], pirate_count=1)
        expected = Fraction(1, 6) * PLUNDER_PAYOUTS[Ware.NUTMEG] - PIRATE_PRICE
        self.assertEqual(ev, expected)

    def test_sums_across_every_loaded_punt(self):
        solo = pirate_slot_ev([(Ware.NUTMEG, 7, 1)], pirate_count=1)
        two_punts = pirate_slot_ev([(Ware.NUTMEG, 7, 1), (Ware.SILK, 7, 1)], pirate_count=1)
        self.assertEqual(two_punts, solo + Fraction(1, 6) * PLUNDER_PAYOUTS[Ware.SILK])

    def test_splitting_with_a_second_pirate_halves_the_share(self):
        alone = pirate_slot_ev([(Ware.JADE, 7, 1)], pirate_count=1)
        shared = pirate_slot_ev([(Ware.JADE, 7, 1)], pirate_count=2)
        # Same PIRATE_PRICE cost either way, but half the payout per punt.
        self.assertEqual(alone - shared, Fraction(1, 6) * (PLUNDER_PAYOUTS[Ware.JADE] - PLUNDER_PAYOUTS[Ware.JADE] // 2))

    def test_rejects_invalid_pirate_count(self):
        with self.assertRaises(ValueError):
            pirate_slot_ev([(Ware.JADE, 7, 1)], pirate_count=3)


class TestDockFillDistribution(unittest.TestCase):
    def test_no_punts_is_certain_zero(self):
        self.assertEqual(dock_fill_distribution([]), {0: Fraction(1)})

    def test_two_independent_punts_matches_hand_enumeration(self):
        # p1=1/2, p2=1/3 -> P(0)=1/2*2/3=1/3, P(1)=1/2*2/3+1/2*1/3=1/2, P(2)=1/2*1/3=1/6.
        dist = dock_fill_distribution([Fraction(1, 2), Fraction(1, 3)])
        self.assertEqual(dist[0], Fraction(1, 3))
        self.assertEqual(dist[1], Fraction(1, 2))
        self.assertEqual(dist[2], Fraction(1, 6))
        self.assertEqual(sum(dist.values()), 1)

    def test_certain_arrivals_stack_deterministically(self):
        dist = dock_fill_distribution([1, 1, 1])
        self.assertEqual(dist[3], Fraction(1))
        self.assertEqual(sum(v for k, v in dist.items() if k != 3), 0)


class TestDockSlotFillProbability(unittest.TestCase):
    def test_slot_a_only_needs_one_arrival(self):
        probs = [Fraction(1, 2), Fraction(1, 2), Fraction(1, 2)]
        self.assertEqual(dock_slot_fill_probability(probs, "A"), 1 - Fraction(1, 8))

    def test_slot_c_needs_all_three(self):
        probs = [Fraction(1, 2), Fraction(1, 2), Fraction(1, 2)]
        self.assertEqual(dock_slot_fill_probability(probs, "C"), Fraction(1, 8))

    def test_rejects_unknown_key(self):
        with self.assertRaises(ValueError):
            dock_slot_fill_probability([Fraction(1, 2)], "D")


class TestDockSlotEV(unittest.TestCase):
    def test_port_c_is_expensive_to_earn_with_few_punts(self):
        # Only one punt in play: slot C can never fill, so its EV is just
        # the sunk price, negative.
        ev = dock_slot_ev("port", "C", [Fraction(1, 2)])
        self.assertEqual(ev, -DEFAULT_PORT_PRICES["C"])

    def test_port_a_is_profitable_when_arrival_is_likely(self):
        ev = dock_slot_ev("port", "A", [Fraction(9, 10), Fraction(9, 10), Fraction(9, 10)])
        p_filled = 1 - Fraction(1, 10) ** 3
        self.assertEqual(ev, p_filled * DEFAULT_PORT_PAYOUTS["A"] - DEFAULT_PORT_PRICES["A"])

    def test_shipyard_uses_shipyard_tables(self):
        ev = dock_slot_ev("shipyard", "B", [Fraction(1, 2), Fraction(1, 2)])
        p_filled = dock_slot_fill_probability([Fraction(1, 2), Fraction(1, 2)], "B")
        self.assertEqual(ev, p_filled * DEFAULT_SHIPYARD_PAYOUTS["B"] - DEFAULT_SHIPYARD_PRICES["B"])

    def test_rejects_unknown_dock(self):
        with self.assertRaises(ValueError):
            dock_slot_ev("harbor", "A", [Fraction(1, 2)])


if __name__ == "__main__":
    unittest.main()
