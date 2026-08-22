import os
import sys
from fractions import Fraction
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from manilla.engine.models import GameState, Share, Ware
from manilla.engine.beliefs import (
    ShareSignal,
    assumed_share_count,
    average_secret_share_value,
    infer_beliefs,
    punt_start_signals,
    secret_pool,
    share_value_estimate,
    total_secret_slots,
    unknown_count,
)


def _make_state():
    state = GameState.new_default_game(["Viewer", "P1", "P2", "P3"])
    for p in state.players:
        p.shares = []
    return state


class TestInferBeliefs(unittest.TestCase):
    def test_viewers_own_shares_are_fully_known(self):
        state = _make_state()
        state.players[0].shares = [Share(ware=Ware.SILK), Share(ware=Ware.JADE)]
        beliefs = infer_beliefs(state, "p0")
        self.assertEqual(beliefs.confirmed_count("p0", Ware.SILK), 1)
        self.assertEqual(beliefs.confirmed_count("p0", Ware.JADE), 1)
        self.assertEqual(unknown_count(state, beliefs, "p0"), 0)

    def test_the_readme_example_averages_secret_shares_uniformly(self):
        # 4 nutmeg + 2 ginseng missing across 3 opponents holding 2 each.
        state = _make_state()
        state.players[1].shares = [Share(ware=Ware.NUTMEG), Share(ware=Ware.NUTMEG)]
        state.players[2].shares = [Share(ware=Ware.NUTMEG), Share(ware=Ware.NUTMEG)]
        state.players[3].shares = [Share(ware=Ware.GINSENG), Share(ware=Ware.GINSENG)]

        beliefs = infer_beliefs(state, "p0")
        pool = secret_pool(state, beliefs)
        self.assertEqual(pool[Ware.NUTMEG], 4)
        self.assertEqual(pool[Ware.GINSENG], 2)
        self.assertEqual(total_secret_slots(state, beliefs), 6)

        # Default black-market prices are all floored at 5.
        avg = average_secret_share_value(state, beliefs)
        self.assertEqual(avg, Fraction(4 * 5 + 2 * 5, 6))

        for opponent in ("p1", "p2", "p3"):
            self.assertEqual(share_value_estimate(state, beliefs, opponent), 2 * avg)

    def test_average_weights_by_a_raised_black_market_price(self):
        state = _make_state()
        state.players[1].shares = [Share(ware=Ware.NUTMEG), Share(ware=Ware.NUTMEG)]
        state.players[2].shares = [Share(ware=Ware.NUTMEG), Share(ware=Ware.NUTMEG)]
        state.players[3].shares = [Share(ware=Ware.GINSENG), Share(ware=Ware.GINSENG)]
        state.black_market.raise_value(Ware.NUTMEG)
        state.black_market.raise_value(Ware.NUTMEG)  # nutmeg now worth 10, not floored

        beliefs = infer_beliefs(state, "p0")
        avg = average_secret_share_value(state, beliefs)
        self.assertEqual(avg, Fraction(4 * 10 + 2 * 5, 6))

    def test_purchase_signal_confirms_a_share_and_shrinks_the_secret_pool(self):
        state = _make_state()
        state.players[1].shares = [Share(ware=Ware.NUTMEG), Share(ware=Ware.NUTMEG)]
        state.players[2].shares = [Share(ware=Ware.NUTMEG), Share(ware=Ware.NUTMEG)]
        state.players[3].shares = [Share(ware=Ware.GINSENG), Share(ware=Ware.GINSENG)]
        state.black_market.raise_value(Ware.NUTMEG)
        state.black_market.raise_value(Ware.NUTMEG)  # nutmeg worth 10

        signals = [ShareSignal(player_id="p1", ware=Ware.NUTMEG, source="purchase")]
        beliefs = infer_beliefs(state, "p0", signals)

        self.assertEqual(beliefs.confirmed_count("p1", Ware.NUTMEG), 1)
        self.assertEqual(unknown_count(state, beliefs, "p1"), 1)
        self.assertEqual(total_secret_slots(state, beliefs), 5)
        self.assertEqual(secret_pool(state, beliefs)[Ware.NUTMEG], 3)

        avg = average_secret_share_value(state, beliefs)
        self.assertEqual(avg, Fraction(3 * 10 + 2 * 5, 5))
        # p1's known nutmeg share at full price, plus their one remaining
        # unknown share at the (now recalculated) secret average.
        self.assertEqual(share_value_estimate(state, beliefs, "p1"), 10 + avg)

    def test_signal_cannot_over_attribute_beyond_actual_share_count(self):
        state = _make_state()
        state.players[1].shares = [Share(ware=Ware.NUTMEG), Share(ware=Ware.NUTMEG)]
        signals = [
            ShareSignal(player_id="p1", ware=Ware.NUTMEG, source="purchase"),
            ShareSignal(player_id="p1", ware=Ware.NUTMEG, source="punt_start"),
            ShareSignal(player_id="p1", ware=Ware.GINSENG, source="punt_start"),  # redundant, p1 only has 2 shares
        ]
        beliefs = infer_beliefs(state, "p0", signals)
        self.assertEqual(beliefs.known_total("p1"), 2)
        self.assertEqual(unknown_count(state, beliefs, "p1"), 0)
        self.assertEqual(beliefs.confirmed_count("p1", Ware.GINSENG), 0)

    def test_signal_about_the_viewer_themselves_is_a_no_op(self):
        state = _make_state()
        state.players[0].shares = [Share(ware=Ware.SILK)]
        signals = [ShareSignal(player_id="p0", ware=Ware.JADE, source="purchase")]
        beliefs = infer_beliefs(state, "p0", signals)
        self.assertEqual(beliefs.confirmed_count("p0", Ware.JADE), 0)
        self.assertEqual(beliefs.confirmed_count("p0", Ware.SILK), 1)

    def test_no_secret_shares_gives_a_zero_average(self):
        state = _make_state()
        beliefs = infer_beliefs(state, "p0")
        self.assertEqual(average_secret_share_value(state, beliefs), 0)
        self.assertEqual(total_secret_slots(state, beliefs), 0)


class TestPuntStartSignals(unittest.TestCase):
    def test_a_forward_start_position_produces_a_signal(self):
        signals = punt_start_signals("p1", [(Ware.JADE, 4), (Ware.SILK, 5)])
        self.assertEqual(
            {(s.player_id, s.ware) for s in signals},
            {("p1", Ware.JADE), ("p1", Ware.SILK)},
        )

    def test_a_modest_start_position_produces_no_signal(self):
        signals = punt_start_signals("p1", [(Ware.JADE, 3), (Ware.SILK, 0)])
        self.assertEqual(signals, [])

    def test_feeds_straight_into_infer_beliefs(self):
        state = _make_state()
        state.players[1].shares = [Share(ware=Ware.JADE), Share(ware=Ware.JADE)]
        signals = punt_start_signals("p1", [(Ware.JADE, 4)])
        beliefs = infer_beliefs(state, "p0", signals)
        self.assertEqual(beliefs.confirmed_count("p1", Ware.JADE), 1)
        self.assertEqual(unknown_count(state, beliefs, "p1"), 1)


class TestAssumedShareCount(unittest.TestCase):
    def test_matches_confirmed_count_when_fully_known(self):
        state = _make_state()
        state.players[0].shares = [Share(ware=Ware.SILK)]
        beliefs = infer_beliefs(state, "p0")
        self.assertEqual(assumed_share_count(state, beliefs, "p0", Ware.SILK), 1)
        self.assertEqual(assumed_share_count(state, beliefs, "p0", Ware.JADE), 0)

    def test_splits_unconfirmed_shares_by_pool_composition(self):
        state = _make_state()
        state.players[1].shares = [Share(ware=Ware.NUTMEG), Share(ware=Ware.NUTMEG)]
        state.players[2].shares = [Share(ware=Ware.NUTMEG), Share(ware=Ware.NUTMEG)]
        state.players[3].shares = [Share(ware=Ware.GINSENG), Share(ware=Ware.GINSENG)]
        beliefs = infer_beliefs(state, "p0")

        self.assertEqual(assumed_share_count(state, beliefs, "p1", Ware.NUTMEG), Fraction(4, 3))
        self.assertEqual(assumed_share_count(state, beliefs, "p1", Ware.GINSENG), Fraction(2, 3))
        self.assertEqual(assumed_share_count(state, beliefs, "p1", Ware.SILK), 0)

    def test_sums_to_share_value_estimate_when_weighted_by_price(self):
        state = _make_state()
        state.players[1].shares = [Share(ware=Ware.NUTMEG), Share(ware=Ware.NUTMEG)]
        state.players[2].shares = [Share(ware=Ware.NUTMEG), Share(ware=Ware.NUTMEG)]
        state.players[3].shares = [Share(ware=Ware.GINSENG), Share(ware=Ware.GINSENG)]
        beliefs = infer_beliefs(state, "p0")

        via_breakdown = sum(
            assumed_share_count(state, beliefs, "p1", w) * state.black_market.share_price(w) for w in Ware
        )
        self.assertEqual(via_breakdown, share_value_estimate(state, beliefs, "p1"))

    def test_zero_confirmed_and_zero_secret_pool_gives_zero(self):
        state = _make_state()
        beliefs = infer_beliefs(state, "p0")
        self.assertEqual(assumed_share_count(state, beliefs, "p1", Ware.SILK), 0)


if __name__ == "__main__":
    unittest.main()
