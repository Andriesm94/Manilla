"""Tests for `manilla.engine.selfplay`, the headless (Tkinter-free) game
engine -- see its module docstring for why it exists and what it does and
doesn't cover relative to `tests/test_rev_policy_integration.py`.

Random-policy games are essentially free (no REV enumeration), so this
file leans on them for broad, cheap coverage (many seeds, all supported
player counts). REV games are genuinely slow (`best_punt_setup` alone is
close to a second per call, several times per voyage) -- REV-specific
tests here are deliberately few and exist to confirm the *policy* behaves
correctly end-to-end, not to duplicate `test_rev_policy_integration.py`'s
job of verifying the real UI wiring.
"""

import random
import unittest

from manilla.engine.models import GameState, Phase, Share, Ware
from manilla.engine.selfplay import (
    compute_fortune,
    new_bot_game,
    run_game,
    run_self_play_games,
    run_voyage,
)


class TestVoyageEndCashSnapshotExcludesTheCostOfTheOffice(unittest.TestCase):
    """`on_voyage_end`'s `cash_after_setup` must be taken *after* the
    auction and the harbor master's share purchase are both paid, so a
    seat's recorded figure is accomplice earnings alone. Mixing the
    price of the office back in would make a bidding model trained on
    it circular -- it'd be fitting a target its own bids had moved."""

    def test_snapshot_equals_cash_at_load_time_not_voyage_start(self):
        for seed in range(6):
            with self.subTest(seed=seed):
                state = new_bot_game([f"P{i}" for i in range(4)], policy="random", seed=seed)
                cash_at_voyage_start = {p.id: p.cash for p in state.players}

                seen = {}

                def on_loaded(s, hm):
                    # _run_load_and_place moves no money, so cash here is
                    # exactly cash the instant setup finished paying out.
                    seen["at_load"] = {p.id: p.cash for p in s.players}
                    seen["hm"] = hm.id

                def on_voyage_end(s, hm, cash_after_setup):
                    seen["snapshot"] = dict(cash_after_setup)

                run_voyage(state, random.Random(seed), on_loaded=on_loaded, on_voyage_end=on_voyage_end)

                self.assertEqual(seen["snapshot"], seen["at_load"])

                # The harbor master really did pay for the office, so the
                # snapshot is strictly below their voyage-start cash --
                # proving the auction/share cost is outside the delta.
                hm = seen["hm"]
                self.assertLess(seen["snapshot"][hm], cash_at_voyage_start[hm])

                # ...and nobody else pays anything during setup, so a
                # losing bidder's snapshot still matches where they began.
                for player_id, started_with in cash_at_voyage_start.items():
                    if player_id != hm:
                        self.assertEqual(seen["snapshot"][player_id], started_with)


class TestRandomPolicySelfPlay(unittest.TestCase):
    def test_full_game_completes_and_money_never_negative(self):
        for player_count in (3, 4, 5):
            with self.subTest(player_count=player_count):
                state = new_bot_game(
                    [f"P{i}" for i in range(player_count)], policy="random", seed=player_count
                )
                result = run_game(state, rng=random.Random(player_count))

                self.assertTrue(state.black_market.is_game_over())
                self.assertEqual(state.phase, Phase.PROFIT_DISTRIBUTION)
                self.assertGreater(result.voyages_played, 0)
                for player in state.players:
                    self.assertGreaterEqual(player.cash, 0)
                self.assertEqual(len(result.standings), player_count)

    def test_standings_are_ranked_highest_wealth_first(self):
        state = new_bot_game(["A", "B", "C", "D"], policy="random", seed=1)
        result = run_game(state, rng=random.Random(1))
        totals = [s.total_wealth for s in result.standings]
        self.assertEqual(totals, sorted(totals, reverse=True))

    def test_deterministic_from_seed(self):
        results = [
            run_self_play_games(1, player_count=4, policy="random", seed=123)[0] for _ in range(2)
        ]
        first, second = results
        self.assertEqual(first.voyages_played, second.voyages_played)
        self.assertEqual(
            [(s.player_id, s.total_wealth) for s in first.standings],
            [(s.player_id, s.total_wealth) for s in second.standings],
        )

    def test_different_seeds_produce_different_games(self):
        a = run_self_play_games(1, player_count=4, policy="random", seed=1)[0]
        b = run_self_play_games(1, player_count=4, policy="random", seed=2)[0]
        # Not a strict guarantee for any two seeds, but overwhelmingly true
        # for two full multi-voyage random games -- a collision here would
        # indicate the seed isn't actually reaching the dice/decisions.
        self.assertNotEqual(
            [(s.player_id, s.total_wealth) for s in a.standings],
            [(s.player_id, s.total_wealth) for s in b.standings],
        )

    def test_run_voyage_rejects_a_non_bot_seat(self):
        state = new_bot_game(["A", "B", "C", "D"], policy="random", seed=1)
        state.players[0].is_bot = False
        with self.assertRaises(ValueError):
            run_voyage(state, random.Random(1))

    def test_max_voyages_guard_raises_instead_of_hanging(self):
        state = new_bot_game(["A", "B", "C", "D"], policy="random", seed=1)
        with self.assertRaises(RuntimeError):
            run_game(state, rng=random.Random(1), max_voyages=0)


class TestSelfPlayBatchRunner(unittest.TestCase):
    def test_returns_one_result_per_game(self):
        results = run_self_play_games(5, player_count=4, policy="random", seed=7)
        self.assertEqual(len(results), 5)
        for result in results:
            self.assertEqual(len(result.standings), 4)
            self.assertTrue(result.state.black_market.is_game_over())


class TestComputeFortune(unittest.TestCase):
    def test_all_shares_unencumbered_counts_at_current_market_value(self):
        state = GameState.new_default_game(["A", "B", "C"])
        player = state.players[0]
        player.cash = 10
        player.shares = [Share(ware=Ware.JADE, encumbered=False)]
        state.black_market.values[Ware.JADE] = 20

        standing = compute_fortune(state, player)

        self.assertEqual(standing.cash, 10)
        self.assertEqual(standing.unencumbering_cost, 0)
        self.assertEqual(standing.shares_value, 20)
        self.assertEqual(standing.total_wealth, 30)
        self.assertEqual(standing.forfeited_shares, 0)

    def test_unencumbers_highest_value_share_first_when_cash_is_short(self):
        # Enough cash (15) to pay off exactly one 15-PESOS unencumbering,
        # with two encumbered shares competing for it -- the pricier one
        # (jade at 20) must win over the cheaper one (ginseng at 5).
        state = GameState.new_default_game(["A", "B", "C"])
        player = state.players[0]
        player.cash = 15
        player.shares = [
            Share(ware=Ware.JADE, encumbered=True),
            Share(ware=Ware.GINSENG, encumbered=True),
        ]
        state.black_market.values[Ware.JADE] = 20
        state.black_market.values[Ware.GINSENG] = 5

        standing = compute_fortune(state, player)

        self.assertEqual(standing.unencumbering_cost, 15)
        self.assertEqual(standing.shares_value, 20)  # only the jade share got freed
        self.assertEqual(standing.total_wealth, 20)  # 0 cash remaining + 20 shares value
        self.assertEqual(standing.forfeited_shares, 1)  # ginseng stays encumbered, worth nothing

    def test_cannot_afford_any_unencumbering_forfeits_all(self):
        state = GameState.new_default_game(["A", "B", "C"])
        player = state.players[0]
        player.cash = 5  # less than SHARE_REPAY_AMOUNT (15)
        player.shares = [Share(ware=Ware.SILK, encumbered=True)]
        state.black_market.values[Ware.SILK] = 30

        standing = compute_fortune(state, player)

        self.assertEqual(standing.unencumbering_cost, 0)
        self.assertEqual(standing.shares_value, 0)
        self.assertEqual(standing.total_wealth, 5)
        self.assertEqual(standing.forfeited_shares, 1)


class TestRevPolicySelfPlay(unittest.TestCase):
    """REV enumeration (`best_punt_setup` and friends) is genuinely
    expensive -- kept to a single small, seeded scenario. Broader REV
    correctness (does the policy pick sensible actions) is already
    covered by `tests/test_harbor_master.py`/`test_policy.py`/
    `test_wealth.py`; this only checks that a full REV-vs-REV game
    actually completes end to end without stalling or going cash-negative.
    """

    def test_rev_bots_complete_a_full_game_without_stalling(self):
        state = new_bot_game(["A", "B", "C", "D"], policy="rev", seed=42)
        result = run_game(state, rng=random.Random(42), max_voyages=40)

        self.assertTrue(state.black_market.is_game_over())
        self.assertEqual(state.phase, Phase.PROFIT_DISTRIBUTION)
        for player in state.players:
            self.assertGreaterEqual(player.cash, 0)
        self.assertEqual(len(result.standings), 4)

    def test_single_rev_voyage_chooses_a_harbor_master_and_loads_three_wares(self):
        state = new_bot_game(["A", "B", "C", "D"], policy="rev", seed=1)
        run_voyage(state, random.Random(1))

        self.assertEqual(state.phase, Phase.PROFIT_DISTRIBUTION)
        self.assertTrue(any(p.is_harbor_master for p in state.players))
        self.assertIsNotNone(state.unloaded_ware)
        self.assertEqual(sum(1 for p in state.punts if p.ware is not None), 3)
        for player in state.players:
            self.assertGreaterEqual(player.cash, 0)


if __name__ == "__main__":
    unittest.main()
