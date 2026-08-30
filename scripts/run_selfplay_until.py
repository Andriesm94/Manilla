"""Run self-play games until a wall-clock deadline, appending to data/.

Built for long unattended runs, so the failure modes that matter are the
slow ones:

* **One game per `record_self_play_games` call.** Each call opens and
  closes the JSONL files, so every finished game is flushed to disk. A
  single long-lived call would buffer results and lose whatever hadn't
  been written if the run were killed or the machine rebooted.
* **A crash in one game doesn't end the run.** Exceptions are logged with
  a traceback and the loop continues; only a long streak of consecutive
  failures aborts, so a systematic breakage doesn't spin for hours.
* **The deadline is checked before starting a game, never mid-game.** A
  REV game takes ~25-60s, so the run can overshoot by about that much
  rather than leaving a half-recorded game behind.

Usage:
    python scripts/run_selfplay_until.py --until 2026-08-30T11:00
"""

from __future__ import annotations

import argparse
import os
import sys
import time
import traceback
from datetime import datetime

# Anchor to the repo so this works regardless of where it's launched from
# (a detached process does not inherit an interactive shell's cwd).
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(REPO)
sys.path.insert(0, REPO)

from manilla.engine.selfplay_data import record_self_play_games  # noqa: E402

MAX_CONSECUTIVE_FAILURES = 10


def log(message: str) -> None:
    print(f"[{datetime.now():%Y-%m-%d %H:%M:%S}] {message}", flush=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--until", required=True, help="local deadline, e.g. 2026-08-30T11:00")
    parser.add_argument("--players", type=int, default=4)
    parser.add_argument("--policy", default="rev", choices=["rev", "random"])
    parser.add_argument("--out-dir", default="data")
    parser.add_argument(
        "--seed-base",
        type=int,
        default=None,
        help="first game's seed; defaults to the current unix time so separate "
        "runs don't regenerate identical games",
    )
    args = parser.parse_args()

    deadline = datetime.fromisoformat(args.until)
    seed_base = args.seed_base if args.seed_base is not None else int(time.time())

    started = time.time()
    games = voyages = failures = consecutive_failures = 0

    log(
        f"start: {args.players} {args.policy.upper()} computers until {deadline:%Y-%m-%d %H:%M} "
        f"| out={args.out_dir} | seed_base={seed_base}"
    )

    while datetime.now() < deadline:
        t0 = time.time()
        try:
            results = record_self_play_games(
                1,
                player_count=args.players,
                policy=args.policy,
                seed=seed_base + games,
                out_dir=args.out_dir,
            )
        except Exception:
            failures += 1
            consecutive_failures += 1
            log(f"game {games + 1} FAILED (failure {failures}):\n{traceback.format_exc()}")
            if consecutive_failures >= MAX_CONSECUTIVE_FAILURES:
                log(f"aborting: {consecutive_failures} consecutive failures")
                return 1
            continue

        consecutive_failures = 0
        games += 1
        result = results[0]
        voyages += result.voyages_played
        winner = result.standings[0]
        remaining = (deadline - datetime.now()).total_seconds()
        log(
            f"game {games}: {result.voyages_played} voyages, {time.time() - t0:5.1f}s "
            f"| winner {winner.name} wealth {winner.total_wealth} "
            f"| totals {voyages} voyages | {remaining / 3600:.1f}h left"
        )

    elapsed = time.time() - started
    log(
        f"done: {games} games, {voyages} voyages, {failures} failures "
        f"in {elapsed / 3600:.2f}h ({elapsed / max(games, 1):.1f}s per game)"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
