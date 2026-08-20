"""Exact (non-simulated) probability math for Manilla's dice-driven movement.

Pure functions over die-roll sums via DP/combinatorics -- no Tkinter import,
no GameState mutation, safe to unit test in isolation. Every ware currently
rolls the same uniform 1-6 die each movement round (see
`manilla.ui.board_setup._roll_dice_and_move`), so these functions are
ware-agnostic; a caller passes in a punt's own position and how many
movement rounds remain.

This module only computes the dice-driven outcome of *whether a punt ends up
sitting on space 13* -- it does not decide plunder vs. safe arrival there,
since boarding a punt caught on 13 is a player/bot decision
(`BoardSetupApp._handle_pirate_boarding`), not a random event.
"""

from __future__ import annotations

from fractions import Fraction
from typing import Dict, Iterable, Optional

from manilla.engine.models import SEA_ROUTE_LENGTH, Ware

DIE_FACES = (1, 2, 3, 4, 5, 6)
DIE_PROB = Fraction(1, len(DIE_FACES))


def movement_distribution(rounds: int) -> Dict[int, Fraction]:
    """Exact distribution of total pips rolled over `rounds` independent
    movement rounds (one 1-6 die roll per round). Returns {total: probability},
    probabilities summing to exactly 1."""
    if rounds < 0:
        raise ValueError("rounds must be >= 0")

    dist: Dict[int, Fraction] = {0: Fraction(1)}
    for _ in range(rounds):
        next_dist: Dict[int, Fraction] = {}
        for total, prob in dist.items():
            for face in DIE_FACES:
                next_dist[total + face] = next_dist.get(total + face, Fraction(0)) + prob * DIE_PROB
        dist = next_dist
    return dist


def movement_distribution_by_ware(
    rounds: int, wares: Optional[Iterable[Ware]] = None
) -> Dict[Ware, Dict[int, Fraction]]:
    """Same distribution as `movement_distribution`, keyed per ware. All
    wares currently share one uniform 1-6 die, so every entry is identical
    today -- this only exists so callers iterating loaded punts by ware
    don't need a separate ware-agnostic special case, and so a future
    house rule giving a ware its own die shape has one place to change."""
    dist = movement_distribution(rounds)
    return {ware: dict(dist) for ware in (wares or list(Ware))}


def position_outcomes(start: int, rounds_remaining: int) -> Dict[str, Fraction]:
    """Resolve a punt sitting at `start` through up to `rounds_remaining`
    more movement rounds, applying the mid-voyage overshoot-arrival rule: a
    punt that passes space SEA_ROUTE_LENGTH on ANY round docks immediately
    and rolls no further rounds that voyage.

    Returns exact probabilities for the three end-of-voyage outcomes,
    summing to exactly 1:
      - "arrived": docked in port because it overshot space
        SEA_ROUTE_LENGTH on some round (early or on the last one available)
      - "caught_on_13": still at sea, sitting exactly on space
        SEA_ROUTE_LENGTH once rounds_remaining is exhausted -- vulnerable to
        pirate plunder, but whether it's actually plundered is a separate,
        non-dice decision (see module docstring)
      - "shipwrecked": still at sea, short of space SEA_ROUTE_LENGTH, once
        rounds_remaining is exhausted
    """
    if not 0 <= start <= SEA_ROUTE_LENGTH:
        raise ValueError(f"start must be between 0 and {SEA_ROUTE_LENGTH}")
    if rounds_remaining < 0:
        raise ValueError("rounds_remaining must be >= 0")

    at_sea: Dict[int, Fraction] = {start: Fraction(1)}
    arrived = Fraction(0)
    for _ in range(rounds_remaining):
        next_at_sea: Dict[int, Fraction] = {}
        for pos, prob in at_sea.items():
            for face in DIE_FACES:
                new_pos = pos + face
                p = prob * DIE_PROB
                if new_pos > SEA_ROUTE_LENGTH:
                    arrived += p
                else:
                    next_at_sea[new_pos] = next_at_sea.get(new_pos, Fraction(0)) + p
        at_sea = next_at_sea

    caught_on_13 = at_sea.get(SEA_ROUTE_LENGTH, Fraction(0))
    shipwrecked = sum((p for pos, p in at_sea.items() if pos < SEA_ROUTE_LENGTH), Fraction(0))
    return {"arrived": arrived, "caught_on_13": caught_on_13, "shipwrecked": shipwrecked}
