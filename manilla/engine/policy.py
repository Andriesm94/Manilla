"""The top-level "pick the best action" policy for a single accomplice
placement: given a live `GameState`, enumerate every currently legal
placement (ware punt, port/shipyard, pirate, pilot, insurance) and choose
whichever scores highest, per `manilla.engine.wealth` /
`manilla.engine.harbor_master`'s REV machinery.

Scoring is REV (`action_impact`'s `total_rev_after`) wherever an action has
a rival effect worth weighing -- ware punt, port/shipyard, and pirate slots
all either split a payout with or endanger someone else's holdings. Where
an action has no rival effect at all, it's scored by plain personal EV
instead: a pilot slot by its best-future-move lookahead value
(`pilot_slot_value`) net of its own price, insurance by `insurance_ev`
(a private financial position, not a competed-for resource -- nobody else's
wealth moves because you took it).

This module deliberately returns a *description* of the chosen action
(`AccompliceChoice`), not a state mutator: driving a live game needs to run
the same payment/turn-advance/UI-refresh code every other placement already
goes through (`BoardSetupApp._place_or_remove_*`), not a bare state
mutation meant for `action_impact`'s own before/after comparison.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

from manilla.engine.beliefs import ShareBeliefs
from manilla.engine.expected_value import Numeric
from manilla.engine.models import GameState, PuntStatus
from manilla.engine.wealth import (
    action_impact,
    apply_dock_slot_placement,
    apply_pirate_placement,
    apply_ware_slot_placement,
    insurance_ev,
    pilot_slot_value,
)


@dataclass
class AccompliceChoice:
    """One legal accomplice placement, tagged for a caller to dispatch to
    its own UI/game-logic action -- see `choose_accomplice_action`."""

    kind: str  # "ware" | "dock" | "pirate" | "pilot" | "insurance"
    punt_id: Optional[int] = None
    dock: Optional[str] = None  # "port" | "shipyard"
    pirate_role: Optional[str] = None  # "captain" | "second"
    pilot_size: Optional[str] = None  # "small" | "large"


def choose_accomplice_action(
    state: GameState,
    beliefs: ShareBeliefs,
    my_id: str,
    p_safe_if_caught: Optional[Numeric] = None,
) -> Optional[AccompliceChoice]:
    """The best currently-legal accomplice placement for `my_id`, by REV
    (or plain EV where nothing competes with a rival) -- `None` if nothing
    is available to place on at all (every ware-punt slot, port/shipyard
    slot, both pirate roles, both pilot sizes, and insurance are all taken
    or otherwise unavailable).
    """
    candidates: List[Tuple[Numeric, AccompliceChoice]] = []

    for punt in state.punts:
        if punt.ware is None or punt.status != PuntStatus.ON_ROUTE:
            continue
        if any(s.occupant is None for s in punt.ware_slots):
            mutator = apply_ware_slot_placement(punt.id, my_id)
            score = action_impact(state, beliefs, my_id, mutator, p_safe_if_caught).total_rev_after
            candidates.append((score, AccompliceChoice(kind="ware", punt_id=punt.id)))

    for dock_name, dock in (("port", state.port), ("shipyard", state.shipyard)):
        if any(s.occupant is None for s in dock.slots.values()):
            mutator = apply_dock_slot_placement(dock_name, my_id)
            score = action_impact(state, beliefs, my_id, mutator, p_safe_if_caught).total_rev_after
            candidates.append((score, AccompliceChoice(kind="dock", dock=dock_name)))

    pb = state.pirate_boat
    if pb.captain.occupant is None:
        pirate_role: Optional[str] = "captain"
    elif pb.second.occupant is None:
        pirate_role = "second"
    else:
        pirate_role = None
    if pirate_role is not None:
        mutator = apply_pirate_placement(pirate_role, my_id)
        score = action_impact(state, beliefs, my_id, mutator, p_safe_if_caught).total_rev_after
        candidates.append((score, AccompliceChoice(kind="pirate", pirate_role=pirate_role)))

    for size, slot in (("small", state.pilot_island.small), ("large", state.pilot_island.large)):
        if slot.occupant is None:
            score = pilot_slot_value(state, beliefs, my_id, size, p_safe_if_caught) - slot.price
            candidates.append((score, AccompliceChoice(kind="pilot", pilot_size=size)))

    if state.insurance.occupant is None:
        score = insurance_ev(state)
        candidates.append((score, AccompliceChoice(kind="insurance")))

    if not candidates:
        return None
    return max(candidates, key=lambda pair: pair[0])[1]
