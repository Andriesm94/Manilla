"""Headless (Tkinter-free) game engine for running full bot-vs-bot games.

Reimplements the same turn-order / auction / dice / profit-distribution
rules that `manilla.ui.board_setup.BoardSetupApp` drives through Tkinter
dialogs and `self.after()` scheduling, but as a synchronous, dialog-free
loop over a `GameState` -- meant for running many games quickly: self-play
training-data generation (Roadmap milestone 5), policy-vs-policy batch
comparison (Roadmap milestone 7), and tests that don't need a real window.

Only supports fully-bot games (every seat `player.is_bot = True`) -- there
is no human input path here at all, matching the "all_bots" branch
`BoardSetupApp` itself takes to skip every messagebox during a simulated
voyage. If you need a human seat, use `manilla.ui.board_setup` instead.

Kept deliberately in lock-step with `BoardSetupApp`'s rules methods (same
constants, same order of operations, same bot-decision probabilities) --
reimplemented rather than imported, since the originals are bound methods
entangled with Tkinter widgets and messagebox calls. If the UI's rules
change, this needs a matching update; see the "architecture debt" note in
the project's Handoff doc.

Every place a player-facing decision needs randomness (random-policy
choices, which unencumbered share to pledge for credit, pirate boarding,
pilot moves, plunder destination) takes an explicit `random.Random`
instance rather than the global `random` module, so a whole game -- or a
batch of them -- can be replayed deterministically from a seed.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional

from manilla.engine.beliefs import infer_beliefs
from manilla.engine.harbor_master import (
    best_punt_setup,
    best_shares_to_buy,
    decide_harbor_master_bid,
    harbor_master_bid_context,
)
from manilla.engine.models import (
    INSURANCE_SHIPYARD_COST,
    MAX_START_SPACE,
    PLUNDER_PAYOUTS,
    PUNT_START_SUM,
    SEA_ROUTE_LENGTH,
    SHARE_LOAN_AMOUNT,
    SHARE_REPAY_AMOUNT,
    AccompliceSlot,
    DockSlots,
    GameState,
    Phase,
    Player,
    Punt,
    PuntStatus,
    Share,
    Ware,
)
from manilla.engine.seat_value import seat_profit_means
from manilla.engine.policy import AccompliceChoice, choose_accomplice_action
from manilla.engine.wealth import best_pilot_move_spec

# ----------------------------------------------------------------------
# Payment / placement primitives (mirrors BoardSetupApp._settle_payment,
# _place_or_remove_accomplice, _place_or_remove_dock_accomplice,
# _place_or_remove_punt_accomplice, _place_or_remove_insurance -- the
# "new placement" branch only, since headless bots never remove one).
# ----------------------------------------------------------------------


def _settle_payment(rng: random.Random, player: Player, amount: int) -> bool:
    while player.cash < amount and player.unencumbered_shares:
        share = rng.choice(player.unencumbered_shares)
        share.encumbered = True
        player.cash += SHARE_LOAN_AMOUNT
    if player.cash < amount:
        return False
    player.cash -= amount
    return True


def _place_accomplice(rng: random.Random, slot: AccompliceSlot, player: Player) -> None:
    if slot.occupant is not None:
        return
    slot.occupant = player.id
    _settle_payment(rng, player, slot.price)


def _place_dock_accomplice(rng: random.Random, dock: DockSlots, player: Player) -> None:
    for key in ["A", "B", "C"]:
        target = dock.slots[key]
        if target.occupant is None:
            _place_accomplice(rng, target, player)
            return


def _place_punt_accomplice(rng: random.Random, punt: Punt, player: Player) -> None:
    vacant = [s for s in punt.ware_slots if s.occupant is None]
    if not vacant:
        return
    cheapest = min(vacant, key=lambda s: s.price)
    _place_accomplice(rng, cheapest, player)


def _place_insurance(state: GameState, player: Player) -> None:
    ins = state.insurance
    if ins.occupant is not None:
        return
    ins.occupant = player.id
    player.cash += ins.payment


# ----------------------------------------------------------------------
# Turn-based accomplice placement (mirrors _bot_take_accomplice_turn /
# _rev_take_accomplice_turn).
# ----------------------------------------------------------------------


def _random_actions(state: GameState, rng: random.Random, player: Player) -> List[Callable[[], None]]:
    actions: List[Callable[[], None]] = []

    for punt in state.punts:
        if punt.ware is None or punt.status != PuntStatus.ON_ROUTE:
            continue
        if any(s.occupant is None for s in punt.ware_slots):
            actions.append(lambda p=punt: _place_punt_accomplice(rng, p, player))

    for dock in (state.port, state.shipyard):
        if any(s.occupant is None for s in dock.slots.values()):
            actions.append(lambda d=dock: _place_dock_accomplice(rng, d, player))

    pb = state.pirate_boat
    if pb.captain.occupant is None:
        actions.append(lambda: _place_accomplice(rng, pb.captain, player))
    elif pb.second.occupant is None:
        actions.append(lambda: _place_accomplice(rng, pb.second, player))

    for slot in (state.pilot_island.small, state.pilot_island.large):
        if slot.occupant is None:
            actions.append(lambda s=slot: _place_accomplice(rng, s, player))

    if state.insurance.occupant is None:
        actions.append(lambda: _place_insurance(state, player))

    return actions


def _apply_accomplice_choice(
    state: GameState, rng: random.Random, player: Player, choice: AccompliceChoice
) -> None:
    if choice.kind == "ware":
        punt = next(p for p in state.punts if p.id == choice.punt_id)
        _place_punt_accomplice(rng, punt, player)
    elif choice.kind == "dock":
        dock = state.port if choice.dock == "port" else state.shipyard
        _place_dock_accomplice(rng, dock, player)
    elif choice.kind == "pirate":
        pb = state.pirate_boat
        slot = pb.captain if choice.pirate_role == "captain" else pb.second
        _place_accomplice(rng, slot, player)
    elif choice.kind == "pilot":
        slot = state.pilot_island.small if choice.pilot_size == "small" else state.pilot_island.large
        _place_accomplice(rng, slot, player)
    elif choice.kind == "insurance":
        _place_insurance(state, player)


def _take_accomplice_turn(state: GameState, rng: random.Random, player: Player) -> None:
    if player.policy == "rev":
        beliefs = infer_beliefs(state, player.id)
        choice = choose_accomplice_action(state, beliefs, player.id)
        if choice is None:
            return
        _apply_accomplice_choice(state, rng, player, choice)
        return

    actions = _random_actions(state, rng, player)
    if actions:
        rng.choice(actions)()


# ----------------------------------------------------------------------
# Dock bookkeeping (mirrors _first_available_dock_key, _dock_punt).
# ----------------------------------------------------------------------


def _first_available_dock_key(state: GameState, status: PuntStatus, exclude_punt: Optional[Punt] = None) -> str:
    for key in ["A", "B", "C"]:
        occupant = next(
            (p for p in state.punts if p is not exclude_punt and p.status == status and p.dock_slot == key),
            None,
        )
        if occupant is None:
            return key
    return "A"  # only reachable if all 3 punts are already docked here


def _dock_punt(state: GameState, punt: Punt, status: PuntStatus, key: str) -> None:
    other = next(
        (p for p in state.punts if p is not punt and p.status == status and p.dock_slot == key), None
    )
    if other is not None:
        other.status = PuntStatus.ON_ROUTE
        other.dock_slot = None
        other.position = SEA_ROUTE_LENGTH
    punt.status = status
    punt.dock_slot = key


# ----------------------------------------------------------------------
# Pirates and pilots (mirrors _handle_pirate_boarding / _bot_board_or_skip,
# _show_pilot_dialogs / _bot_small_pilot_move / _bot_large_pilot_move,
# _resolve_plunder_destination). Pirate boarding and the plunder
# destination stay policy-independent random choices, matching the UI.
# Pilot moves do NOT -- a "rev" player uses wealth.best_pilot_move_spec,
# matching a 2026-08-28 fix to BoardSetupApp itself (the REV pilot-
# lookahead machinery in wealth.py existed since milestone 6 but was
# never actually wired into the live pilot-phase decision, only used to
# value *taking* the pilot slot in the first place -- REV bots were
# picking pilot moves at random in both the UI and here, identically).
# ----------------------------------------------------------------------


def _bot_board_or_skip(rng: random.Random, state: GameState, boat_slot: AccompliceSlot, candidates: List[Punt]) -> None:
    player = state.player_by_id(boat_slot.occupant)
    boardable = [p for p in candidates if any(s.occupant is None for s in p.ware_slots)]
    if player is None or not boardable:
        return
    if rng.random() < 0.6:
        punt = rng.choice(boardable)
        ware_slot = next(s for s in punt.ware_slots if s.occupant is None)
        ware_slot.occupant = player.id
        boat_slot.occupant = None  # the pirate's piece physically left the boat


def _handle_pirate_boarding(state: GameState, rng: random.Random) -> None:
    candidates = [
        p
        for p in state.punts
        if p.ware is not None and p.status == PuntStatus.ON_ROUTE and p.position == SEA_ROUTE_LENGTH
    ]
    if not candidates:
        return
    pb = state.pirate_boat
    if pb.captain.occupant:
        _bot_board_or_skip(rng, state, pb.captain, candidates)
    if pb.second.occupant:
        _bot_board_or_skip(rng, state, pb.second, candidates)
    if pb.captain.occupant is None and pb.second.occupant is not None:
        pb.captain.occupant = pb.second.occupant
        pb.second.occupant = None


def _eligible_pilot_punts(state: GameState) -> List[Punt]:
    return [p for p in state.punts if p.ware is not None and p.status == PuntStatus.ON_ROUTE]


def _apply_pilot_move(state: GameState, punt: Punt, delta: int) -> None:
    if punt.status != PuntStatus.ON_ROUTE:
        return
    punt.position = max(0, punt.position + delta)
    if punt.position > SEA_ROUTE_LENGTH:
        key = _first_available_dock_key(state, PuntStatus.IN_PORT, exclude_punt=punt)
        _dock_punt(state, punt, PuntStatus.IN_PORT, key)
        punt.position = SEA_ROUTE_LENGTH


def _bot_small_pilot_move(state: GameState, rng: random.Random, player_id: str) -> None:
    player = state.player_by_id(player_id)
    eligible = _eligible_pilot_punts(state)
    if player is None or not eligible:
        return
    if player.policy == "rev":
        beliefs = infer_beliefs(state, player.id)
        for punt_id, delta in best_pilot_move_spec(state, beliefs, player.id, "small"):
            punt = next(p for p in state.punts if p.id == punt_id)
            _apply_pilot_move(state, punt, delta)
        return
    if rng.random() < 0.5:
        punt = rng.choice(eligible)
        _apply_pilot_move(state, punt, rng.choice([1, -1]))


def _bot_large_pilot_move(state: GameState, rng: random.Random, player_id: str) -> None:
    player = state.player_by_id(player_id)
    eligible = _eligible_pilot_punts(state)
    if player is None or not eligible:
        return
    if player.policy == "rev":
        beliefs = infer_beliefs(state, player.id)
        for punt_id, delta in best_pilot_move_spec(state, beliefs, player.id, "large"):
            punt = next(p for p in state.punts if p.id == punt_id)
            _apply_pilot_move(state, punt, delta)
        return
    roll = rng.random()
    if roll < 0.35:
        return  # skip
    if roll < 0.65 or len(eligible) < 2:
        punt = rng.choice(eligible)
        _apply_pilot_move(state, punt, rng.choice([2, -2]))
    else:
        a, b = rng.sample(eligible, 2)
        _apply_pilot_move(state, a, rng.choice([1, -1]))
        _apply_pilot_move(state, b, rng.choice([1, -1]))


def _run_pilot_phase(state: GameState, rng: random.Random) -> None:
    small_slot = state.pilot_island.small
    large_slot = state.pilot_island.large
    if small_slot.occupant:
        _bot_small_pilot_move(state, rng, small_slot.occupant)
    if large_slot.occupant:
        _bot_large_pilot_move(state, rng, large_slot.occupant)


def _resolve_plunder_destination(state: GameState, rng: random.Random, punt: Punt) -> None:
    for slot in punt.ware_slots:
        slot.occupant = None  # accomplices are returned, empty-handed
    send_to_port = rng.random() < 0.5
    status = PuntStatus.IN_PORT if send_to_port else PuntStatus.IN_SHIPYARD
    key = _first_available_dock_key(state, status, exclude_punt=punt)
    _dock_punt(state, punt, status, key)


# ----------------------------------------------------------------------
# Dice and movement (mirrors _roll_dice_and_move, minus every UI message).
# ----------------------------------------------------------------------


def _roll_dice_and_move(state: GameState, rng: random.Random) -> List[Punt]:
    """One dice roll and its movement/arrival resolution. Handles pirate
    boarding internally when this is the round-2 roll. Returns the punts
    plundered by pirates -- always empty unless this is the third and
    final roll."""
    rolls: Dict[Ware, int] = {}
    for punt in state.punts:
        if punt.ware is None or punt.status != PuntStatus.ON_ROUTE:
            continue
        roll = rng.randint(1, 6)
        rolls[punt.ware] = roll
        punt.position += roll
    state.last_dice = rolls
    state.movement_round_index += 1

    for punt in state.punts:
        if punt.ware is not None and punt.status == PuntStatus.ON_ROUTE and punt.position > SEA_ROUTE_LENGTH:
            key = _first_available_dock_key(state, PuntStatus.IN_PORT, exclude_punt=punt)
            _dock_punt(state, punt, PuntStatus.IN_PORT, key)
            punt.position = SEA_ROUTE_LENGTH

    board_after = state.movement_round_index == 2
    round3_done = state.movement_round_index >= 3

    plundered_punts: List[Punt] = []
    if round3_done:
        for punt in state.punts:
            if punt.ware is None or punt.status != PuntStatus.ON_ROUTE:
                continue
            if punt.position == SEA_ROUTE_LENGTH:
                pb = state.pirate_boat
                if pb.captain.occupant or pb.second.occupant:
                    _resolve_plunder_destination(state, rng, punt)
                    plundered_punts.append(punt)
                else:
                    key = _first_available_dock_key(state, PuntStatus.IN_PORT, exclude_punt=punt)
                    _dock_punt(state, punt, PuntStatus.IN_PORT, key)
            else:
                key = _first_available_dock_key(state, PuntStatus.IN_SHIPYARD, exclude_punt=punt)
                _dock_punt(state, punt, PuntStatus.IN_SHIPYARD, key)

    if board_after:
        _handle_pirate_boarding(state, rng)

    return plundered_punts


# ----------------------------------------------------------------------
# Profit distribution (mirrors _distribute_profits and its four _pay_*
# helpers plus _raise_ware_values_for_arrivals -- minus the summary message).
# ----------------------------------------------------------------------


def _pay_port_shipyard_rewards(state: GameState) -> None:
    for dock, status in ((state.port, PuntStatus.IN_PORT), (state.shipyard, PuntStatus.IN_SHIPYARD)):
        for key, slot in dock.slots.items():
            if slot.occupant is None:
                continue
            punt = next((p for p in state.punts if p.status == status and p.dock_slot == key), None)
            if punt is None:
                continue
            player = state.player_by_id(slot.occupant)
            if player is not None:
                player.cash += slot.payout


def _pay_ware_profits(state: GameState) -> None:
    for punt in state.punts:
        if punt.ware is None or punt.status != PuntStatus.IN_PORT:
            continue
        occupied = [s for s in punt.ware_slots if s.occupant is not None]
        if not occupied:
            continue
        share = PLUNDER_PAYOUTS.get(punt.ware, 0) // len(occupied)
        for slot in occupied:
            player = state.player_by_id(slot.occupant)
            if player is not None:
                player.cash += share


def _pay_insurance_cost(state: GameState, rng: random.Random) -> None:
    ins = state.insurance
    if ins.occupant is None:
        return
    wrecked = sum(1 for p in state.punts if p.status == PuntStatus.IN_SHIPYARD)
    cost = INSURANCE_SHIPYARD_COST.get(wrecked, 0)
    if cost == 0:
        return
    player = state.player_by_id(ins.occupant)
    if player is None:
        return
    _settle_payment(rng, player, cost)


def _pay_pirate_plunder(state: GameState, plundered_punts: List[Punt]) -> None:
    pb = state.pirate_boat
    pirate_ids = [pid for pid in (pb.captain.occupant, pb.second.occupant) if pid]
    if not pirate_ids:
        return
    for punt in plundered_punts:
        payout = PLUNDER_PAYOUTS.get(punt.ware, 0)
        share = payout // len(pirate_ids)
        for pid in pirate_ids:
            player = state.player_by_id(pid)
            if player is not None:
                player.cash += share


def _raise_ware_values_for_arrivals(state: GameState) -> None:
    for punt in state.punts:
        if punt.ware is None or punt.status != PuntStatus.IN_PORT:
            continue
        state.black_market.raise_value(punt.ware)


def _distribute_profits(state: GameState, rng: random.Random, plundered_punts: List[Punt]) -> None:
    _pay_port_shipyard_rewards(state)
    _pay_ware_profits(state)
    _pay_insurance_cost(state, rng)
    _pay_pirate_plunder(state, plundered_punts)
    _raise_ware_values_for_arrivals(state)
    state.phase = Phase.PROFIT_DISTRIBUTION


# ----------------------------------------------------------------------
# Auction (mirrors _show_auction_dialog's bot-only path: bot_take_turn,
# on_bid, on_pass, end_auction).
# ----------------------------------------------------------------------


def _run_auction(state: GameState, rng: random.Random) -> None:
    players = state.players
    if not players:
        return

    # Measured mean earnings of each seat in the rotation, replacing the
    # old per-voyage random first-mover coefficient -- fixed for the whole
    # game so bidding doesn't drift between voyages. See
    # manilla.engine.seat_value.
    seat_means = seat_profit_means(len(players))
    start_player = next((p for p in players if p.is_harbor_master), players[0])
    start_idx = players.index(start_player)
    order = players[start_idx:] + players[:start_idx]
    active = list(order)
    highest_bid = 0
    highest_bidder: Optional[Player] = None
    turn_idx = 0
    rev_bid_context_cache: dict = {}

    while len(active) > 1:
        cp = active[turn_idx % len(active)]
        affordable = cp.cash + SHARE_LOAN_AMOUNT * len(cp.unencumbered_shares)
        bid_amount: Optional[int] = None

        if cp.policy == "rev":
            beliefs = infer_beliefs(state, cp.id)
            if cp.id not in rev_bid_context_cache:
                rev_bid_context_cache[cp.id] = harbor_master_bid_context(state, beliefs, cp.id)
            candidate = decide_harbor_master_bid(
                state,
                beliefs,
                cp.id,
                [p.id for p in order],
                [p.id for p in active],
                highest_bid,
                seat_means,
                precomputed_bid_context=rev_bid_context_cache[cp.id],
            )
            if candidate is not None and candidate <= affordable:
                bid_amount = candidate
        else:
            candidate = rng.randint(1, 20)
            if candidate > highest_bid and candidate <= affordable:
                bid_amount = candidate

        if bid_amount is not None:
            highest_bid = bid_amount
            highest_bidder = cp
            turn_idx = (turn_idx + 1) % len(active)
        else:
            idx = active.index(cp)
            active.pop(idx)
            turn_idx = idx if idx < len(active) else 0

    winner = active[0] if active else None
    if winner is not None and highest_bid > 0:
        for p in state.players:
            p.is_harbor_master = p.id == winner.id
        _settle_payment(rng, winner, highest_bid)
    # else: no bids at all -- the harbor master (if any, from a previous
    # voyage) is left unchanged, exactly matching end_auction.


def _run_buy_share(state: GameState, rng: random.Random, harbor_master: Player) -> None:
    available = [w for w in Ware if state.shares_available(w) > 0]
    if not available:
        return

    if harbor_master.policy == "rev":
        beliefs = infer_beliefs(state, harbor_master.id)
        candidates = best_shares_to_buy(state, beliefs, harbor_master.id)
        if candidates:
            ware = rng.choice(candidates)
            price = state.black_market.share_price(ware)
            if _settle_payment(rng, harbor_master, price):
                harbor_master.shares.append(Share(ware=ware))
        return

    if rng.random() < 0.5:
        ware = rng.choice(available)
        price = state.black_market.share_price(ware)
        if _settle_payment(rng, harbor_master, price):
            harbor_master.shares.append(Share(ware=ware))


def _run_load_and_place(state: GameState, rng: random.Random, harbor_master: Player) -> None:
    if harbor_master.policy == "rev":
        beliefs = infer_beliefs(state, harbor_master.id)
        (loaded, positions), _ = best_punt_setup(state, beliefs, harbor_master.id)
    else:
        wares = list(Ware)
        rng.shuffle(wares)
        loaded = wares[:3]
        positions: Dict[Ware, int] = {}
        for _ in range(200):
            a = rng.randint(0, MAX_START_SPACE)
            b = rng.randint(0, MAX_START_SPACE)
            c = PUNT_START_SUM - a - b
            if 0 <= c <= MAX_START_SPACE:
                positions = {loaded[0]: a, loaded[1]: b, loaded[2]: c}
                break
        else:
            positions = {loaded[0]: 4, loaded[1]: 3, loaded[2]: 2}

    unloaded = next(w for w in Ware if w not in loaded)
    for punt, ware in zip(state.punts, loaded):
        punt.ware = ware
        punt.position = positions[ware]
        punt.status = PuntStatus.ON_ROUTE
        punt.dock_slot = None
        punt.ware_slots = Punt.new(punt.id, ware).ware_slots
    state.unloaded_ware = unloaded
    state.phase = Phase.ACCOMPLICE_ROUND
    state.current_turn_player_id = harbor_master.id


def _start_next_voyage(state: GameState) -> None:
    state.voyage_number += 1
    for punt in state.punts:
        punt.ware = None
        punt.position = 0
        punt.status = PuntStatus.ON_ROUTE
        punt.dock_slot = None
        punt.ware_slots = []
    state.unloaded_ware = None
    state.accomplice_round_index = 0
    state.movement_round_index = 0
    state.last_dice = {}

    for key in ("A", "B", "C"):
        state.port.slots[key].occupant = None
        state.shipyard.slots[key].occupant = None
    state.pirate_boat.captain.occupant = None
    state.pirate_boat.second.occupant = None
    state.pilot_island.small.occupant = None
    state.pilot_island.large.occupant = None
    state.insurance.occupant = None

    state.phase = Phase.AUCTION


def run_voyage(
    state: GameState,
    rng: random.Random,
    on_loaded: Optional[Callable[[GameState, Player], None]] = None,
    on_voyage_end: Optional[Callable[[GameState, Player, Dict[str, int]], None]] = None,
) -> None:
    """Runs one full voyage -- auction through profit distribution --
    exactly matching `BoardSetupApp`'s dialog chain but with no UI at all.
    Every player must be bot-controlled (`is_bot = True`).

    `on_loaded(state, harbor_master)`, if given, fires once wares are
    loaded and punts are positioned (before any accomplice placement) --
    the earliest point the harbor master's punt-positioning choice
    actually exists. `on_voyage_end(state, harbor_master,
    cash_after_setup)`, if given, fires once this voyage's profits are
    fully distributed; `cash_after_setup` is `{player_id: cash}` taken
    *after* both the auction and the harbor master's share purchase have
    been paid for, so `cash_after - cash_after_setup` isolates what the
    accomplices themselves earned this voyage -- placement costs, cargo
    and port/shipyard payouts, plunder, the insurance bonus -- with the
    price of winning the office excluded.

    Snapshotting here rather than at the top of the voyage keeps two
    different questions apart: what a seat position is *worth* in
    accomplice earnings, versus whether the current bidder overpaid for
    it. Mixing them makes a learned bidding model circular, since it
    would be trained on a target its own bids had already moved.
    `_run_load_and_place` costs nothing, so taking the snapshot before
    or after it is numerically identical.

    One caveat this does *not* net out: a player who runs short mid-
    voyage encumbers a share via `_settle_payment`, which raises their
    cash by `SHARE_LOAN_AMOUNT` in exchange for a debt. That shows up
    here as earnings even though it's a loan, so a seat's mean will read
    slightly high wherever encumbering is common.

    Both exist for `manilla.engine.selfplay_data`'s training/analysis
    data collection -- see there for what actually gets recorded."""
    if not state.players or any(not p.is_bot for p in state.players):
        raise ValueError("run_voyage only supports fully-bot games (every player.is_bot must be True)")

    _run_auction(state, rng)
    harbor_master = next((p for p in state.players if p.is_harbor_master), None)
    if harbor_master is None:
        # Nobody bid at all (only possible if every player is flat broke,
        # which never happens from STARTING_CASH) -- nothing to load, so
        # the voyage can't proceed, matching _show_load_and_place_dialog's
        # own no-op in this case.
        return

    _run_buy_share(state, rng, harbor_master)

    # Taken here, not at the top of the voyage: everything the harbor
    # master paid to *get* the office (winning bid, share purchase) is
    # already settled, so the delta measured against this is accomplice
    # earnings alone. See the docstring for why the two are kept apart.
    cash_after_setup = {p.id: p.cash for p in state.players}

    _run_load_and_place(state, rng, harbor_master)
    if on_loaded is not None:
        on_loaded(state, harbor_master)

    while True:
        players = state.players
        start = state.player_by_id(state.current_turn_player_id) or harbor_master
        start_idx = players.index(start)
        order = players[start_idx:] + players[:start_idx]
        for player in order:
            state.current_turn_player_id = player.id
            _take_accomplice_turn(state, rng, player)

        if state.movement_round_index == 2:
            _run_pilot_phase(state, rng)
            plundered = _roll_dice_and_move(state, rng)
            _distribute_profits(state, rng, plundered)
            if on_voyage_end is not None:
                on_voyage_end(state, harbor_master, cash_after_setup)
            return

        _roll_dice_and_move(state, rng)
        state.current_turn_player_id = harbor_master.id


# ----------------------------------------------------------------------
# End-of-game standings (mirrors _compute_fortune).
# ----------------------------------------------------------------------


@dataclass
class PlayerStanding:
    player_id: str
    name: str
    color: str
    cash: int
    unencumbering_cost: int
    shares_value: int
    total_wealth: int
    forfeited_shares: int


def compute_fortune(state: GameState, player: Player) -> PlayerStanding:
    market = state.black_market
    cash = player.cash
    remaining = cash

    by_value_desc = sorted(player.encumbered_shares, key=lambda s: market.values[s.ware], reverse=True)
    paid_off = []
    for share in by_value_desc:
        if remaining < SHARE_REPAY_AMOUNT:
            break
        remaining -= SHARE_REPAY_AMOUNT
        paid_off.append(share)

    unencumbered_cost = cash - remaining
    counted_shares = player.unencumbered_shares + paid_off
    shares_value = sum(market.values[s.ware] for s in counted_shares)
    total = remaining + shares_value
    forfeited = len(player.encumbered_shares) - len(paid_off)
    return PlayerStanding(
        player_id=player.id,
        name=player.name,
        color=player.color,
        cash=cash,
        unencumbering_cost=unencumbered_cost,
        shares_value=shares_value,
        total_wealth=total,
        forfeited_shares=forfeited,
    )


@dataclass
class GameResult:
    state: GameState
    standings: List[PlayerStanding]  # ranked highest total_wealth first
    voyages_played: int


def run_game(
    state: GameState,
    rng: Optional[random.Random] = None,
    max_voyages: int = 200,
    on_loaded: Optional[Callable[[GameState, Player], None]] = None,
    on_voyage_end: Optional[Callable[[GameState, Player, Dict[str, int]], None]] = None,
) -> GameResult:
    """Plays `state` forward, voyage after voyage, until a ware's black-
    market value reaches 30 (`BlackMarket.is_game_over`). `max_voyages`
    is a defensive cap only -- a real game always ends since the market
    only ever rises -- so hitting it means the voyage loop stalled and
    should be treated as a bug, not a slow game. `on_loaded`/
    `on_voyage_end` are passed straight through to every `run_voyage`
    call -- see there."""
    rng = rng or random.Random()
    voyages = 0
    while not state.black_market.is_game_over():
        voyages += 1
        if voyages > max_voyages:
            raise RuntimeError(f"Game did not end within {max_voyages} voyages -- likely a stalled voyage loop.")
        run_voyage(state, rng, on_loaded=on_loaded, on_voyage_end=on_voyage_end)
        if not state.black_market.is_game_over():
            _start_next_voyage(state)

    standings = sorted(
        (compute_fortune(state, p) for p in state.players), key=lambda s: s.total_wealth, reverse=True
    )
    return GameResult(state=state, standings=standings, voyages_played=voyages)


def new_bot_game(
    player_names: List[str], policy: str = "rev", seed: Optional[int] = None
) -> GameState:
    """A fresh, fully-bot `GameState` ready for `run_voyage`/`run_game` --
    every seat `is_bot = True` with the given `policy` ("rev" or
    "random"), `game_setup_confirmed = True` (so nothing waits on the
    new-game dialog)."""
    state = GameState.new_default_game(player_names, seed=seed)
    for player in state.players:
        player.is_bot = True
        player.policy = policy
    state.game_setup_confirmed = True
    return state


def run_self_play_games(
    n: int,
    player_count: int = 4,
    policy: str = "rev",
    seed: Optional[int] = None,
    max_voyages: int = 200,
) -> List[GameResult]:
    """Runs `n` independent full games, every seat bot-controlled under
    `policy` -- the headless self-play environment behind Roadmap
    milestone 5 (regression training data) and milestone 7 (policy-vs-
    policy batch comparison). Deterministic end-to-end from `seed`."""
    rng = random.Random(seed)
    results: List[GameResult] = []
    for _ in range(n):
        names = [f"Player {i + 1}" for i in range(player_count)]
        state = new_bot_game(names, policy=policy, seed=rng.randrange(2**31))
        results.append(run_game(state, rng=random.Random(rng.randrange(2**31)), max_voyages=max_voyages))
    return results
