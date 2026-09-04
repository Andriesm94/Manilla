"""Data model for a snapshot of a Manilla game.

This module only describes the *shape* of a game state (board layout,
player holdings, prices) -- it contains no turn-order or rules-resolution
logic. That belongs to a future `manilla.engine.rules` module.

Accomplice slot prices/payouts below reflect the user's stated rules for
this project's variant of the game. They are plain data on each slot and
fully editable from the board-setup UI.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
import json
import random
from typing import Dict, List, Optional


class Ware(Enum):
    NUTMEG = "nutmeg"
    SILK = "silk"
    GINSENG = "ginseng"
    JADE = "jade"


class Phase(Enum):
    AUCTION = "auction"
    LOAD_GOODS = "load_goods"
    PLACE_PUNTS = "place_punts"
    ACCOMPLICE_ROUND = "accomplice_round"
    MOVEMENT_ROUND = "movement_round"
    PILOT_PHASE = "pilot_phase"
    PROFIT_DISTRIBUTION = "profit_distribution"
    WARE_RISE = "ware_rise"


class PuntStatus(Enum):
    ON_ROUTE = "on_route"
    IN_PORT = "in_port"
    IN_SHIPYARD = "in_shipyard"
    CAPTURED = "captured"  # plundered by pirates, removed from board


WARE_SLOT_COUNT: Dict[Ware, int] = {
    Ware.NUTMEG: 3,
    Ware.SILK: 3,
    Ware.GINSENG: 3,
    Ware.JADE: 4,
}

# Ascending slot prices per ware punt (paid when an accomplice is placed).
DEFAULT_WARE_SLOT_PRICES: Dict[Ware, List[int]] = {
    Ware.GINSENG: [3, 2, 1],
    Ware.NUTMEG: [4, 3, 2],
    Ware.SILK: [5, 4, 3],
    Ware.JADE: [5, 5, 4, 3],
}

# Payouts for port / shipyard slots (A/B/C) -- these fill in that order (see
# GameState mandatory-highest-available placement rule), so A is the lowest
# reward and C is the highest.
DEFAULT_PORT_PAYOUTS: Dict[str, int] = {"A": 6, "B": 8, "C": 15}
DEFAULT_SHIPYARD_PAYOUTS: Dict[str, int] = {"A": 6, "B": 8, "C": 15}

# Cost to place an accomplice on a port / shipyard space -- separate from the
# payout above, which is only earned if a punt lands there.
DEFAULT_PORT_PRICES: Dict[str, int] = {"A": 4, "B": 3, "C": 2}
DEFAULT_SHIPYARD_PRICES: Dict[str, int] = {"A": 4, "B": 3, "C": 2}

DEFAULT_PILOT_PRICES = {"small": 2, "large": 5}
PIRATE_PRICE = 5
INSURANCE_PAYMENT = 10

# Plunder payout per ware when a punt is caught on space 13 after the third
# movement round, split evenly between however many pirates are present.
PLUNDER_PAYOUTS: Dict[Ware, int] = {
    Ware.NUTMEG: 24,
    Ware.GINSENG: 18,
    Ware.JADE: 36,
    Ware.SILK: 30,
}
SHARE_LOAN_AMOUNT = 12
SHARE_REPAY_AMOUNT = 15

# What the insurance-holder pays for repairs, by how many punts wrecked in
# the shipyard this voyage.
INSURANCE_SHIPYARD_COST: Dict[int, int] = {1: 6, 2: 14, 3: 29}
STARTING_CASH = 30
STARTING_SHARES_PER_PLAYER = 2
SHARES_PER_WARE = 5
# The opening deal is made from a shuffled pile of 3 of each ware, not from
# all 20 shares (rules p.2, "Preparation"): whatever isn't dealt joins the
# stock beside the board. So no ware can start with more than 3 in hands.
DEAL_POOL_PER_WARE = 3
SEA_ROUTE_LENGTH = 13  # spaces 0-13
PUNT_START_SUM = 9
MAX_START_SPACE = 5
GAME_END_VALUE = 30
BLACK_MARKET_LEVELS = [0, 5, 10, 20, 30]


@dataclass
class Share:
    """One share certificate. `ware` is None for a share the game state
    knows a player holds but whose identity was never recorded -- a human
    who hand-dealt without typing their hand in. The count is public either
    way (rules p.8: an encumbered share is set aside *face-down*, so even
    taking credit doesn't reveal it); the ware only has to be named at game
    end, when everyone reveals to score."""

    ware: Optional[Ware] = None
    encumbered: bool = False

    @property
    def is_recorded(self) -> bool:
        return self.ware is not None

    def to_dict(self) -> dict:
        return {"ware": self.ware.value if self.ware else None, "encumbered": self.encumbered}

    @staticmethod
    def from_dict(d: dict) -> "Share":
        ware = d.get("ware")
        return Share(ware=Ware(ware) if ware else None, encumbered=d["encumbered"])


@dataclass
class BlackMarket:
    values: Dict[Ware, int] = field(
        default_factory=lambda: {w: 0 for w in Ware}
    )

    def share_price(self, ware: Ware) -> int:
        return max(5, self.values[ware])

    def raise_value(self, ware: Ware) -> None:
        """Move one step up the (non-uniform) 0-5-10-20-30 track."""
        current = self.values[ware]
        idx = BLACK_MARKET_LEVELS.index(current) if current in BLACK_MARKET_LEVELS else 0
        if idx < len(BLACK_MARKET_LEVELS) - 1:
            self.values[ware] = BLACK_MARKET_LEVELS[idx + 1]

    def is_game_over(self) -> bool:
        return any(v >= GAME_END_VALUE for v in self.values.values())

    def to_dict(self) -> dict:
        return {w.value: v for w, v in self.values.items()}

    @staticmethod
    def from_dict(d: dict) -> "BlackMarket":
        return BlackMarket(values={Ware(k): v for k, v in d.items()})


@dataclass
class AccompliceSlot:
    """A single place an accomplice can be deployed."""

    price: int = 0
    payout: int = 0
    occupant: Optional[str] = None  # player id, or None if vacant

    def to_dict(self) -> dict:
        return {"price": self.price, "payout": self.payout, "occupant": self.occupant}

    @staticmethod
    def from_dict(d: dict) -> "AccompliceSlot":
        return AccompliceSlot(price=d["price"], payout=d["payout"], occupant=d.get("occupant"))


@dataclass
class Punt:
    id: int
    ware: Optional[Ware] = None
    position: int = 0
    status: PuntStatus = PuntStatus.ON_ROUTE
    dock_slot: Optional[str] = None  # 'A' / 'B' / 'C' once in port or shipyard
    ware_slots: List[AccompliceSlot] = field(default_factory=list)

    @staticmethod
    def new(punt_id: int, ware: Optional[Ware] = None) -> "Punt":
        slots = []
        if ware is not None:
            prices = DEFAULT_WARE_SLOT_PRICES[ware]
            slots = [AccompliceSlot(price=p) for p in prices]
        return Punt(id=punt_id, ware=ware, ware_slots=slots)

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "ware": self.ware.value if self.ware else None,
            "position": self.position,
            "status": self.status.value,
            "dock_slot": self.dock_slot,
            "ware_slots": [s.to_dict() for s in self.ware_slots],
        }

    @staticmethod
    def from_dict(d: dict) -> "Punt":
        return Punt(
            id=d["id"],
            ware=Ware(d["ware"]) if d["ware"] else None,
            position=d["position"],
            status=PuntStatus(d["status"]),
            dock_slot=d.get("dock_slot"),
            ware_slots=[AccompliceSlot.from_dict(s) for s in d["ware_slots"]],
        )


@dataclass
class DockSlots:
    """Port or shipyard: three labelled slots A/B/C."""

    slots: Dict[str, AccompliceSlot] = field(default_factory=dict)

    @staticmethod
    def new(payouts: Dict[str, int], prices: Optional[Dict[str, int]] = None) -> "DockSlots":
        prices = prices or {}
        return DockSlots(
            slots={k: AccompliceSlot(payout=v, price=prices.get(k, 0)) for k, v in payouts.items()}
        )

    def to_dict(self) -> dict:
        return {k: s.to_dict() for k, s in self.slots.items()}

    @staticmethod
    def from_dict(d: dict) -> "DockSlots":
        return DockSlots(slots={k: AccompliceSlot.from_dict(v) for k, v in d.items()})


@dataclass
class PirateBoat:
    captain: AccompliceSlot = field(default_factory=lambda: AccompliceSlot(price=PIRATE_PRICE))
    second: AccompliceSlot = field(default_factory=lambda: AccompliceSlot(price=PIRATE_PRICE))

    def to_dict(self) -> dict:
        return {"captain": self.captain.to_dict(), "second": self.second.to_dict()}

    @staticmethod
    def from_dict(d: dict) -> "PirateBoat":
        return PirateBoat(
            captain=AccompliceSlot.from_dict(d["captain"]), second=AccompliceSlot.from_dict(d["second"])
        )


@dataclass
class PilotIsland:
    small: AccompliceSlot = field(
        default_factory=lambda: AccompliceSlot(price=DEFAULT_PILOT_PRICES["small"])
    )
    large: AccompliceSlot = field(
        default_factory=lambda: AccompliceSlot(price=DEFAULT_PILOT_PRICES["large"])
    )

    def to_dict(self) -> dict:
        return {"small": self.small.to_dict(), "large": self.large.to_dict()}

    @staticmethod
    def from_dict(d: dict) -> "PilotIsland":
        return PilotIsland(
            small=AccompliceSlot.from_dict(d["small"]),
            large=AccompliceSlot.from_dict(d["large"]),
        )


@dataclass
class InsuranceOffice:
    occupant: Optional[str] = None
    payment: int = INSURANCE_PAYMENT

    def to_dict(self) -> dict:
        return {"occupant": self.occupant, "payment": self.payment}

    @staticmethod
    def from_dict(d: dict) -> "InsuranceOffice":
        return InsuranceOffice(occupant=d.get("occupant"), payment=d["payment"])


@dataclass
class Player:
    id: str
    name: str
    color: str
    cash: int = STARTING_CASH
    accomplices_in_hand: int = 3
    accomplices_deployed: int = 0
    shares: List[Share] = field(default_factory=list)
    is_harbor_master: bool = False
    is_bot: bool = False  # computer-controlled
    policy: str = "random"  # "random" (default) or "rev" -- see manilla.engine.policy

    @property
    def unencumbered_shares(self) -> List[Share]:
        return [s for s in self.shares if not s.encumbered]

    @property
    def encumbered_shares(self) -> List[Share]:
        return [s for s in self.shares if s.encumbered]

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "name": self.name,
            "color": self.color,
            "cash": self.cash,
            "accomplices_in_hand": self.accomplices_in_hand,
            "accomplices_deployed": self.accomplices_deployed,
            "shares": [s.to_dict() for s in self.shares],
            "is_harbor_master": self.is_harbor_master,
            "is_bot": self.is_bot,
            "policy": self.policy,
        }

    @staticmethod
    def from_dict(d: dict) -> "Player":
        return Player(
            id=d["id"],
            name=d["name"],
            color=d["color"],
            cash=d["cash"],
            accomplices_in_hand=d["accomplices_in_hand"],
            accomplices_deployed=d["accomplices_deployed"],
            shares=[Share.from_dict(s) for s in d["shares"]],
            is_harbor_master=d["is_harbor_master"],
            is_bot=d.get("is_bot", False),
            policy=d.get("policy", "random"),
        )


@dataclass
class GameState:
    voyage_number: int = 1
    phase: Phase = Phase.AUCTION
    players: List[Player] = field(default_factory=list)
    punts: List[Punt] = field(default_factory=list)
    unloaded_ware: Optional[Ware] = None
    black_market: BlackMarket = field(default_factory=BlackMarket)
    port: DockSlots = field(
        default_factory=lambda: DockSlots.new(DEFAULT_PORT_PAYOUTS, DEFAULT_PORT_PRICES)
    )
    shipyard: DockSlots = field(
        default_factory=lambda: DockSlots.new(DEFAULT_SHIPYARD_PAYOUTS, DEFAULT_SHIPYARD_PRICES)
    )
    pirate_boat: PirateBoat = field(default_factory=PirateBoat)
    pilot_island: PilotIsland = field(default_factory=PilotIsland)
    insurance: InsuranceOffice = field(default_factory=InsuranceOffice)
    last_dice: Dict[Ware, int] = field(default_factory=dict)
    accomplice_round_index: int = 0
    movement_round_index: int = 0
    current_turn_player_id: Optional[str] = None
    game_setup_confirmed: bool = False  # player count/colors locked once True
    # Shares held by a player whose hand was never recorded -- see `Share`.
    # Normally empty, and then availability is just `SHARES_PER_WARE -
    # shares_owned`, exactly as the rules imply (p.2: every undealt share
    # goes on the table). When a hand is left unrecorded its shares are
    # missing from `shares_owned`, so they are counted here instead: enough
    # to keep the stock right, without attributing a ware to anybody.
    unrecorded_holdings: Dict[Ware, int] = field(default_factory=dict)

    @property
    def player_count(self) -> int:
        return len(self.players)

    @property
    def accomplice_rounds_total(self) -> int:
        return 4 if self.player_count == 3 else 3

    @property
    def movement_rounds_total(self) -> int:
        return 3

    def player_by_id(self, player_id: str) -> Optional[Player]:
        return next((p for p in self.players if p.id == player_id), None)

    def shares_owned(self, ware: Ware) -> int:
        return sum(1 for p in self.players for s in p.shares if s.ware == ware)

    def shares_available(self, ware: Ware) -> int:
        unrecorded = self.unrecorded_holdings.get(ware, 0)
        return max(0, SHARES_PER_WARE - self.shares_owned(ware) - unrecorded)

    def unrecorded_share_count(self) -> int:
        """How many held shares have no ware recorded, across all players."""
        return sum(1 for p in self.players for s in p.shares if not s.is_recorded)

    def record_share_identity(self, share: Share, ware: Ware) -> None:
        """Name a share that was being held unrecorded -- at game end, when
        the rules have everyone reveal to score.

        Moving it out of `unrecorded_holdings` as it lands in `shares_owned`
        is what keeps `shares_available` unchanged: revealing a share you
        already held doesn't put anything back on the table."""
        if share.is_recorded:
            raise ValueError("That share's ware is already recorded.")
        if self.unrecorded_holdings.get(ware, 0) <= 0:
            raise ValueError(
                f"No unrecorded {ware.value} share is outstanding to name."
            )
        share.ware = ware
        self.unrecorded_holdings[ware] -= 1
        if self.unrecorded_holdings[ware] == 0:
            del self.unrecorded_holdings[ware]

    def validate(self) -> List[str]:
        """Non-blocking sanity warnings, not hard rule enforcement."""
        warnings: List[str] = []

        if self.player_count not in (3, 4, 5):
            warnings.append(f"Player count {self.player_count} is outside 3-5.")

        on_route = [p for p in self.punts if p.status == PuntStatus.ON_ROUTE]
        start_positions = [p.position for p in on_route if p.position <= MAX_START_SPACE]
        if len(on_route) == 3 and all(p.position <= MAX_START_SPACE for p in on_route):
            total = sum(p.position for p in on_route)
            if total != PUNT_START_SUM:
                warnings.append(
                    f"Punt start positions sum to {total}, expected {PUNT_START_SUM}."
                )

        for p in self.punts:
            if p.position > SEA_ROUTE_LENGTH:
                warnings.append(f"Punt {p.id} position {p.position} exceeds route length.")
            if p.status == PuntStatus.ON_ROUTE and p.position <= MAX_START_SPACE:
                pass  # legal starting spot
            elif p.status == PuntStatus.ON_ROUTE and p.position > SEA_ROUTE_LENGTH:
                warnings.append(f"Punt {p.id} on route but past space {SEA_ROUTE_LENGTH}.")

        wares_loaded = [p.ware for p in self.punts if p.ware is not None]
        if len(wares_loaded) != len(set(wares_loaded)):
            warnings.append("Two punts are loaded with the same ware.")
        if self.unloaded_ware is not None and self.unloaded_ware in wares_loaded:
            warnings.append("Unloaded ware is also loaded onto a punt.")
        if len(wares_loaded) == 3 and self.unloaded_ware is None:
            warnings.append("Three wares loaded but no ware marked as left ashore.")

        occupied_ports = [k for k, s in self.port.slots.items() if s.occupant]
        occupied_yards = [k for k, s in self.shipyard.slots.items() if s.occupant]
        if len(occupied_ports) > 3 or len(occupied_yards) > 3:
            warnings.append("More than 3 punts docked at port or shipyard.")

        harbor_masters = [p for p in self.players if p.is_harbor_master]
        if len(harbor_masters) > 1:
            warnings.append("More than one player marked as harbor master.")

        for ware in Ware:
            owned = self.shares_owned(ware)
            if owned > SHARES_PER_WARE:
                warnings.append(
                    f"{owned} {ware.value} shares are owned across all players, but only {SHARES_PER_WARE} exist."
                )
            unrecorded = self.unrecorded_holdings.get(ware, 0)
            if owned + unrecorded > SHARES_PER_WARE:
                warnings.append(
                    f"{owned} {ware.value} shares are recorded and {unrecorded} unrecorded, "
                    f"which is more than the {SHARES_PER_WARE} that exist."
                )
        outstanding = sum(self.unrecorded_holdings.values())
        if outstanding != self.unrecorded_share_count():
            warnings.append(
                f"{outstanding} shares are held unrecorded by ware, but "
                f"{self.unrecorded_share_count()} unnamed shares are in hands."
            )

        return warnings

    def to_dict(self) -> dict:
        return {
            "voyage_number": self.voyage_number,
            "phase": self.phase.value,
            "players": [p.to_dict() for p in self.players],
            "punts": [p.to_dict() for p in self.punts],
            "unloaded_ware": self.unloaded_ware.value if self.unloaded_ware else None,
            "black_market": self.black_market.to_dict(),
            "port": self.port.to_dict(),
            "shipyard": self.shipyard.to_dict(),
            "pirate_boat": self.pirate_boat.to_dict(),
            "pilot_island": self.pilot_island.to_dict(),
            "insurance": self.insurance.to_dict(),
            "last_dice": {w.value: v for w, v in self.last_dice.items()},
            "accomplice_round_index": self.accomplice_round_index,
            "movement_round_index": self.movement_round_index,
            "current_turn_player_id": self.current_turn_player_id,
            "game_setup_confirmed": self.game_setup_confirmed,
            "unrecorded_holdings": {
                w.value: n for w, n in self.unrecorded_holdings.items() if n
            },
        }

    @staticmethod
    def from_dict(d: dict) -> "GameState":
        return GameState(
            voyage_number=d["voyage_number"],
            phase=Phase(d["phase"]),
            players=[Player.from_dict(p) for p in d["players"]],
            punts=[Punt.from_dict(p) for p in d["punts"]],
            unloaded_ware=Ware(d["unloaded_ware"]) if d["unloaded_ware"] else None,
            black_market=BlackMarket.from_dict(d["black_market"]),
            port=DockSlots.from_dict(d["port"]),
            shipyard=DockSlots.from_dict(d["shipyard"]),
            pirate_boat=PirateBoat.from_dict(d["pirate_boat"]),
            pilot_island=PilotIsland.from_dict(d["pilot_island"]),
            insurance=InsuranceOffice.from_dict(d["insurance"]),
            last_dice={Ware(k): v for k, v in d["last_dice"].items()},
            accomplice_round_index=d["accomplice_round_index"],
            movement_round_index=d["movement_round_index"],
            current_turn_player_id=d.get("current_turn_player_id"),
            game_setup_confirmed=d.get("game_setup_confirmed", False),
            unrecorded_holdings={
                Ware(k): n for k, n in d.get("unrecorded_holdings", {}).items()
            },
        )

    def save(self, path: str) -> None:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2)

    @staticmethod
    def load(path: str) -> "GameState":
        with open(path, "r", encoding="utf-8") as f:
            return GameState.from_dict(json.load(f))

    @staticmethod
    def new_default_game(
        player_names: List[str], colors: Optional[List[str]] = None, seed: Optional[int] = None
    ) -> "GameState":
        """Build the documented starting setup for a game with 3-5 players."""
        count = len(player_names)
        if count not in (3, 4, 5):
            raise ValueError("Manilla is played with 3 to 5 players.")

        rng = random.Random(seed)
        default_colors = ["red", "blue", "yellow", "black", "orange"]
        colors = colors or default_colors[:count]

        accomplices_in_hand = 4 if count == 3 else 3

        players = [
            Player(id=f"p{i}", name=name, color=colors[i], accomplices_in_hand=accomplices_in_hand)
            for i, name in enumerate(player_names)
        ]

        # Deal hidden starting shares: pool is 3 of each ware, shuffled.
        pool: List[Share] = []
        for ware in Ware:
            pool.extend(Share(ware=ware) for _ in range(3))
        rng.shuffle(pool)
        for player in players:
            for _ in range(STARTING_SHARES_PER_PLAYER):
                if pool:
                    player.shares.append(pool.pop())

        # No default harbor master -- that's decided by the auction dialog
        # (falling back to player 0 as the very first voyage's starting
        # bidder only, not as a pre-assigned office).

        punts = [Punt.new(i) for i in range(3)]

        return GameState(
            voyage_number=1,
            phase=Phase.AUCTION,
            players=players,
            punts=punts,
            black_market=BlackMarket(),
        )

    @staticmethod
    def hand_deal_problems(
        hands: List[Optional[List[Ware]]], available: Optional[Dict[Ware, int]] = None
    ) -> List[str]:
        """Reasons `hands`/`available` couldn't be a real opening position.

        A `None` hand is a seat that didn't say what it was dealt -- a human
        who would rather not tell the game state. Those two shares still
        exist and still came off the table, so what makes the position add
        up is the stock: whatever isn't dealt and isn't for sale has to be
        exactly the unrecorded hands, no more and no less.

        Split out from `new_hand_dealt_game` so the setup dialog can show
        these as you type instead of only when you press Confirm -- both go
        through this one list, so the dialog can't drift from what the
        factory will actually accept.
        """
        problems: List[str] = []
        unrecorded_seats = sum(1 for hand in hands if hand is None)
        recorded = [hand for hand in hands if hand is not None]

        for i, hand in enumerate(hands):
            if hand is not None and len(hand) != STARTING_SHARES_PER_PLAYER:
                problems.append(
                    f"Player {i + 1} has {len(hand)} shares; every player starts with "
                    f"{STARTING_SHARES_PER_PLAYER}."
                )

        if available is None:
            if unrecorded_seats:
                problems.append(
                    "With a hand left unrecorded, how many of each ware are for sale "
                    "has to be filled in -- it's the only thing left that says where "
                    "those shares went."
                )
            available = {}

        hidden_total = 0
        for ware in Ware:
            dealt = sum(1 for hand in recorded for w in hand if w == ware)
            # Checked before anything to do with the stock, so an over-dealt
            # ware reports the deal itself rather than the negative stock
            # that is only a consequence of it.
            if dealt > DEAL_POOL_PER_WARE:
                problems.append(
                    f"{dealt} {ware.value} shares are dealt, but the opening deal is "
                    f"made from only {DEAL_POOL_PER_WARE} of each ware."
                )
                continue
            spare = SHARES_PER_WARE - dealt
            asked = available.get(ware, spare)
            if asked < 0:
                problems.append(f"Available {ware.value} shares can't be negative.")
                continue
            if asked > spare:
                problems.append(
                    f"{asked} {ware.value} shares can't be available: {dealt} of the "
                    f"{SHARES_PER_WARE} are dealt out, leaving at most {spare}."
                )
                continue
            hidden = spare - asked
            hidden_total += hidden
            # The deal comes off a pile of DEAL_POOL_PER_WARE of each ware,
            # so this bounds the recorded and unrecorded hands together.
            if dealt + hidden > DEAL_POOL_PER_WARE:
                problems.append(
                    f"{dealt + hidden} {ware.value} shares would be in hands, but the "
                    f"opening deal is made from only {DEAL_POOL_PER_WARE} of each ware."
                )

        expected_hidden = unrecorded_seats * STARTING_SHARES_PER_PLAYER
        if not problems and hidden_total != expected_hidden:
            if unrecorded_seats == 0:
                problems.append(
                    f"{hidden_total} shares are neither dealt out nor for sale, but "
                    f"every hand is recorded, so there is nowhere for them to be."
                )
            else:
                short = expected_hidden - hidden_total
                seats = "hand holds" if unrecorded_seats == 1 else "hands hold"
                problems.append(
                    f"{unrecorded_seats} unrecorded {seats} {expected_hidden} shares, "
                    f"but {hidden_total} are unaccounted for: "
                    + (
                        f"take {short} more off the shares for sale."
                        if short > 0
                        else f"put {-short} back on the shares for sale."
                    )
                )

        return problems

    @staticmethod
    def new_hand_dealt_game(
        player_names: List[str],
        hands: List[Optional[List[Ware]]],
        available: Optional[Dict[Ware, int]] = None,
        colors: Optional[List[str]] = None,
    ) -> "GameState":
        """The documented starting setup, but with the opening deal specified
        rather than shuffled -- for reproducing a position from a real table.

        A `None` hand means that seat's shares stay unrecorded: they're
        counted in `unrecorded_holdings` from the stock numbers instead of
        being attributed to anyone, and named only at game end.

        `available` is how many of each ware are for sale. It can be left out
        only when every hand is recorded, in which case it follows the rules
        (everything undealt is on the table).

        Recorded hands are *set*, not *revealed*: nothing tells the computer
        players who holds what, so they still infer opponents' hands from
        public signals the same way (see `manilla.engine.beliefs`).
        """
        if len(hands) != len(player_names):
            raise ValueError(
                f"Got {len(hands)} hands for {len(player_names)} players."
            )
        problems = GameState.hand_deal_problems(hands, available)
        if problems:
            raise ValueError("; ".join(problems))

        state = GameState.new_default_game(player_names, colors=colors)
        for player, hand in zip(state.players, hands):
            if hand is None:
                player.shares = [Share() for _ in range(STARTING_SHARES_PER_PLAYER)]
            else:
                player.shares = [Share(ware=ware) for ware in hand]

        if available is not None:
            for ware in Ware:
                dealt = sum(
                    1 for hand in hands if hand for w in hand if w == ware
                )
                hidden = SHARES_PER_WARE - dealt - available.get(ware, SHARES_PER_WARE - dealt)
                if hidden:
                    state.unrecorded_holdings[ware] = hidden

        return state
