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
SEA_ROUTE_LENGTH = 13  # spaces 0-13
PUNT_START_SUM = 9
MAX_START_SPACE = 5
GAME_END_VALUE = 30
BLACK_MARKET_LEVELS = [0, 5, 10, 20, 30]


@dataclass
class Share:
    ware: Ware
    encumbered: bool = False

    def to_dict(self) -> dict:
        return {"ware": self.ware.value, "encumbered": self.encumbered}

    @staticmethod
    def from_dict(d: dict) -> "Share":
        return Share(ware=Ware(d["ware"]), encumbered=d["encumbered"])


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
    is_bot: bool = False  # computer-controlled with a random policy

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
        return max(0, SHARES_PER_WARE - self.shares_owned(ware))

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
