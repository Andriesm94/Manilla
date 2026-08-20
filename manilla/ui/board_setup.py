"""Tkinter interface for visualizing and manually editing a Manilla board setup.

This is a sandbox tool: it lets you construct an arbitrary snapshot of a
game (players, cash, shares, punt positions/wares, black market values,
accomplice placements) to later feed into the rules/AI engine. It does not
enforce game rules -- `GameState.validate()` warnings are shown but never
block an edit.
"""

from __future__ import annotations

import random
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from typing import Callable, List, Optional, Tuple

from manilla.engine.models import (
    BLACK_MARKET_LEVELS,
    INSURANCE_SHIPYARD_COST,
    PLUNDER_PAYOUTS,
    SHARE_LOAN_AMOUNT,
    SHARES_PER_WARE,
    STARTING_CASH,
    AccompliceSlot,
    BlackMarket,
    GameState,
    MAX_START_SPACE,
    Phase,
    Player,
    Punt,
    PuntStatus,
    PUNT_START_SUM,
    SEA_ROUTE_LENGTH,
    Share,
    Ware,
)

WARE_COLORS = {
    Ware.NUTMEG: "#8B5A2B",
    Ware.SILK: "#4A90D9",
    Ware.GINSENG: "#D8CB6A",
    Ware.JADE: "#4CAF50",
}
# Readable text color against each ware's punt-token fill above.
WARE_TEXT_COLORS = {
    Ware.NUTMEG: "white",
    Ware.SILK: "white",
    Ware.GINSENG: "#333333",
    Ware.JADE: "white",
}

DEFAULT_PLAYER_COLORS = ["red", "blue", "yellow", "black", "orange"]
VACANT_FILL = "#ffffff"

CANVAS_WIDTH = 760  # visible viewport; drawn content is wider and scrolls horizontally
CANVAS_HEIGHT = 400  # tall enough to fit all drawn content without vertical clipping
LANE_LEFT_X = 190
CELL_W = 46
LANE_TOP_Y = 60
LANE_GAP = 90
PUNT_RADIUS = 14
SLOT_RADIUS = 12  # ware-punt accomplice slots
DOCK_RADIUS = 14  # port / shipyard accomplice slots
BIG_SLOT_RADIUS = 16  # pirate / pilot / insurance accomplice slots
MINI_SLOT_RADIUS = 7  # ware-punt accomplice slots once a punt has docked

# Port / shipyard layout: [accomplice circle] [boat-dock rectangle], mirrored
# for shipyard, with a row per A/B/C. The boat-dock rectangle is wide enough
# to hold the docked punt's own ware-accomplice mini-strip inside it (up to
# 4 for jade), instead of a separate strip below.
DOCK_ROW_H = 110
DOCK_COL_GAP = 72  # accomplice circle <-> boat-dock rectangle, within one side
DOCK_MID_GAP = 26  # port's boat rectangles <-> shipyard's boat rectangles
DOCK_RECT_W = 70
DOCK_RECT_H = 26
DOCK_MINI_SPACING = 16
DOCK_LABEL_DY = -(DOCK_RADIUS + 12)
DOCK_TITLE_DY = DOCK_LABEL_DY - 16

# Row holding the pirate boat, pilot island, and insurance office.
MISC_ROW_Y = LANE_TOP_Y + 3 * LANE_GAP + 30
PIRATE_X = LANE_LEFT_X
PILOT_X = LANE_LEFT_X + 220
INSURANCE_X = LANE_LEFT_X + 440
MISC_TITLE_DY = -(BIG_SLOT_RADIUS + 26)  # group title, above the sub-label
MISC_LABEL_DY = -(BIG_SLOT_RADIUS + 10)  # sub-label, clear of the circle's top edge

PLAYERS_PANEL_WIDTH = 300


ClickRegion = Tuple[int, int, int, int, Callable[[], None]]


class BoardSetupApp(tk.Frame):
    def __init__(self, master: tk.Misc, state: Optional[GameState] = None):
        super().__init__(master)
        self.state_obj: GameState = state or GameState.new_default_game(
            ["Player 1", "Player 2", "Player 3", "Player 4"]
        )
        self._click_regions: List[ClickRegion] = []
        self._player_widgets: dict = {}
        self._round_placements = 0  # accomplice placements so far in the current round (session-only, not saved)

        self._build_toolbar()
        self._build_body()
        self.refresh()

        if not self.state_obj.game_setup_confirmed:
            self.after(50, self._show_new_game_setup_dialog)

    # ------------------------------------------------------------------
    # Layout scaffolding
    # ------------------------------------------------------------------
    def _build_toolbar(self) -> None:
        bar = ttk.Frame(self)
        bar.pack(side=tk.TOP, fill=tk.X, padx=6, pady=6)

        ttk.Button(bar, text="New default game...", command=self.on_new_default_game).pack(
            side=tk.LEFT, padx=2
        )
        ttk.Button(bar, text="Randomize", command=self.on_randomize).pack(side=tk.LEFT, padx=2)
        ttk.Button(bar, text="Save...", command=self.on_save).pack(side=tk.LEFT, padx=2)
        ttk.Button(bar, text="Load...", command=self.on_load).pack(side=tk.LEFT, padx=2)
        self.next_voyage_btn = ttk.Button(
            bar, text="Start next voyage...", command=self.on_start_next_voyage, state="disabled"
        )
        self.next_voyage_btn.pack(side=tk.LEFT, padx=(12, 2))

        ttk.Label(bar, text="Voyage:").pack(side=tk.LEFT, padx=(20, 2))
        self.voyage_var = tk.IntVar()
        self.voyage_spin = ttk.Spinbox(
            bar, from_=1, to=999, width=4, textvariable=self.voyage_var, command=self.on_voyage_change
        )
        self.voyage_spin.pack(side=tk.LEFT)
        self.voyage_spin.bind("<Return>", lambda e: self.on_voyage_change())

        ttk.Label(bar, text="Phase:").pack(side=tk.LEFT, padx=(20, 2))
        self.phase_var = tk.StringVar()
        self.phase_combo = ttk.Combobox(
            bar,
            textvariable=self.phase_var,
            values=[p.value for p in Phase],
            state="readonly",
            width=20,
        )
        self.phase_combo.pack(side=tk.LEFT)
        self.phase_combo.bind("<<ComboboxSelected>>", lambda e: self.on_phase_change())

        turn_bar = ttk.Frame(self)
        turn_bar.pack(side=tk.TOP, fill=tk.X, padx=6, pady=(0, 6))
        self.turn_var = tk.StringVar()
        self.turn_label = tk.Label(
            turn_bar, textvariable=self.turn_var, font=("Segoe UI", 12, "bold"), anchor="w", padx=8, pady=4
        )
        self.turn_label.pack(side=tk.LEFT, fill=tk.X)
        ttk.Label(
            turn_bar,
            text="Click a vacant accomplice circle to place for the current player.",
            foreground="#555",
        ).pack(side=tk.LEFT, padx=(12, 0))

    def _build_body(self) -> None:
        body = ttk.Frame(self)
        body.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        # The left column (board + punts/market panels) can be taller than a
        # modest screen, so it lives inside its own vertically-scrollable area.
        left_outer = ttk.Frame(body)
        left_outer.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=6, pady=6)

        left_scroll = tk.Canvas(left_outer, borderwidth=0, highlightthickness=0)
        left_vsb = ttk.Scrollbar(left_outer, orient="vertical", command=left_scroll.yview)
        left = ttk.Frame(left_scroll)
        left.bind("<Configure>", lambda e: left_scroll.configure(scrollregion=left_scroll.bbox("all")))
        left_scroll.create_window((0, 0), window=left, anchor="nw")
        left_scroll.configure(yscrollcommand=left_vsb.set)
        left_scroll.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        left_vsb.pack(side=tk.LEFT, fill=tk.Y)

        # The board itself is wider (and, with the port/shipyard docks, taller)
        # than most screens, so it gets its own scrollbars in both directions
        # rather than relying on content always fitting a fixed size.
        board_frame = ttk.Frame(left)
        board_frame.pack(side=tk.TOP, fill=tk.X)
        board_frame.columnconfigure(0, weight=1)
        board_frame.rowconfigure(0, weight=1)

        self.canvas = tk.Canvas(board_frame, width=CANVAS_WIDTH, height=CANVAS_HEIGHT, bg="#dff0f7")
        board_vsb = ttk.Scrollbar(board_frame, orient="vertical", command=self.canvas.yview)
        board_hsb = ttk.Scrollbar(board_frame, orient="horizontal", command=self.canvas.xview)
        self.canvas.configure(xscrollcommand=board_hsb.set, yscrollcommand=board_vsb.set)
        self.canvas.grid(row=0, column=0, sticky="nsew")
        board_vsb.grid(row=0, column=1, sticky="ns")
        board_hsb.grid(row=1, column=0, sticky="ew")
        self.canvas.bind("<Button-1>", self.on_canvas_click)

        def on_board_mousewheel(event):
            self.canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")

        self.canvas.bind("<Enter>", lambda e: self.canvas.bind_all("<MouseWheel>", on_board_mousewheel))
        self.canvas.bind("<Leave>", lambda e: self.canvas.unbind_all("<MouseWheel>"))

        self.warnings_var = tk.StringVar(value="")
        warnings_label = tk.Label(
            left, textvariable=self.warnings_var, fg="#b02a2a", justify=tk.LEFT, anchor="w", wraplength=CANVAS_WIDTH
        )
        warnings_label.pack(side=tk.TOP, fill=tk.X, pady=(4, 0))

        self._build_punts_panel(left)
        self._build_market_panel(left)
        self._build_shares_panel(left)

        right = ttk.Frame(body, width=PLAYERS_PANEL_WIDTH)
        right.pack(side=tk.RIGHT, fill=tk.Y, padx=6, pady=2)
        right.pack_propagate(False)

        player_bar = ttk.Frame(right)
        player_bar.pack(side=tk.TOP, fill=tk.X)
        ttk.Label(player_bar, text="Players", font=("Segoe UI", 11, "bold")).pack(side=tk.LEFT)
        self.add_player_btn = ttk.Button(player_bar, text="+ Add", command=self.on_add_player)
        self.add_player_btn.pack(side=tk.RIGHT)

        canvas_frame = tk.Canvas(right, borderwidth=0, highlightthickness=0)
        scrollbar = ttk.Scrollbar(right, orient="vertical", command=canvas_frame.yview)
        self.players_container = ttk.Frame(canvas_frame)
        players_window = canvas_frame.create_window((0, 0), window=self.players_container, anchor="nw")

        def sync_scrollregion(_event=None):
            canvas_frame.configure(scrollregion=canvas_frame.bbox("all"))

        def sync_inner_width(event):
            # keep the inner frame exactly as wide as the visible canvas so
            # content wraps/clips vertically only -- no horizontal scroll needed.
            canvas_frame.itemconfigure(players_window, width=event.width)

        self.players_container.bind("<Configure>", sync_scrollregion)
        canvas_frame.bind("<Configure>", sync_inner_width)
        canvas_frame.configure(yscrollcommand=scrollbar.set)
        canvas_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        def on_mousewheel(event):
            canvas_frame.yview_scroll(int(-1 * (event.delta / 120)), "units")

        def bind_wheel(_event=None):
            canvas_frame.bind_all("<MouseWheel>", on_mousewheel)

        def unbind_wheel(_event=None):
            canvas_frame.unbind_all("<MouseWheel>")

        canvas_frame.bind("<Enter>", bind_wheel)
        canvas_frame.bind("<Leave>", unbind_wheel)

    def _build_punts_panel(self, parent: tk.Misc) -> None:
        frame = ttk.LabelFrame(parent, text="Punts")
        frame.pack(side=tk.TOP, fill=tk.X, pady=(8, 0))
        self.punt_rows: List[dict] = []
        for i in range(3):
            row = ttk.Frame(frame)
            row.pack(side=tk.TOP, fill=tk.X, pady=2)
            ttk.Label(row, text=f"Punt {i}", width=8).pack(side=tk.LEFT)

            ware_var = tk.StringVar()
            ware_combo = ttk.Combobox(
                row, textvariable=ware_var, values=["(none)"] + [w.value for w in Ware], state="readonly", width=10
            )
            ware_combo.pack(side=tk.LEFT, padx=4)

            status_var = tk.StringVar()
            status_combo = ttk.Combobox(
                row, textvariable=status_var, values=[s.value for s in PuntStatus], state="readonly", width=12
            )
            status_combo.pack(side=tk.LEFT, padx=4)

            pos_var = tk.IntVar()
            pos_spin = ttk.Spinbox(row, from_=0, to=SEA_ROUTE_LENGTH, width=4, textvariable=pos_var)
            pos_spin.pack(side=tk.LEFT, padx=4)

            def make_handler(idx=i):
                return lambda *_: self.on_punt_row_change(idx)

            handler = make_handler()
            ware_combo.bind("<<ComboboxSelected>>", handler)
            status_combo.bind("<<ComboboxSelected>>", handler)
            pos_spin.bind("<Return>", handler)
            pos_spin.bind("<FocusOut>", handler)

            self.punt_rows.append(
                {
                    "ware": ware_var,
                    "status": status_var,
                    "position": pos_var,
                    "ware_widget": ware_combo,
                    "status_widget": status_combo,
                    "position_widget": pos_spin,
                }
            )

        ttk.Label(
            frame,
            text="Tip: click a cell on a sea lane in the board above to set that punt's position directly.",
            foreground="#555",
        ).pack(side=tk.TOP, anchor="w", pady=(2, 4))

        unloaded_row = ttk.Frame(frame)
        unloaded_row.pack(side=tk.TOP, fill=tk.X, pady=2)
        ttk.Label(unloaded_row, text="Ware left ashore:", width=16).pack(side=tk.LEFT)
        self.unloaded_var = tk.StringVar()
        unloaded_combo = ttk.Combobox(
            unloaded_row,
            textvariable=self.unloaded_var,
            values=["(none)"] + [w.value for w in Ware],
            state="readonly",
            width=10,
        )
        unloaded_combo.pack(side=tk.LEFT)
        unloaded_combo.bind("<<ComboboxSelected>>", lambda e: self.on_unloaded_change())
        self.unloaded_widget = unloaded_combo

    def _build_market_panel(self, parent: tk.Misc) -> None:
        frame = ttk.LabelFrame(parent, text="Black market values")
        frame.pack(side=tk.TOP, fill=tk.X, pady=(8, 0))
        self.market_vars: dict = {}
        self.market_widgets: dict = {}
        for ware in Ware:
            col = ttk.Frame(frame)
            col.pack(side=tk.LEFT, padx=12, pady=4)
            ttk.Label(col, text=ware.value.title(), foreground=WARE_COLORS[ware]).pack()
            var = tk.IntVar()
            spin = ttk.Spinbox(
                col,
                values=BLACK_MARKET_LEVELS,
                width=4,
                textvariable=var,
                command=lambda w=ware: self.on_market_change(w),
            )
            spin.pack()
            spin.bind("<Return>", lambda e, w=ware: self.on_market_change(w))
            self.market_vars[ware] = var
            self.market_widgets[ware] = spin

    def _build_shares_panel(self, parent: tk.Misc) -> None:
        frame = ttk.LabelFrame(parent, text=f"Shares available to buy (of {SHARES_PER_WARE} each)")
        frame.pack(side=tk.TOP, fill=tk.X, pady=(8, 0))
        self.shares_canvases: dict = {}
        self.shares_labels: dict = {}
        for ware in Ware:
            col = ttk.Frame(frame)
            col.pack(side=tk.LEFT, padx=12, pady=4)
            ttk.Label(col, text=ware.value.title(), foreground=WARE_COLORS[ware]).pack()
            mini = tk.Canvas(col, width=SHARES_PER_WARE * 22, height=22, highlightthickness=0)
            mini.pack()
            lbl = ttk.Label(col, text="")
            lbl.pack()
            self.shares_canvases[ware] = mini
            self.shares_labels[ware] = lbl

    def update_shares_panel(self) -> None:
        for ware in Ware:
            mini = self.shares_canvases[ware]
            mini.delete("all")
            available = self.state_obj.shares_available(ware)
            for i in range(SHARES_PER_WARE):
                cx, cy = 11 + i * 22, 11
                filled = i < available
                fill = WARE_COLORS[ware] if filled else "#e0e0e0"
                mini.create_oval(cx - 9, cy - 9, cx + 9, cy + 9, fill=fill, outline="#555")
            self.shares_labels[ware].configure(text=f"{available}/{SHARES_PER_WARE} available")

    # ------------------------------------------------------------------
    # Refresh / draw
    # ------------------------------------------------------------------
    def refresh(self) -> None:
        self.voyage_var.set(self.state_obj.voyage_number)
        self.phase_var.set(self.state_obj.phase.value)
        self.unloaded_var.set(self.state_obj.unloaded_ware.value if self.state_obj.unloaded_ware else "(none)")

        for ware, var in self.market_vars.items():
            var.set(self.state_obj.black_market.values[ware])

        for i, punt in enumerate(self.state_obj.punts):
            row = self.punt_rows[i]
            row["ware"].set(punt.ware.value if punt.ware else "(none)")
            row["status"].set(punt.status.value)
            row["position"].set(punt.position)

        self.rebuild_players_panel()
        self.draw_canvas()
        self.update_warnings()
        self.update_shares_panel()
        self.update_turn_label()
        self.update_lock_states()

    def update_lock_states(self) -> None:
        confirmed = self.state_obj.game_setup_confirmed
        self.add_player_btn.configure(state="disabled" if confirmed else "normal")

        # Once the game has started, loading wares, setting punt start
        # positions, and moving/docking punts are all handled exclusively by
        # the game's own flow (the harbor master's dialog, dice rolls, pilots,
        # pirates) -- these controls are locked here so there's only one path
        # to change them.
        for row in self.punt_rows:
            widget_state = "disabled" if confirmed else "readonly"
            row["ware_widget"].configure(state=widget_state)
            row["status_widget"].configure(state=widget_state)
            row["position_widget"].configure(state=("disabled" if confirmed else "normal"))
        self.unloaded_widget.configure(state="disabled" if confirmed else "readonly")

        # Same treatment for the remaining manual-editing escape hatches --
        # money, the black market, the phase, and the voyage number are all
        # driven by the game's own flow once it has actually started.
        self.voyage_spin.configure(state="disabled" if confirmed else "normal")
        self.phase_combo.configure(state="disabled" if confirmed else "readonly")
        for spin in self.market_widgets.values():
            spin.configure(state="disabled" if confirmed else "normal")

        self.next_voyage_btn.configure(
            state="normal" if self.state_obj.phase == Phase.PROFIT_DISTRIBUTION else "disabled"
        )

    def update_warnings(self) -> None:
        warnings = self.state_obj.validate()
        self.warnings_var.set("\n".join(f"- {w}" for w in warnings) if warnings else "")

    # ------------------------------------------------------------------
    # Canvas drawing
    # ------------------------------------------------------------------
    def draw_canvas(self) -> None:
        c = self.canvas
        c.delete("all")
        self._click_regions = []

        self._draw_sea_lanes()
        self._draw_docks()
        self._draw_pirate_boat()
        self._draw_pilot_island()
        self._draw_insurance()

        bbox = c.bbox("all")
        if bbox:
            padded = (bbox[0] - 10, bbox[1] - 10, bbox[2] + 10, bbox[3] + 10)
            c.configure(scrollregion=padded)

    def _register_click(self, x1: int, y1: int, x2: int, y2: int, callback: Callable[[], None]) -> None:
        self._click_regions.append((x1, y1, x2, y2, callback))

    def on_canvas_click(self, event: tk.Event) -> None:
        # event.x/y are viewport-relative; click regions are stored in canvas
        # item-coordinate space, so they must be converted via canvasx/canvasy
        # or clicks drift by the current scroll offset (worse the more you've
        # scrolled).
        cx = self.canvas.canvasx(event.x)
        cy = self.canvas.canvasy(event.y)
        for x1, y1, x2, y2, callback in reversed(self._click_regions):
            if x1 <= cx <= x2 and y1 <= cy <= y2:
                callback()
                self.refresh()
                return

    def _player_color(self, player_id: Optional[str]) -> str:
        if player_id is None:
            return "#ffffff"
        player = self.state_obj.player_by_id(player_id)
        return player.color if player else "#ffffff"

    # ------------------------------------------------------------------
    # Turn-based accomplice placement
    # ------------------------------------------------------------------
    def _current_turn_player(self) -> Optional[Player]:
        players = self.state_obj.players
        if not players:
            return None
        player = self.state_obj.player_by_id(self.state_obj.current_turn_player_id)
        if player is None:
            player = next((p for p in players if p.is_harbor_master), players[0])
            self.state_obj.current_turn_player_id = player.id
        return player

    def _advance_turn(self) -> None:
        players = self.state_obj.players
        if not players:
            return
        self._round_placements += 1
        if self._round_placements >= len(players):
            self._round_placements = 0
            if self.state_obj.movement_round_index == 2:
                # Pilots act just before the third (final) movement round.
                self._show_pilot_dialogs(self._finish_round_end_roll)
            else:
                self._finish_round_end_roll()
        else:
            current = self.state_obj.player_by_id(self.state_obj.current_turn_player_id)
            idx = players.index(current) if current in players else 0
            self.state_obj.current_turn_player_id = players[(idx + 1) % len(players)].id

    def _finish_round_end_roll(self) -> None:
        self._roll_dice_and_move()
        players = self.state_obj.players
        if players:
            harbor_master = next((p for p in players if p.is_harbor_master), players[0])
            self.state_obj.current_turn_player_id = harbor_master.id
        self.refresh()

    def _settle_payment(self, player: Player, amount: int) -> bool:
        """Deduct `amount` from player's cash. Money never goes negative:
        if they're short, they MUST take credit -- encumbering one of their
        own shares at a time for SHARE_LOAN_AMOUNT PESOS each, their choice
        of which -- until they can cover it or run out of shares. Returns
        False if even that isn't enough (no cash, no shares left); the
        caller decides what that means (e.g. a blind-passenger placement)."""
        while player.cash < amount and player.unencumbered_shares:
            share = self._prompt_encumber_share(player)
            if share is None:
                break
            share.encumbered = True
            player.cash += SHARE_LOAN_AMOUNT
        if player.cash < amount:
            return False
        player.cash -= amount
        return True

    def _prompt_encumber_share(self, player: Player) -> Optional[Share]:
        """Blocks (via a nested event loop) until the player picks one of
        their unencumbered shares to encumber. Only they need to see which
        ware it is, so this is a short-lived dialog, not broadcast anywhere
        else in the app."""
        shares = player.unencumbered_shares
        if not shares:
            return None

        dialog = tk.Toplevel(self.winfo_toplevel())
        dialog.title(f"{player.name}: take credit")
        dialog.transient(self.winfo_toplevel())
        dialog.grab_set()
        dialog.resizable(False, False)
        dialog.protocol("WM_DELETE_WINDOW", lambda: None)  # must choose -- credit is mandatory here

        ttk.Label(
            dialog,
            text=f"{player.name} ({player.color}) doesn't have enough cash ({player.cash} PESOS).\n"
            f"Choose a share to encumber for {SHARE_LOAN_AMOUNT} PESOS credit.",
            wraplength=300,
            justify=tk.LEFT,
            font=("Segoe UI", 10, "bold"),
        ).pack(padx=12, pady=(12, 8))

        chosen: dict = {"share": None}

        def pick(s: Share) -> None:
            chosen["share"] = s
            dialog.destroy()

        for share in shares:
            ttk.Button(
                dialog,
                text=f"Encumber a {share.ware.value.title()} share (+{SHARE_LOAN_AMOUNT})",
                command=lambda s=share: pick(s),
            ).pack(padx=12, pady=2, fill=tk.X)

        self.wait_window(dialog)
        return chosen["share"]

    def _place_or_remove_accomplice(self, slot: AccompliceSlot) -> None:
        """Click a vacant space: the current-turn player occupies it, pays
        its cost, and the turn passes on. Click an occupied space: refund
        and vacate it (a correction/undo -- doesn't affect whose turn it is).

        Locked entirely once a voyage's profit distribution has happened --
        placement only resumes once the next voyage's auction/loading is done.
        """
        if self.state_obj.phase == Phase.PROFIT_DISTRIBUTION:
            return
        if slot.occupant is None:
            player = self._current_turn_player()
            if player is None:
                return
            slot.occupant = player.id
            # If they can't fully afford it (even after taking credit),
            # they ride as a blind passenger -- placed for free.
            self._settle_payment(player, slot.price)
            self._advance_turn()
        else:
            player = self.state_obj.player_by_id(slot.occupant)
            slot.occupant = None
            if player is not None:
                player.cash += slot.price

    def _place_or_remove_dock_accomplice(self, dock: DockSlots, clicked_slot: AccompliceSlot) -> None:
        """Port/shipyard accomplice placement is mandatory-highest-available:
        clicking any vacant circle in the group always places into the
        lowest-lettered (A, then B, then C) vacant slot -- the "highest"
        priority one, which also carries the lowest reward. Clicking an
        occupied circle still removes exactly that one (a correction/undo).
        """
        if clicked_slot.occupant is not None:
            self._place_or_remove_accomplice(clicked_slot)
            return
        for key in ["A", "B", "C"]:
            target = dock.slots[key]
            if target.occupant is None:
                self._place_or_remove_accomplice(target)
                return

    def _place_or_remove_punt_accomplice(self, punt: Punt, clicked_slot: AccompliceSlot) -> None:
        """Ware-punt accomplice placement always takes the cheapest vacant
        space on that punt, regardless of which circle was clicked. Clicking
        an occupied circle still removes exactly that one."""
        if clicked_slot.occupant is not None:
            self._place_or_remove_accomplice(clicked_slot)
            return
        vacant = [s for s in punt.ware_slots if s.occupant is None]
        if not vacant:
            return
        cheapest = min(vacant, key=lambda s: s.price)
        self._place_or_remove_accomplice(cheapest)

    def _place_or_remove_insurance(self) -> None:
        if self.state_obj.phase == Phase.PROFIT_DISTRIBUTION:
            return
        ins = self.state_obj.insurance
        if ins.occupant is None:
            player = self._current_turn_player()
            if player is None:
                return
            ins.occupant = player.id
            player.cash += ins.payment
            self._advance_turn()
        else:
            player = self.state_obj.player_by_id(ins.occupant)
            ins.occupant = None
            if player is not None:
                player.cash -= ins.payment

    def _roll_dice_and_move(self) -> None:
        rolls: dict = {}
        for punt in self.state_obj.punts:
            if punt.ware is None or punt.status != PuntStatus.ON_ROUTE:
                continue
            roll = random.randint(1, 6)
            rolls[punt.ware] = roll
            punt.position += roll
        self.state_obj.last_dice = rolls
        self.state_obj.movement_round_index += 1

        # A punt that moved past space 13 arrives immediately, any round.
        for punt in self.state_obj.punts:
            if punt.ware is not None and punt.status == PuntStatus.ON_ROUTE and punt.position > SEA_ROUTE_LENGTH:
                key = self._first_available_dock_key(PuntStatus.IN_PORT, exclude_punt=punt)
                self._dock_punt(punt, PuntStatus.IN_PORT, key)
                punt.position = SEA_ROUTE_LENGTH

        summary = ", ".join(f"{w.value} {r}" for w, r in rolls.items()) or "no loaded punts"
        message = f"Dice rolled: {summary}."
        board_after = self.state_obj.movement_round_index == 2

        if self.state_obj.movement_round_index >= 3:
            # Voyage's third movement round is done -- resolve every punt
            # still at sea: exactly on 13 with pirates present is plundered;
            # exactly on 13 with no pirates still reaches port; anything
            # short of 13 is shipwrecked to the shipyard.
            for punt in self.state_obj.punts:
                if punt.ware is None or punt.status != PuntStatus.ON_ROUTE:
                    continue
                if punt.position == SEA_ROUTE_LENGTH:
                    pb = self.state_obj.pirate_boat
                    if pb.captain.occupant or pb.second.occupant:
                        self._resolve_plunder(punt)
                    else:
                        key = self._first_available_dock_key(PuntStatus.IN_PORT, exclude_punt=punt)
                        self._dock_punt(punt, PuntStatus.IN_PORT, key)
                else:
                    key = self._first_available_dock_key(PuntStatus.IN_SHIPYARD, exclude_punt=punt)
                    self._dock_punt(punt, PuntStatus.IN_SHIPYARD, key)

            paid = self._pay_port_shipyard_rewards() + self._pay_ware_profits() + self._pay_insurance_cost()
            risen = self._raise_ware_values_for_arrivals()
            self.state_obj.phase = Phase.PROFIT_DISTRIBUTION
            message += "\n\nThird movement round complete -- all punts have landed."
            if paid:
                totals: dict = {}
                for color, amount in paid:
                    totals[color] = totals.get(color, 0) + amount
                color_order = {p.color: i for i, p in enumerate(self.state_obj.players)}
                lines = [
                    f"{color}: {'+' if amt >= 0 else ''}{amt}"
                    for color, amt in sorted(totals.items(), key=lambda kv: color_order.get(kv[0], 999))
                ]
                message += "\n\nVoyage profit summary:\n" + "\n".join(lines)
            if risen:
                message += "\n\nWare values rise:\n" + "\n".join(risen)
            if self.state_obj.black_market.is_game_over():
                message += "\n\nGAME OVER -- a ware's value has reached 30!"

        messagebox.showinfo("Punts move", message)

        if board_after:
            self._handle_pirate_boarding()

    def _pay_port_shipyard_rewards(self) -> List[Tuple[str, int]]:
        """Port/shipyard accomplices are paid once, at the end of the third
        movement round, if a punt ended up docked in their exact slot."""
        paid: List[Tuple[str, int]] = []
        for dock, status in ((self.state_obj.port, PuntStatus.IN_PORT), (self.state_obj.shipyard, PuntStatus.IN_SHIPYARD)):
            for key, slot in dock.slots.items():
                if slot.occupant is None:
                    continue
                punt = next((p for p in self.state_obj.punts if p.status == status and p.dock_slot == key), None)
                if punt is None:
                    continue
                player = self.state_obj.player_by_id(slot.occupant)
                if player is not None:
                    player.cash += slot.payout
                    paid.append((player.color, slot.payout))
        return paid

    def _pay_ware_profits(self) -> List[Tuple[str, int]]:
        """Ware-punt accomplices split the cargo's value (the same base
        value used for pirate plunder) equally, but only if their punt
        actually reached the port -- shipwrecked or plundered punts pay
        nothing (plundered punts' accomplices are already cleared out).
        The slot counts (3/3/3/4) always divide these values evenly, so
        integer division keeps cash a whole number."""
        paid: List[Tuple[str, int]] = []
        for punt in self.state_obj.punts:
            if punt.ware is None or punt.status != PuntStatus.IN_PORT:
                continue
            occupied = [s for s in punt.ware_slots if s.occupant is not None]
            if not occupied:
                continue
            share = PLUNDER_PAYOUTS.get(punt.ware, 0) // len(occupied)
            for slot in occupied:
                player = self.state_obj.player_by_id(slot.occupant)
                if player is not None:
                    player.cash += share
                    paid.append((player.color, share))
        return paid

    def _pay_insurance_cost(self) -> List[Tuple[str, int]]:
        """The insurance-holder pays for shipyard repairs, a flat amount
        based on how many punts wrecked this voyage (1 -> 6, 2 -> 14,
        3 -> 29 PESOS) -- via credit if they can't afford it outright."""
        ins = self.state_obj.insurance
        if ins.occupant is None:
            return []
        wrecked = sum(1 for p in self.state_obj.punts if p.status == PuntStatus.IN_SHIPYARD)
        cost = INSURANCE_SHIPYARD_COST.get(wrecked, 0)
        if cost == 0:
            return []
        player = self.state_obj.player_by_id(ins.occupant)
        if player is None:
            return []
        self._settle_payment(player, cost)
        return [(player.color, -cost)]

    def _raise_ware_values_for_arrivals(self) -> List[str]:
        """Each ware whose punt reached port this voyage rises one step on
        the black market track (0-5-10-20-30)."""
        risen: List[str] = []
        for punt in self.state_obj.punts:
            if punt.ware is None or punt.status != PuntStatus.IN_PORT:
                continue
            before = self.state_obj.black_market.values[punt.ware]
            self.state_obj.black_market.raise_value(punt.ware)
            after = self.state_obj.black_market.values[punt.ware]
            if after != before:
                risen.append(f"{punt.ware.value.title()}: {before} -> {after}")
        return risen

    def _resolve_plunder(self, punt: Punt) -> None:
        payout = PLUNDER_PAYOUTS.get(punt.ware, 0)
        pb = self.state_obj.pirate_boat
        pirate_ids = [pid for pid in (pb.captain.occupant, pb.second.occupant) if pid]
        if pirate_ids:
            share = payout // len(pirate_ids)
            for pid in pirate_ids:
                player = self.state_obj.player_by_id(pid)
                if player is not None:
                    player.cash += share

        for slot in punt.ware_slots:
            slot.occupant = None  # accomplices are returned, empty-handed

        captain_player = self.state_obj.player_by_id(pb.captain.occupant) if pb.captain.occupant else None
        captain_name = captain_player.name if captain_player else "The pirates"
        send_to_port = messagebox.askyesno(
            "Pirates plunder!",
            f"Punt {punt.id} ({punt.ware.value}) was caught on space 13!\n"
            f"{captain_name} splits {payout} PESOS with the crew.\n\n"
            f"Send this punt to the PORT? (No sends it to the shipyard instead.)",
        )
        status = PuntStatus.IN_PORT if send_to_port else PuntStatus.IN_SHIPYARD
        key = self._first_available_dock_key(status, exclude_punt=punt)
        self._dock_punt(punt, status, key)

    # ------------------------------------------------------------------
    # Pirate boarding (right after the second movement round)
    # ------------------------------------------------------------------
    def _handle_pirate_boarding(self) -> None:
        """Punts caught on space 13 after round 2 (still ON_ROUTE) may be
        boarded for free by the pirates -- captain first, then the second
        pirate -- if a vacant ware-accomplice space is available on them."""
        candidates = [
            p
            for p in self.state_obj.punts
            if p.ware is not None and p.status == PuntStatus.ON_ROUTE and p.position == SEA_ROUTE_LENGTH
        ]
        if not candidates:
            return
        pb = self.state_obj.pirate_boat

        def after_captain() -> None:
            if pb.second.occupant:
                self._show_boarding_dialog(pb.second.occupant, "Second pirate", candidates, self.refresh)
            else:
                self.refresh()

        if pb.captain.occupant:
            self._show_boarding_dialog(pb.captain.occupant, "Pirate captain", candidates, after_captain)
        else:
            after_captain()

    def _show_boarding_dialog(
        self, pirate_player_id: str, role_label: str, candidates: List[Punt], then: Callable[[], None]
    ) -> None:
        player = self.state_obj.player_by_id(pirate_player_id)
        boardable = [p for p in candidates if any(s.occupant is None for s in p.ware_slots)]
        if player is None or not boardable:
            then()
            return

        dialog = tk.Toplevel(self.winfo_toplevel())
        dialog.title(f"{role_label}: {player.name} ({player.color})")
        dialog.transient(self.winfo_toplevel())
        dialog.grab_set()
        dialog.resizable(False, False)

        ttk.Label(
            dialog,
            text=f"{player.name} ({player.color}), {role_label.lower()}: a punt is stuck on space 13! "
            f"Board one for free (if it has room), or skip.",
            wraplength=320,
            justify=tk.LEFT,
            font=("Segoe UI", 10, "bold"),
        ).pack(padx=12, pady=(12, 8))

        def resolve() -> None:
            dialog.destroy()
            then()

        def board(punt: Punt) -> None:
            slot = next(s for s in punt.ware_slots if s.occupant is None)
            slot.occupant = player.id
            resolve()

        for punt in boardable:
            ttk.Button(
                dialog,
                text=f"Board Punt {punt.id} ({punt.ware.value.title()})",
                command=lambda p=punt: board(p),
            ).pack(padx=12, pady=2, fill=tk.X)
        ttk.Button(dialog, text="Skip", command=resolve).pack(pady=(8, 10))

    # ------------------------------------------------------------------
    # Pilots (just before the third movement round)
    # ------------------------------------------------------------------
    def _apply_pilot_move(self, punt: Punt, delta: int) -> None:
        if punt.status != PuntStatus.ON_ROUTE:
            return
        punt.position = max(0, punt.position + delta)
        if punt.position > SEA_ROUTE_LENGTH:
            key = self._first_available_dock_key(PuntStatus.IN_PORT, exclude_punt=punt)
            self._dock_punt(punt, PuntStatus.IN_PORT, key)
            punt.position = SEA_ROUTE_LENGTH

    def _eligible_pilot_punts(self) -> List[Punt]:
        return [p for p in self.state_obj.punts if p.ware is not None and p.status == PuntStatus.ON_ROUTE]

    def _show_pilot_dialogs(self, then: Callable[[], None]) -> None:
        small_slot = self.state_obj.pilot_island.small
        large_slot = self.state_obj.pilot_island.large

        def after_small() -> None:
            if large_slot.occupant:
                self._show_large_pilot_dialog(large_slot.occupant, then)
            else:
                then()

        if small_slot.occupant:
            self._show_small_pilot_dialog(small_slot.occupant, after_small)
        else:
            after_small()

    def _show_small_pilot_dialog(self, player_id: str, then: Callable[[], None]) -> None:
        player = self.state_obj.player_by_id(player_id)
        eligible = self._eligible_pilot_punts()
        if player is None or not eligible:
            then()
            return

        dialog = tk.Toplevel(self.winfo_toplevel())
        dialog.title(f"Small Pilot: {player.name} ({player.color})")
        dialog.transient(self.winfo_toplevel())
        dialog.grab_set()
        dialog.resizable(False, False)

        ttk.Label(
            dialog,
            text=f"{player.name} ({player.color}), small pilot: move ONE punt forward or backward by "
            f"1 space, or skip.",
            wraplength=320,
            justify=tk.LEFT,
            font=("Segoe UI", 10, "bold"),
        ).pack(padx=12, pady=(12, 8))

        options = [f"Punt {p.id} ({p.ware.value}, at space {p.position})" for p in eligible]
        punt_var = tk.StringVar(value=options[0])
        ttk.Combobox(dialog, textvariable=punt_var, values=options, state="readonly", width=28).pack(padx=12)

        def resolve() -> None:
            dialog.destroy()
            then()

        def move(direction: int) -> None:
            idx = options.index(punt_var.get())
            self._apply_pilot_move(eligible[idx], direction)
            resolve()

        btn_row = ttk.Frame(dialog)
        btn_row.pack(pady=10)
        ttk.Button(btn_row, text="Move forward (+1)", command=lambda: move(1)).pack(side=tk.LEFT, padx=4)
        ttk.Button(btn_row, text="Move backward (-1)", command=lambda: move(-1)).pack(side=tk.LEFT, padx=4)
        ttk.Button(btn_row, text="Skip", command=resolve).pack(side=tk.LEFT, padx=4)

    def _show_large_pilot_dialog(self, player_id: str, then: Callable[[], None]) -> None:
        player = self.state_obj.player_by_id(player_id)
        eligible = self._eligible_pilot_punts()
        if player is None or not eligible:
            then()
            return

        dialog = tk.Toplevel(self.winfo_toplevel())
        dialog.title(f"Large Pilot: {player.name} ({player.color})")
        dialog.transient(self.winfo_toplevel())
        dialog.grab_set()
        dialog.resizable(False, False)

        ttk.Label(
            dialog,
            text=f"{player.name} ({player.color}), large pilot: move ONE punt by 2 spaces, OR two "
            f"DIFFERENT punts by 1 space each, or skip.",
            wraplength=340,
            justify=tk.LEFT,
            font=("Segoe UI", 10, "bold"),
        ).pack(padx=12, pady=(12, 8))

        options = [f"Punt {p.id} ({p.ware.value}, at space {p.position})" for p in eligible]

        mode_var = tk.StringVar(value="one")
        mode_frame = ttk.Frame(dialog)
        mode_frame.pack(padx=12, pady=4, anchor="w")

        one_frame = ttk.Frame(dialog)
        two_frame = ttk.Frame(dialog)

        punt1_var = tk.StringVar(value=options[0])
        dir1_var = tk.StringVar(value="forward")
        ttk.Label(one_frame, text="Punt:").grid(row=0, column=0, sticky="w")
        ttk.Combobox(one_frame, textvariable=punt1_var, values=options, state="readonly", width=24).grid(
            row=0, column=1, padx=4
        )
        ttk.Radiobutton(one_frame, text="Forward (+2)", variable=dir1_var, value="forward").grid(
            row=1, column=0, columnspan=2, sticky="w"
        )
        ttk.Radiobutton(one_frame, text="Backward (-2)", variable=dir1_var, value="backward").grid(
            row=2, column=0, columnspan=2, sticky="w"
        )

        punt2a_var = tk.StringVar(value=options[0])
        dir2a_var = tk.StringVar(value="forward")
        punt2b_var = tk.StringVar(value=options[min(1, len(options) - 1)])
        dir2b_var = tk.StringVar(value="forward")
        ttk.Label(two_frame, text="Punt A:").grid(row=0, column=0, sticky="w")
        ttk.Combobox(two_frame, textvariable=punt2a_var, values=options, state="readonly", width=20).grid(
            row=0, column=1, padx=4
        )
        ttk.Combobox(two_frame, textvariable=dir2a_var, values=["forward", "backward"], state="readonly", width=9).grid(
            row=0, column=2
        )
        ttk.Label(two_frame, text="Punt B:").grid(row=1, column=0, sticky="w")
        ttk.Combobox(two_frame, textvariable=punt2b_var, values=options, state="readonly", width=20).grid(
            row=1, column=1, padx=4
        )
        ttk.Combobox(two_frame, textvariable=dir2b_var, values=["forward", "backward"], state="readonly", width=9).grid(
            row=1, column=2
        )

        error_var = tk.StringVar(value="")
        tk.Label(dialog, textvariable=error_var, fg="#b02a2a").pack(padx=12)

        def update_mode() -> None:
            error_var.set("")
            if mode_var.get() == "one":
                two_frame.pack_forget()
                one_frame.pack(padx=12, pady=6, anchor="w")
            else:
                one_frame.pack_forget()
                two_frame.pack(padx=12, pady=6, anchor="w")

        ttk.Radiobutton(
            mode_frame, text="Move one punt by 2 spaces", variable=mode_var, value="one", command=update_mode
        ).pack(anchor="w")
        ttk.Radiobutton(
            mode_frame, text="Move two punts by 1 space each", variable=mode_var, value="two", command=update_mode
        ).pack(anchor="w")

        def resolve() -> None:
            dialog.destroy()
            then()

        def on_confirm() -> None:
            if mode_var.get() == "one":
                idx = options.index(punt1_var.get())
                delta = 2 if dir1_var.get() == "forward" else -2
                self._apply_pilot_move(eligible[idx], delta)
            else:
                idx_a = options.index(punt2a_var.get())
                idx_b = options.index(punt2b_var.get())
                if idx_a == idx_b:
                    error_var.set("Pick two different punts.")
                    return
                delta_a = 1 if dir2a_var.get() == "forward" else -1
                delta_b = 1 if dir2b_var.get() == "forward" else -1
                self._apply_pilot_move(eligible[idx_a], delta_a)
                self._apply_pilot_move(eligible[idx_b], delta_b)
            resolve()

        action_row = ttk.Frame(dialog)
        action_row.pack(pady=10)
        ttk.Button(action_row, text="Move", command=on_confirm).pack(side=tk.LEFT, padx=4)
        ttk.Button(action_row, text="Skip", command=resolve).pack(side=tk.LEFT, padx=4)

        update_mode()

    def update_turn_label(self) -> None:
        player = self._current_turn_player()
        if player is None:
            self.turn_var.set("No players yet")
            self.turn_label.configure(bg=self.master.cget("bg") if hasattr(self.master, "cget") else "#f0f0f0")
            return
        self.turn_var.set(
            f"Turn: {player.name} ({player.color})  -  round {self.state_obj.movement_round_index + 1} of 3"
        )
        self.turn_label.configure(bg=player.color, fg="white" if player.color in ("red", "blue", "black") else "black")

    def _draw_accomplice_circle(
        self,
        cx: float,
        cy: float,
        radius: float,
        occupant_id: Optional[str],
        click_cb: Optional[Callable[[], None]] = None,
        ring_color: str = "#333333",
        ring_width: int = 2,
    ) -> None:
        """An accomplice placement space: a circle filled with the occupying
        player's color, or white when vacant."""
        fill = self._player_color(occupant_id) if occupant_id else VACANT_FILL
        self.canvas.create_oval(
            cx - radius, cy - radius, cx + radius, cy + radius, fill=fill, outline=ring_color, width=ring_width
        )
        if click_cb is not None:
            self._register_click(cx - radius, cy - radius, cx + radius, cy + radius, click_cb)

    def _draw_ware_slots_strip(
        self, punt: Punt, anchor_x: float, anchor_y: float, radius: float, spacing: float, interactive: bool = True
    ) -> None:
        """The punt's ware-accomplice slots, centered under (anchor_x, anchor_y).

        Used both on the sea lane (following the punt as it moves) and, once
        docked, below its boat-dock rectangle -- so accomplices always stay
        visibly attached to their punt. Once a punt has arrived (docked),
        `interactive=False` locks these spaces: no click, no price shown,
        and a vacant space is greyed out to show it can no longer be bought.
        """
        n = len(punt.ware_slots)
        if n == 0:
            return
        start_x = anchor_x - (n - 1) * spacing / 2
        for slot_idx, slot in enumerate(punt.ware_slots):
            sx = start_x + slot_idx * spacing

            if interactive:
                def make_cb(p=punt, s=slot):
                    return lambda: self._place_or_remove_punt_accomplice(p, s)

                self._draw_accomplice_circle(sx, anchor_y, radius, slot.occupant, make_cb())
                if slot.occupant is None:
                    self.canvas.create_text(sx, anchor_y, text=str(slot.price), font=("Segoe UI", 7, "bold"))
            else:
                # Locked: the punt has already arrived, so no clicking, no
                # price, and a vacant space is greyed out (can't be bought).
                fill = self._player_color(slot.occupant) if slot.occupant else "#e5e5e5"
                self.canvas.create_oval(
                    sx - radius, anchor_y - radius, sx + radius, anchor_y + radius, fill=fill, outline="#999999"
                )

    def _draw_sea_lanes(self) -> None:
        c = self.canvas
        for lane_idx, punt in enumerate(self.state_obj.punts):
            y = LANE_TOP_Y + lane_idx * LANE_GAP
            ware_label = punt.ware.value.title() if punt.ware else "empty"
            ware_color = WARE_COLORS.get(punt.ware, "#888888")
            c.create_text(LANE_LEFT_X - 90, y, text=f"Punt {punt.id}\n({ware_label})", fill=ware_color, width=90)

            for pos in range(SEA_ROUTE_LENGTH + 1):
                x = LANE_LEFT_X + pos * CELL_W
                fill = "#ffffff"
                if pos <= MAX_START_SPACE:
                    fill = "#eef7ff"
                if pos == SEA_ROUTE_LENGTH:
                    fill = "#f7d9a0"  # pirate danger space
                rect = c.create_rectangle(x, y - 16, x + CELL_W - 4, y + 16, fill=fill, outline="#7a9cb5")
                c.create_text(x + (CELL_W - 4) / 2, y - 24, text=str(pos), font=("Segoe UI", 7), fill="#555")

                def make_cb(lane=lane_idx, position=pos):
                    return lambda: self._set_punt_position(lane, position)

                self._register_click(x, y - 16, x + CELL_W - 4, y + 16, make_cb())

            if punt.status == PuntStatus.ON_ROUTE:
                punt_x = LANE_LEFT_X + punt.position * CELL_W + (CELL_W - 4) / 2
                c.create_oval(
                    punt_x - PUNT_RADIUS,
                    y - PUNT_RADIUS,
                    punt_x + PUNT_RADIUS,
                    y + PUNT_RADIUS,
                    fill=ware_color,
                    outline="black",
                )
                if punt.ware is not None:
                    # The plunder payout if pirates catch this punt on space 13.
                    c.create_text(
                        punt_x,
                        y,
                        text=str(PLUNDER_PAYOUTS.get(punt.ware, 0)),
                        font=("Segoe UI", 7, "bold"),
                        fill=WARE_TEXT_COLORS.get(punt.ware, "black"),
                    )
                # Ware accomplice slots ride along under the punt as it moves.
                # Once docked, they're drawn under its boat-dock rectangle
                # instead (see _draw_docks).
                self._draw_ware_slots_strip(punt, punt_x, y + 32, SLOT_RADIUS, 30)

    def _set_punt_position(self, lane: int, position: int) -> None:
        if self.state_obj.game_setup_confirmed:
            return  # punts move only via the game's own flow once it has started
        punt = self.state_obj.punts[lane]
        punt.position = position
        punt.status = PuntStatus.ON_ROUTE
        punt.dock_slot = None

    def _draw_docks(self) -> None:
        c = self.canvas
        base_x = LANE_LEFT_X + (SEA_ROUTE_LENGTH + 1) * CELL_W + 30

        # Layout, left to right: [Port accomplice circles] [Port boat docks]
        # [Shipyard boat docks] [Shipyard accomplice circles]. The boat-dock
        # rectangles hold the actual punt token; the circles are just for the
        # accomplice placed on that space.
        port_circle_x = base_x
        port_rect_x = base_x + DOCK_COL_GAP
        shipyard_rect_x = port_rect_x + DOCK_RECT_W + DOCK_MID_GAP
        shipyard_circle_x = shipyard_rect_x + DOCK_COL_GAP

        groups = [
            ("Port", self.state_obj.port, PuntStatus.IN_PORT, port_circle_x, port_rect_x),
            ("Shipyard", self.state_obj.shipyard, PuntStatus.IN_SHIPYARD, shipyard_circle_x, shipyard_rect_x),
        ]

        for group_idx, (title, dock, status_when_docked, circle_x, rect_x) in enumerate(groups):
            c.create_text(
                (circle_x + rect_x) / 2, LANE_TOP_Y + DOCK_TITLE_DY, text=title, font=("Segoe UI", 10, "bold")
            )

            for row_idx, key in enumerate(["A", "B", "C"]):
                slot = dock.slots[key]
                y = LANE_TOP_Y + row_idx * DOCK_ROW_H
                docked_punt = next(
                    (p for p in self.state_obj.punts if p.status == status_when_docked and p.dock_slot == key), None
                )

                c.create_text(circle_x, y + DOCK_LABEL_DY, text=f"{key} (+{slot.payout})", font=("Segoe UI", 8))

                def make_occ_cb(s=slot, d=dock):
                    return lambda: self._place_or_remove_dock_accomplice(d, s)

                self._draw_accomplice_circle(circle_x, y, DOCK_RADIUS, slot.occupant, make_occ_cb())
                self._draw_slot_price_if_vacant(circle_x, y, slot)

                # Boat-dock rectangle: the physical punt (if any) sitting here.
                fill = WARE_COLORS.get(docked_punt.ware, "#f4f4f4") if docked_punt else "#f4f4f4"
                outline = "black" if docked_punt else "#999999"
                c.create_rectangle(
                    rect_x - DOCK_RECT_W / 2,
                    y - DOCK_RECT_H / 2,
                    rect_x + DOCK_RECT_W / 2,
                    y + DOCK_RECT_H / 2,
                    fill=fill,
                    outline=outline,
                    width=2,
                )
                def make_dock_cb(k=key, grp=group_idx):
                    return lambda: self._cycle_dock(grp, k)

                self._register_click(
                    rect_x - DOCK_RECT_W / 2,
                    y - DOCK_RECT_H / 2,
                    rect_x + DOCK_RECT_W / 2,
                    y + DOCK_RECT_H / 2,
                    make_dock_cb(),
                )

                if docked_punt is not None:
                    # Ware accomplice slots keep riding along with the punt
                    # after it docks, now shown right inside its boat
                    # rectangle -- locked, since the punt has already arrived.
                    self._draw_ware_slots_strip(
                        docked_punt, rect_x, y, MINI_SLOT_RADIUS, DOCK_MINI_SPACING, interactive=False
                    )

    def _cycle_dock(self, group_idx: int, key: str) -> None:
        status = PuntStatus.IN_PORT if group_idx == 0 else PuntStatus.IN_SHIPYARD
        dock = self.state_obj.port if group_idx == 0 else self.state_obj.shipyard

        current = next((p for p in self.state_obj.punts if p.status == status and p.dock_slot == key), None)
        candidates = [None] + list(self.state_obj.punts)
        idx = candidates.index(current) if current in candidates else 0
        idx = (idx + 1) % len(candidates)
        nxt = candidates[idx]

        while nxt is not None and (nxt.status == status and nxt.dock_slot != key):
            idx = (idx + 1) % len(candidates)
            nxt = candidates[idx]
            if nxt is current:
                break

        if current is not None:
            current.status = PuntStatus.ON_ROUTE
            current.dock_slot = None
            current.position = SEA_ROUTE_LENGTH

        if nxt is not None:
            self._dock_punt(nxt, status, key)

    def _first_available_dock_key(self, status: PuntStatus, exclude_punt: Optional[Punt] = None) -> str:
        """The highest-priority vacant port/shipyard slot: A, then B, then C
        -- matching the rule that the first punt to arrive takes space A, the
        second takes B, and the third takes C."""
        for key in ["A", "B", "C"]:
            occupant = next(
                (p for p in self.state_obj.punts if p is not exclude_punt and p.status == status and p.dock_slot == key),
                None,
            )
            if occupant is None:
                return key
        return "A"  # only reachable if all 3 punts are already docked here

    def _dock_punt(self, punt: Punt, status: PuntStatus, key: str) -> None:
        other = next(
            (p for p in self.state_obj.punts if p is not punt and p.status == status and p.dock_slot == key), None
        )
        if other is not None:
            other.status = PuntStatus.ON_ROUTE
            other.dock_slot = None
            other.position = SEA_ROUTE_LENGTH
        punt.status = status
        punt.dock_slot = key

    def _draw_slot_price_if_vacant(self, cx: float, cy: float, slot: AccompliceSlot) -> None:
        if slot.occupant is None:
            self.canvas.create_text(cx, cy, text=str(slot.price), font=("Segoe UI", 8, "bold"), fill="#333333")

    def _draw_pirate_boat(self) -> None:
        c = self.canvas
        c.create_text(PIRATE_X + 35, MISC_ROW_Y + MISC_TITLE_DY, text="Pirate boat", font=("Segoe UI", 10, "bold"))
        for i, (label, slot) in enumerate(
            [("Captain", self.state_obj.pirate_boat.captain), ("Second", self.state_obj.pirate_boat.second)]
        ):
            cx = PIRATE_X + i * 70
            c.create_text(cx, MISC_ROW_Y + MISC_LABEL_DY, text=label, font=("Segoe UI", 8))

            def make_cb(s=slot, is_second=(i == 1)):
                return lambda: self._place_or_remove_pirate_slot(s, is_second)

            self._draw_accomplice_circle(cx, MISC_ROW_Y, BIG_SLOT_RADIUS, slot.occupant, make_cb())
            self._draw_slot_price_if_vacant(cx, MISC_ROW_Y, slot)

    def _place_or_remove_pirate_slot(self, slot: AccompliceSlot, is_second: bool) -> None:
        """The second pirate space can't be filled until the captain's is."""
        if is_second and slot.occupant is None and self.state_obj.pirate_boat.captain.occupant is None:
            return
        self._place_or_remove_accomplice(slot)

    def _draw_pilot_island(self) -> None:
        c = self.canvas
        c.create_text(PILOT_X + 35, MISC_ROW_Y + MISC_TITLE_DY, text="Pilot island", font=("Segoe UI", 10, "bold"))
        for i, (label, slot) in enumerate(
            [("Small", self.state_obj.pilot_island.small), ("Large", self.state_obj.pilot_island.large)]
        ):
            cx = PILOT_X + i * 70
            c.create_text(cx, MISC_ROW_Y + MISC_LABEL_DY, text=label, font=("Segoe UI", 8))

            def make_cb(s=slot):
                return lambda: self._place_or_remove_accomplice(s)

            self._draw_accomplice_circle(cx, MISC_ROW_Y, BIG_SLOT_RADIUS, slot.occupant, make_cb())
            self._draw_slot_price_if_vacant(cx, MISC_ROW_Y, slot)

    def _draw_insurance(self) -> None:
        c = self.canvas
        ins = self.state_obj.insurance
        c.create_text(INSURANCE_X, MISC_ROW_Y + MISC_TITLE_DY, text="Insurance", font=("Segoe UI", 10, "bold"))
        c.create_text(INSURANCE_X, MISC_ROW_Y + MISC_LABEL_DY, text=f"(+{ins.payment})", font=("Segoe UI", 8))

        self._draw_accomplice_circle(
            INSURANCE_X, MISC_ROW_Y, BIG_SLOT_RADIUS, ins.occupant, self._place_or_remove_insurance
        )
        if ins.occupant is None:
            # Insurance always costs 0 to place -- you're paid to take it.
            c.create_text(INSURANCE_X, MISC_ROW_Y, text="0", font=("Segoe UI", 8, "bold"), fill="#333333")

    # ------------------------------------------------------------------
    # Game-flow dialogs: new-game setup -> auction -> harbor master load/place
    # ------------------------------------------------------------------
    def _show_new_game_setup_dialog(self) -> None:
        dialog = tk.Toplevel(self.winfo_toplevel())
        dialog.title("New Game Setup")
        dialog.transient(self.winfo_toplevel())
        dialog.grab_set()
        dialog.resizable(False, False)

        ttk.Label(dialog, text="Number of players:", font=("Segoe UI", 10, "bold")).pack(
            anchor="w", padx=12, pady=(12, 4)
        )
        count_var = tk.IntVar(value=max(4, min(5, self.state_obj.player_count or 4)))

        rows_frame = ttk.Frame(dialog)
        color_vars: List[tk.StringVar] = []

        error_var = tk.StringVar(value="")

        def rebuild_rows() -> None:
            for child in rows_frame.winfo_children():
                child.destroy()
            color_vars.clear()
            try:
                count = int(count_var.get())
            except (tk.TclError, ValueError):
                count = 4
            count = max(4, min(5, count))
            count_var.set(count)
            for i in range(count):
                row = ttk.Frame(rows_frame)
                row.pack(side=tk.TOP, fill=tk.X, pady=2)
                ttk.Label(row, text=f"Player {i + 1}:", width=10).pack(side=tk.LEFT)
                default_color = DEFAULT_PLAYER_COLORS[i % len(DEFAULT_PLAYER_COLORS)]
                var = tk.StringVar(value=default_color)
                combo = ttk.Combobox(
                    row, textvariable=var, values=DEFAULT_PLAYER_COLORS, width=8, state="readonly"
                )
                combo.pack(side=tk.LEFT)
                color_vars.append(var)

        count_spin = ttk.Spinbox(dialog, from_=4, to=5, width=4, textvariable=count_var, command=rebuild_rows)
        count_spin.pack(anchor="w", padx=12)
        count_spin.bind("<Return>", lambda e: rebuild_rows())
        count_spin.bind("<FocusOut>", lambda e: rebuild_rows())

        rows_frame.pack(fill=tk.X, padx=12, pady=8)

        tk.Label(dialog, textvariable=error_var, fg="#b02a2a").pack(padx=12)

        def on_confirm() -> None:
            colors = [v.get() for v in color_vars]
            if len(set(colors)) != len(colors):
                error_var.set("Each player needs a distinct color.")
                return
            names = [f"Player {i + 1}" for i in range(len(colors))]
            self.state_obj = GameState.new_default_game(names, colors=colors)
            self.state_obj.game_setup_confirmed = True
            self._round_placements = 0
            dialog.destroy()
            self.refresh()
            self._show_auction_dialog()

        ttk.Button(dialog, text="Confirm", command=on_confirm).pack(pady=(4, 12))

        rebuild_rows()

    def _show_auction_dialog(self) -> None:
        players = self.state_obj.players
        if not players:
            return

        dialog = tk.Toplevel(self.winfo_toplevel())
        dialog.title(f"Voyage {self.state_obj.voyage_number} - Harbor Master Auction")
        dialog.transient(self.winfo_toplevel())
        dialog.grab_set()
        dialog.resizable(False, False)

        # The previous harbor master starts the bidding (or player 0, for the
        # very first voyage); bidding then proceeds in player-list order.
        start_player = next((p for p in players if p.is_harbor_master), players[0])
        start_idx = players.index(start_player)
        order = players[start_idx:] + players[:start_idx]

        auction = {"highest_bid": 0, "highest_bidder": None, "active": list(order), "turn_idx": 0}

        ttk.Label(
            dialog, text=f"Voyage {self.state_obj.voyage_number}: Harbor Master Auction", font=("Segoe UI", 11, "bold")
        ).pack(padx=12, pady=(12, 6))

        status_var = tk.StringVar()
        tk.Label(dialog, textvariable=status_var, font=("Segoe UI", 10, "bold"), justify=tk.LEFT, anchor="w").pack(
            fill=tk.X, padx=12
        )

        log_lines: List[str] = []
        log_var = tk.StringVar(value="")
        tk.Label(dialog, textvariable=log_var, fg="#555", justify=tk.LEFT, anchor="w", wraplength=320).pack(
            fill=tk.X, padx=12, pady=(6, 6)
        )

        bid_row = ttk.Frame(dialog)
        bid_row.pack(padx=12, pady=6)
        ttk.Label(bid_row, text="Bid amount:").pack(side=tk.LEFT)
        bid_var = tk.IntVar(value=1)
        ttk.Spinbox(bid_row, from_=1, to=9999, width=6, textvariable=bid_var).pack(side=tk.LEFT, padx=6)
        ttk.Button(bid_row, text="Bid", command=lambda: on_bid()).pack(side=tk.LEFT, padx=4)
        ttk.Button(bid_row, text="Pass", command=lambda: on_pass()).pack(side=tk.LEFT, padx=4)

        def log(line: str) -> None:
            log_lines.append(line)
            log_var.set("\n".join(log_lines[-6:]))

        def current_player() -> Player:
            return auction["active"][auction["turn_idx"] % len(auction["active"])]

        def refresh_status() -> None:
            holder = auction["highest_bidder"]
            holder_text = f"{holder.name} ({holder.color})" if holder else "no bids yet"
            cp = current_player()
            status_var.set(f"Highest bid: {auction['highest_bid']} PESOS - {holder_text}\n\n{cp.name}'s turn ({cp.color})")
            bid_var.set(auction["highest_bid"] + 1)

        def end_auction() -> None:
            winner = auction["active"][0] if auction["active"] else None
            if winner is not None and auction["highest_bid"] > 0:
                for p in self.state_obj.players:
                    p.is_harbor_master = p.id == winner.id
                self._settle_payment(winner, auction["highest_bid"])
                log(f"{winner.name} wins the auction for {auction['highest_bid']} PESOS and becomes harbor master.")
            else:
                log("No bids were placed -- the harbor master does not change.")
            dialog.destroy()
            self.refresh()
            if winner is not None and auction["highest_bid"] > 0:
                self._show_buy_share_dialog(winner, self._show_load_and_place_dialog)
            else:
                self._show_load_and_place_dialog()

        def on_bid() -> None:
            cp = current_player()
            try:
                amount = int(bid_var.get())
            except (tk.TclError, ValueError):
                log("Enter a valid bid amount.")
                return
            if amount <= auction["highest_bid"]:
                log(f"Bid must be higher than {auction['highest_bid']}.")
                return
            affordable = cp.cash + SHARE_LOAN_AMOUNT * len(cp.unencumbered_shares)
            if amount > affordable:
                log(f"{cp.name} cannot raise more than {affordable} PESOS (cash + credit).")
                return
            auction["highest_bid"] = amount
            auction["highest_bidder"] = cp
            log(f"{cp.name} bids {amount}.")
            auction["turn_idx"] = (auction["turn_idx"] + 1) % len(auction["active"])
            refresh_status()

        def on_pass() -> None:
            cp = current_player()
            log(f"{cp.name} passes.")
            idx = auction["active"].index(cp)
            auction["active"].pop(idx)
            if len(auction["active"]) <= 1:
                end_auction()
                return
            auction["turn_idx"] = idx if idx < len(auction["active"]) else 0
            refresh_status()

        refresh_status()

    def _show_buy_share_dialog(self, player: Player, then: Callable[[], None]) -> None:
        market = self.state_obj.black_market
        available = [w for w in Ware if self.state_obj.shares_available(w) > 0]
        if not available:
            then()
            return

        dialog = tk.Toplevel(self.winfo_toplevel())
        dialog.title(f"Harbor Master: {player.name} - Buy a Share?")
        dialog.transient(self.winfo_toplevel())
        dialog.grab_set()
        dialog.resizable(False, False)

        ttk.Label(
            dialog,
            text=f"{player.name} ({player.color}) may buy one share (price = the ware's current "
            f"black market value, minimum 5 PESOS).",
            wraplength=340,
            justify=tk.LEFT,
            font=("Segoe UI", 10, "bold"),
        ).pack(padx=12, pady=(12, 8))

        ware_var = tk.StringVar(value=available[0].value)
        for ware in available:
            price = market.share_price(ware)
            left = self.state_obj.shares_available(ware)
            ttk.Radiobutton(
                dialog,
                text=f"{ware.value.title()} - {price} PESOS ({left} left)",
                variable=ware_var,
                value=ware.value,
            ).pack(anchor="w", padx=12)

        def resolve() -> None:
            dialog.destroy()
            then()

        def on_buy() -> None:
            ware = Ware(ware_var.get())
            price = market.share_price(ware)
            if not self._settle_payment(player, price):
                messagebox.showinfo("Buy share", f"{player.name} doesn't have enough cash or credit for this share.")
                return
            player.shares.append(Share(ware=ware))
            self.refresh()  # reflect immediately in the shares-available panel and the player's card
            resolve()

        btn_row = ttk.Frame(dialog)
        btn_row.pack(pady=10)
        ttk.Button(btn_row, text="Buy", command=on_buy).pack(side=tk.LEFT, padx=4)
        ttk.Button(btn_row, text="Skip", command=resolve).pack(side=tk.LEFT, padx=4)

    def _show_load_and_place_dialog(self) -> None:
        harbor_master = next((p for p in self.state_obj.players if p.is_harbor_master), None)
        if harbor_master is None:
            return  # no bids were placed, so nothing to load/place yet

        dialog = tk.Toplevel(self.winfo_toplevel())
        dialog.title(f"Harbor Master: {harbor_master.name}")
        dialog.transient(self.winfo_toplevel())
        dialog.grab_set()
        dialog.resizable(False, False)

        ttk.Label(
            dialog,
            text=f"{harbor_master.name} ({harbor_master.color}) - load 3 wares and set their start "
            f"positions (0-{MAX_START_SPACE} each, summing to {PUNT_START_SUM})",
            font=("Segoe UI", 10, "bold"),
            wraplength=340,
            justify=tk.LEFT,
        ).pack(padx=12, pady=(12, 8))

        rows_frame = ttk.Frame(dialog)
        rows_frame.pack(padx=12, pady=4, fill=tk.X)

        load_vars: dict = {}
        pos_vars: dict = {}
        pos_spins: dict = {}

        status_var = tk.StringVar()
        tk.Label(dialog, textvariable=status_var, fg="#333").pack(padx=12, pady=(4, 0))
        error_var = tk.StringVar(value="")
        tk.Label(dialog, textvariable=error_var, fg="#b02a2a").pack(padx=12)

        def update_status() -> None:
            loaded = [w for w in Ware if load_vars[w].get()]
            total = 0
            for w in loaded:
                try:
                    total += int(pos_vars[w].get())
                except (tk.TclError, ValueError):
                    pass
            for w in Ware:
                pos_spins[w].configure(state=("normal" if load_vars[w].get() else "disabled"))
            status_var.set(f"Loaded: {len(loaded)}/3 wares.  Sum of start positions: {total}/{PUNT_START_SUM}.")
            confirm_btn.configure(
                state=("normal" if len(loaded) == 3 and total == PUNT_START_SUM else "disabled")
            )

        for ware in Ware:
            row = ttk.Frame(rows_frame)
            row.pack(side=tk.TOP, fill=tk.X, pady=2)
            load_var = tk.BooleanVar(value=False)
            ttk.Checkbutton(
                row, text=ware.value.title(), variable=load_var, command=update_status
            ).pack(side=tk.LEFT, ipadx=20)
            pos_var = tk.IntVar(value=0)
            pos_var.trace_add("write", lambda *_: update_status())
            spin = ttk.Spinbox(
                row, from_=0, to=MAX_START_SPACE, width=4, textvariable=pos_var, state="disabled", command=update_status
            )
            spin.pack(side=tk.LEFT, padx=8)
            load_vars[ware] = load_var
            pos_vars[ware] = pos_var
            pos_spins[ware] = spin

        def on_confirm() -> None:
            loaded = [w for w in Ware if load_vars[w].get()]
            total = sum(int(pos_vars[w].get()) for w in loaded)
            if len(loaded) != 3 or total != PUNT_START_SUM:
                error_var.set(f"Pick exactly 3 wares with start positions summing to {PUNT_START_SUM}.")
                return
            unloaded = next(w for w in Ware if w not in loaded)
            for punt, ware in zip(self.state_obj.punts, loaded):
                punt.ware = ware
                punt.position = int(pos_vars[ware].get())
                punt.status = PuntStatus.ON_ROUTE
                punt.dock_slot = None
                punt.ware_slots = Punt.new(punt.id, ware).ware_slots
            self.state_obj.unloaded_ware = unloaded
            self.state_obj.phase = Phase.ACCOMPLICE_ROUND
            self.state_obj.current_turn_player_id = harbor_master.id
            self._round_placements = 0
            dialog.destroy()
            self.refresh()

        confirm_btn = ttk.Button(dialog, text="Confirm", command=on_confirm, state="disabled")
        confirm_btn.pack(pady=(4, 12))

        update_status()

    # ------------------------------------------------------------------
    # Toolbar handlers
    # ------------------------------------------------------------------
    def on_new_default_game(self) -> None:
        self._show_new_game_setup_dialog()

    def on_randomize(self) -> None:
        rng = random.Random()
        state = self.state_obj

        wares = list(Ware)
        rng.shuffle(wares)
        unloaded = wares[0]
        loaded = wares[1:]
        state.unloaded_ware = unloaded
        for punt, ware in zip(state.punts, loaded):
            punt.ware = ware
            punt.status = PuntStatus.ON_ROUTE
            punt.dock_slot = None

        # random start positions 0-5 summing to 9
        for _ in range(200):
            a = rng.randint(0, MAX_START_SPACE)
            b = rng.randint(0, MAX_START_SPACE)
            c_ = PUNT_START_SUM - a - b
            if 0 <= c_ <= MAX_START_SPACE:
                positions = [a, b, c_]
                break
        else:
            positions = [4, 3, 2]
        rng.shuffle(positions)
        for punt, pos in zip(state.punts, positions):
            punt.position = pos

        for ware in Ware:
            state.black_market.values[ware] = rng.choice(BLACK_MARKET_LEVELS[:-1])  # never randomize a finished game

        for player in state.players:
            player.cash = rng.randint(0, 60)

        self.refresh()

    def on_start_next_voyage(self) -> None:
        state = self.state_obj
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

        # All accomplices return to their players between voyages.
        for key in ("A", "B", "C"):
            state.port.slots[key].occupant = None
            state.shipyard.slots[key].occupant = None
        state.pirate_boat.captain.occupant = None
        state.pirate_boat.second.occupant = None
        state.pilot_island.small.occupant = None
        state.pilot_island.large.occupant = None
        state.insurance.occupant = None

        state.phase = Phase.AUCTION
        self._round_placements = 0
        self.refresh()
        self._show_auction_dialog()

    def on_save(self) -> None:
        path = filedialog.asksaveasfilename(defaultextension=".json", filetypes=[("JSON", "*.json")])
        if not path:
            return
        self.state_obj.save(path)

    def on_load(self) -> None:
        path = filedialog.askopenfilename(filetypes=[("JSON", "*.json")])
        if not path:
            return
        try:
            self.state_obj = GameState.load(path)
        except Exception as exc:  # noqa: BLE001 - surface any load error to the user
            messagebox.showerror("Load failed", str(exc))
            return
        self._round_placements = 0
        self.refresh()

    def on_voyage_change(self) -> None:
        self.state_obj.voyage_number = self.voyage_var.get()

    def on_phase_change(self) -> None:
        self.state_obj.phase = Phase(self.phase_var.get())

    def on_market_change(self, ware: Ware) -> None:
        self.state_obj.black_market.values[ware] = self.market_vars[ware].get()
        self.draw_canvas()
        self.update_warnings()

    def on_unloaded_change(self) -> None:
        val = self.unloaded_var.get()
        self.state_obj.unloaded_ware = None if val == "(none)" else Ware(val)
        self.draw_canvas()
        self.update_warnings()

    def on_punt_row_change(self, idx: int) -> None:
        row = self.punt_rows[idx]
        punt = self.state_obj.punts[idx]

        ware_val = row["ware"].get()
        new_ware = None if ware_val in ("", "(none)") else Ware(ware_val)
        if new_ware != punt.ware:
            punt.ware = new_ware
            punt.ware_slots = Punt.new(punt.id, new_ware).ware_slots

        new_status = PuntStatus(row["status"].get())
        try:
            punt.position = int(row["position"].get())
        except (tk.TclError, ValueError):
            pass

        if new_status in (PuntStatus.IN_PORT, PuntStatus.IN_SHIPYARD):
            # Arriving at port/shipyard always fills the highest-priority
            # vacant slot (A, then B, then C) automatically.
            key = self._first_available_dock_key(new_status, exclude_punt=punt)
            self._dock_punt(punt, new_status, key)
        else:
            punt.status = new_status
            punt.dock_slot = None

        self.draw_canvas()
        self.update_warnings()

    # ------------------------------------------------------------------
    # Player panel
    # ------------------------------------------------------------------
    def rebuild_players_panel(self) -> None:
        for child in self.players_container.winfo_children():
            child.destroy()

        for player in self.state_obj.players:
            self._build_player_card(player)

    def _build_player_card(self, player: Player) -> None:
        card = ttk.LabelFrame(self.players_container, text=player.name)
        card.pack(side=tk.TOP, fill=tk.X, pady=2, padx=2)

        # Shares column is packed first, on the right, so it claims the full
        # card height and sits flush with the top -- alongside name/color/
        # cash on the left, not just alongside the body row below the name.
        right_col = ttk.Frame(card)
        right_col.pack(side=tk.RIGHT, anchor="n", padx=(10, 4), pady=(4, 0))

        ttk.Label(right_col, text="Shares (unenc/enc):").pack(side=tk.TOP, anchor="w")

        # Shares are read-only here -- they only change via legitimate game
        # actions (initial deal, the harbor master buying one, taking
        # credit), never by hand.
        for ware in Ware:
            wrow = ttk.Frame(right_col)
            wrow.pack(side=tk.TOP, fill=tk.X)
            ttk.Label(wrow, text=ware.value.title(), width=7, foreground=WARE_COLORS[ware]).pack(side=tk.LEFT)

            unenc = sum(1 for s in player.shares if s.ware == ware and not s.encumbered)
            enc = sum(1 for s in player.shares if s.ware == ware and s.encumbered)
            ttk.Label(wrow, text=f"{unenc} / {enc}", width=6).pack(side=tk.LEFT, padx=2)

        name_row = ttk.Frame(card)
        name_row.pack(side=tk.TOP, fill=tk.X, pady=2)

        name_var = tk.StringVar(value=player.name)
        name_entry = ttk.Entry(name_row, textvariable=name_var, width=10)
        name_entry.pack(side=tk.LEFT)

        def on_name_change(*_ , p=player, var=name_var):
            p.name = var.get()
            card.configure(text=p.name)

        name_var.trace_add("write", on_name_change)

        # Left column: color, cash, harbor-master badge -- fills the space
        # remaining next to the shares column.
        color_row = ttk.Frame(card)
        color_row.pack(side=tk.TOP, fill=tk.X, pady=1)
        color_var = tk.StringVar(value=player.color)
        color_state = "disabled" if self.state_obj.game_setup_confirmed else "readonly"
        color_combo = ttk.Combobox(
            color_row, textvariable=color_var, values=DEFAULT_PLAYER_COLORS, width=7, state=color_state
        )
        color_combo.pack(side=tk.LEFT)

        def on_color_change(e=None, p=player, var=color_var):
            p.color = var.get()
            self.draw_canvas()

        color_combo.bind("<<ComboboxSelected>>", on_color_change)

        cash_row = ttk.Frame(card)
        cash_row.pack(side=tk.TOP, fill=tk.X, pady=1)
        ttk.Label(cash_row, text="Cash:").pack(side=tk.LEFT)
        cash_var = tk.IntVar(value=player.cash)
        cash_state = "disabled" if self.state_obj.game_setup_confirmed else "normal"
        cash_spin = ttk.Spinbox(cash_row, from_=0, to=999, width=4, textvariable=cash_var, state=cash_state)
        cash_spin.pack(side=tk.LEFT, padx=(2, 0))

        def on_cash_change(*_ , p=player, var=cash_var):
            try:
                p.cash = var.get()
            except tk.TclError:
                pass

        cash_var.trace_add("write", on_cash_change)

        # Harbor master is decided exclusively by the auction dialog now, not
        # hand-picked here -- this is a read-only status indicator.
        if player.is_harbor_master:
            ttk.Label(card, text="★ Harbor master", foreground="#b8860b").pack(side=tk.TOP, anchor="w", pady=(2, 0))

        # Remove now lives at the bottom of the left column (swapped with
        # shares, which claims the top-right instead) so it's out of the way
        # of the everyday info at a glance.
        remove_btn = ttk.Button(card, text="Remove", width=8, command=lambda p=player: self.on_remove_player(p))
        remove_btn.pack(side=tk.TOP, anchor="w", pady=(4, 2))
        if self.state_obj.game_setup_confirmed:
            remove_btn.configure(state="disabled")

    def on_add_player(self) -> None:
        if self.state_obj.player_count >= 5:
            messagebox.showinfo("Manilla", "Manilla supports at most 5 players.")
            return
        idx = self.state_obj.player_count
        color = DEFAULT_PLAYER_COLORS[idx % len(DEFAULT_PLAYER_COLORS)]
        self.state_obj.players.append(
            Player(id=f"p{idx}", name=f"Player {idx + 1}", color=color, cash=STARTING_CASH)
        )
        self.refresh()

    def on_remove_player(self, player: Player) -> None:
        if self.state_obj.player_count <= 4:
            messagebox.showinfo("Manilla", "This setup tool assumes at least 4 players.")
            return
        self.state_obj.players = [p for p in self.state_obj.players if p.id != player.id]
        self.refresh()


def launch(state: Optional[GameState] = None) -> None:
    root = tk.Tk()
    root.title("Manilla - Board Setup")
    root.geometry("1180x650")
    root.minsize(820, 480)
    app = BoardSetupApp(root, state=state)
    app.pack(fill=tk.BOTH, expand=True)
    root.mainloop()


if __name__ == "__main__":
    launch()
