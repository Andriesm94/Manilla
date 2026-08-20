# Manilla

A Python project for the board game *Manilla* (Zoch Verlag, 2005): a board-setup and
turn-based simulation tool, with the eventual goal of building an AI that calculates
the best possible action and can compete against other AI players.

The full rules are in [`Manila_rules_EN.pdf`](Manila_rules_EN.pdf).

## What's here

- **`manilla/engine/models.py`** — the data model for a game snapshot: players, punts,
  wares, the black market, accomplice placement spaces (ware punts, port, shipyard,
  pirate boat, pilot island, insurance), prices and payouts.
- **`manilla/ui/board_setup.py`** — a Tkinter interface for visualizing and playing
  through a voyage: click accomplice spaces to place them (turn-based, with payment),
  move punts along the sea lanes, and once everyone has placed, the dice roll and
  punts move automatically. Handles arrivals, shipwrecks, and pirate plunder.
- **`tests/`** — unit tests for the data model.

## Running it

```bash
python main.py
```

This launches the board-setup/play window. From the toolbar you can start a new
default game (4-5 players), randomize a board state, or save/load a snapshot as JSON.

Run the tests with:

```bash
python -m unittest discover -s tests
```

## Status

Currently implemented: the board data model, the setup/visualization UI, and a
playable accomplice-placement + dice-movement voyage loop (arrivals, shipwrecks,
pirate plunder, port/shipyard rewards).

Not yet implemented: the harbor master auction, buying shares, loading/choosing
wares, pilot influence, profit distribution for ware-punt accomplices, ware value
rises, and the AI/ECR decision-making engine described in [`ideas.txt`](ideas.txt).
