"""A learned model for the harbor master's share-buying choice.

Milestone 5's scope, and only that: which of the still-available shares is
worth buying, given how many shares of each ware are already in play and
which punts the harbor master is about to favour. Bidding -- including
`first_mover_value` -- stays heuristic (see `manilla.engine.seat_value`).

**What it predicts.** Not "which ware to buy" as a category, but each
ware's *final black-market level*, one regression per ware. A share is
worth what its ware is worth when the game ends, so predicting that and
subtracting what the share costs today turns four predictions into a
decision: buy whichever ware has the best predicted level minus price, if
any of them clears zero. Framing it this way means every row in
`bid_buy_training.jsonl` yields four training examples instead of one, and
the model stays interpretable -- you can ask it what it thinks jade will
be worth, and check that against a real game.

**Ridge, solved directly.** Ordinary least squares plus a small penalty on
the coefficients, solved through the normal equations with Gaussian
elimination -- a handful of features means the matrix is tiny, so there's
no need for an iterative optimiser or an external library. The penalty
exists because the features are correlated (a ware being favoured and its
share count both track how attractive it looks), which makes unpenalised
OLS coefficients jumpy without changing predictions much. The bias term is
deliberately left unpenalised: shrinking it would just bias every
prediction toward zero.

**Splitting by game, not by row.** Every voyage in one game carries the
*same* label -- that game's final black market -- so rows from a game are
anything but independent. Splitting rows at random would put voyage 3 of a
game in training and voyage 4 in test, and the model would score well by
recognising the game rather than by learning anything. `train_test_split`
splits on `game_id` for that reason; any evaluation that doesn't will
report a number far better than the truth.

**Why the current black-market level is in here.** The first version of
this model used only share counts, the favoured punts, and the voyage
number -- and lost to a one-line heuristic ("just buy the favoured ware"):
0.412 top-pick accuracy against 0.430, at R^2 0.086. The missing input was
the *current* level, which bounds the final one from below: a ware at 20
needs one more rise, a ware at 0 needs four. Adding it moved R^2 from 0.05
to 0.35 and top-pick accuracy from 0.38 to 0.61 against the same 0.37
heuristic, on identical data.

Worth knowing what that also revealed: `shares_in_play` was carrying a
real coefficient (+0.63) only because it proxied for the market level.
With the level itself present its coefficient collapses to about -0.03 --
it was standing in for the thing that actually mattered. This is why rows
written before 2026-08-30 can't train this model, and `load_rows` skips
them rather than defaulting the field to zero, which would teach it that
every ware starts worthless.
"""

from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

from manilla.engine.models import GameState, Ware

FEATURE_NAMES: Tuple[str, ...] = (
    "bias",
    "shares_in_play",
    "is_favored",
    "voyage_number",
    "total_shares_in_play",
    "favored_x_shares",
    "black_market",
    "total_black_market",
    "black_market_x_favored",
)

DEFAULT_DATA_PATH = Path("data") / "bid_buy_training.jsonl"

# Rows below this carry no current black-market level -- see the module
# docstring for why that makes them unusable here rather than merely worse.
MIN_SCHEMA_VERSION = 2

# Trained on 10,548 four-player REV voyages (the training half of 14,108
# across 2,324 games), 2026-08-31 -- the distribution these coefficients
# will actually face, replacing a provisional fit on random-policy games.
#
# On held-out REV games this scores R^2 0.273 and picks the best ware 54.4%
# of the time, against 42.9% for the "just buy the favoured ware" heuristic
# and 29.2% for random choice. The earlier random-policy fit scored 0.208 /
# 52.8% on the same held-out games -- so it transferred better than it had
# any right to, which is decent evidence the relationship really is
# structural, but training on the right distribution is still worth 1.6
# points of accuracy.
DEFAULT_COEFFICIENTS: Tuple[float, ...] = (
    6.5493,
    -1.3943,
    4.4272,
    -0.6611,
    1.2479,
    0.3012,
    1.0527,
    -0.2267,
    0.0511,
)


def extract_features(
    ware: Ware,
    shares_in_play: Dict[str, int],
    favored_wares: Sequence[str],
    voyage_number: int,
    black_market: Dict[str, int],
) -> List[float]:
    """Feature row for one ware in one voyage -- see `FEATURE_NAMES`.

    `favored_wares` is the pair of loaded wares the harbor master gave the
    most advanced start positions (`selfplay_data._favored_wares`). Note
    it can't distinguish the third loaded ware from the one left ashore,
    since only the top two are recorded.

    `black_market` is the level of every ware right now, which is what
    makes this model work at all -- see the module docstring.
    """
    own = float(shares_in_play.get(ware.value, 0))
    favored = 1.0 if ware.value in favored_wares else 0.0
    level = float(black_market.get(ware.value, 0))
    return [
        1.0,
        own,
        favored,
        float(voyage_number),
        float(sum(shares_in_play.values())),
        favored * own,
        level,
        float(sum(black_market.values())),
        level * favored,
    ]


def _solve(matrix: List[List[float]], rhs: List[float]) -> List[float]:
    """Solve `matrix @ w = rhs` by Gaussian elimination with partial
    pivoting. Small and dense by construction -- one row per feature."""
    n = len(rhs)
    aug = [row[:] + [rhs[i]] for i, row in enumerate(matrix)]

    for col in range(n):
        pivot = max(range(col, n), key=lambda r: abs(aug[r][col]))
        if abs(aug[pivot][col]) < 1e-12:
            raise ValueError("singular matrix -- features are exactly collinear")
        aug[col], aug[pivot] = aug[pivot], aug[col]
        for row in range(col + 1, n):
            factor = aug[row][col] / aug[col][col]
            for k in range(col, n + 1):
                aug[row][k] -= factor * aug[col][k]

    weights = [0.0] * n
    for row in reversed(range(n)):
        total = aug[row][n] - sum(aug[row][k] * weights[k] for k in range(row + 1, n))
        weights[row] = total / aug[row][row]
    return weights


def fit_ridge(features: List[List[float]], targets: List[float], alpha: float = 1.0) -> List[float]:
    """Ridge coefficients for `features` -> `targets`. The first feature is
    assumed to be the bias and is left unpenalised."""
    if not features:
        raise ValueError("no training rows")
    n_features = len(features[0])

    gram = [[0.0] * n_features for _ in range(n_features)]
    moment = [0.0] * n_features
    for row, target in zip(features, targets):
        for i in range(n_features):
            moment[i] += row[i] * target
            for j in range(n_features):
                gram[i][j] += row[i] * row[j]

    for i in range(1, n_features):  # skip the bias
        gram[i][i] += alpha
    return _solve(gram, moment)


@dataclass
class ShareValueModel:
    """Predicts a ware's final black-market level from `FEATURE_NAMES`."""

    coefficients: List[float]
    feature_names: Tuple[str, ...] = FEATURE_NAMES

    def predict(
        self,
        ware: Ware,
        shares_in_play: Dict[str, int],
        favored_wares: Sequence[str],
        voyage_number: int,
        black_market: Dict[str, int],
    ) -> float:
        row = extract_features(ware, shares_in_play, favored_wares, voyage_number, black_market)
        return sum(c * x for c, x in zip(self.coefficients, row))

    def describe(self) -> str:
        width = max(len(n) for n in self.feature_names)
        return "\n".join(
            f"  {name:<{width}}  {coef:+9.4f}" for name, coef in zip(self.feature_names, self.coefficients)
        )

    def to_dict(self) -> dict:
        return {"coefficients": list(self.coefficients), "feature_names": list(self.feature_names)}

    @staticmethod
    def from_dict(payload: dict) -> "ShareValueModel":
        return ShareValueModel(
            coefficients=list(payload["coefficients"]),
            feature_names=tuple(payload["feature_names"]),
        )


def load_rows(path: Optional[Path] = None, policy: str = "rev") -> List[dict]:
    path = Path(path) if path is not None else DEFAULT_DATA_PATH
    if not path.exists():
        return []
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        # Rows predating the black_market field can't train this model, and
        # defaulting it to zero would teach it every ware starts worthless.
        # Version 1 rows carry no schema_version at all, hence the default.
        if row.get("schema_version", 1) < MIN_SCHEMA_VERSION:
            continue
        if row.get("policy") == policy and row.get("final_black_market") and row.get("black_market"):
            rows.append(row)
    return rows


def rows_to_examples(rows: Sequence[dict]) -> Tuple[List[List[float]], List[float]]:
    """One example per (voyage, ware) pair -- four per row."""
    features: List[List[float]] = []
    targets: List[float] = []
    for row in rows:
        for ware in Ware:
            features.append(
                extract_features(
                    ware,
                    row["shares_in_play"],
                    row["favored_wares"],
                    row["voyage_number"],
                    row["black_market"],
                )
            )
            targets.append(float(row["final_black_market"][ware.value]))
    return features, targets


def train_test_split(
    rows: Sequence[dict], test_fraction: float = 0.25, seed: int = 0
) -> Tuple[List[dict], List[dict]]:
    """Split on `game_id`, never on rows -- see the module docstring."""
    game_ids = sorted({row["game_id"] for row in rows})
    random.Random(seed).shuffle(game_ids)
    cut = int(len(game_ids) * (1.0 - test_fraction))
    train_ids = set(game_ids[:cut])
    return (
        [r for r in rows if r["game_id"] in train_ids],
        [r for r in rows if r["game_id"] not in train_ids],
    )


def default_model() -> ShareValueModel:
    """The shipped model -- see `DEFAULT_COEFFICIENTS` for its provenance
    and why it should be retrained on REV rows when they exist."""
    return ShareValueModel(coefficients=list(DEFAULT_COEFFICIENTS))


def train(rows: Sequence[dict], alpha: float = 1.0) -> ShareValueModel:
    features, targets = rows_to_examples(rows)
    return ShareValueModel(coefficients=fit_ridge(features, targets, alpha))


@dataclass
class Evaluation:
    n_examples: int
    n_games: int
    r_squared: float
    mean_absolute_error: float
    baseline_mae: float
    top_pick_accuracy: float
    favored_pick_accuracy: float
    random_pick_accuracy: float

    def __str__(self) -> str:
        return (
            f"  examples          {self.n_examples} across {self.n_games} games\n"
            f"  R^2               {self.r_squared:.4f}\n"
            f"  MAE               {self.mean_absolute_error:.3f} "
            f"(predict-the-mean baseline {self.baseline_mae:.3f})\n"
            f"  top-pick accuracy {self.top_pick_accuracy:.3f} "
            f"(favoured-ware rule {self.favored_pick_accuracy:.3f}, "
            f"random {self.random_pick_accuracy:.3f})"
        )


def evaluate(model: ShareValueModel, rows: Sequence[dict], seed: int = 0) -> Evaluation:
    """Fit quality plus the metric that actually matters: how often the
    model's best-predicted ware is genuinely the one that ended highest.

    Ties on the true best ware count as a hit for any tied ware -- several
    wares often finish level, so demanding one exact answer would punish a
    correct call. The random baseline is scored the same way, so the
    comparison stays fair.
    """
    features, targets = rows_to_examples(rows)
    predictions = [sum(c * x for c, x in zip(model.coefficients, row)) for row in features]

    mean_target = sum(targets) / len(targets)
    ss_res = sum((t - p) ** 2 for t, p in zip(targets, predictions))
    ss_tot = sum((t - mean_target) ** 2 for t in targets)
    mae = sum(abs(t - p) for t, p in zip(targets, predictions)) / len(targets)
    baseline_mae = sum(abs(t - mean_target) for t in targets) / len(targets)

    wares = list(Ware)
    rng = random.Random(seed)
    hits = favored_hits = random_hits = 0
    for row in rows:
        finals = row["final_black_market"]
        best = max(finals[w.value] for w in wares)
        winners = {w.value for w in wares if finals[w.value] == best}

        scored = [
            (
                model.predict(
                    w, row["shares_in_play"], row["favored_wares"], row["voyage_number"], row["black_market"]
                ),
                w,
            )
            for w in wares
        ]
        hits += max(scored, key=lambda pair: pair[0])[1].value in winners
        favored_hits += (row["favored_wares"][0] if row["favored_wares"] else wares[0].value) in winners
        random_hits += rng.choice(wares).value in winners

    n = len(rows)
    return Evaluation(
        n_examples=len(targets),
        n_games=len({r["game_id"] for r in rows}),
        r_squared=1.0 - ss_res / ss_tot if ss_tot else 0.0,
        mean_absolute_error=mae,
        baseline_mae=baseline_mae,
        top_pick_accuracy=hits / n,
        favored_pick_accuracy=favored_hits / n,
        random_pick_accuracy=random_hits / n,
    )


def favored_wares_from_setup(wares_loaded: Sequence[Ware], positions: Dict[Ware, int]) -> List[str]:
    """The two loaded wares with the most advanced start positions -- the
    same rule `selfplay_data._favored_wares` applies to a live board, but
    computed from a *planned* setup, before it's been applied."""
    ordered = sorted(wares_loaded, key=lambda w: positions.get(w, 0), reverse=True)
    return [w.value for w in ordered[:2]]


def best_share_to_buy(
    model: ShareValueModel,
    state: GameState,
    favored_wares: Sequence[str],
) -> Optional[Ware]:
    """The ware whose share is worth buying now, or `None` if none of them
    is: predicted final level minus what the share costs today, best first,
    and nothing bought when even the best is a loss.

    Only wares with shares still available are considered.
    """
    best: Optional[Tuple[float, Ware]] = None
    for ware in Ware:
        if state.shares_available(ware) <= 0:
            continue
        predicted = model.predict(
            ware,
            {w.value: state.shares_owned(w) for w in Ware},
            favored_wares,
            state.voyage_number,
            {w.value: v for w, v in state.black_market.values.items()},
        )
        net = predicted - state.black_market.share_price(ware)
        if net > 0 and (best is None or net > best[0]):
            best = (net, ware)
    return best[1] if best else None
