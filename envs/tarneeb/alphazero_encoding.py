"""State and action encoding utilities for the AlphaZero Tarneeb agent.

Action space (ACTION_SIZE = 89)
--------------------------------
Indices  0 –  51 : DeckCard plays         (suit_idx × 13 + rank_idx)
Indices 52 –  79 : BidAction with suit    (52 + (value-7) × 4 + suit_idx)
Indices 80 –  86 : BidAction suns (None)  (80 + (value-7))
Index   87       : TarneebGameActions.PASS
Index   88       : TarneebGameActions.DOUBLE

Card rank ordering (consistent with feature_extractor.py):
  rank_idx = card.value() - 1   where value() = number-1 if number>1 else 13
  → 2→0, 3→1, …, K→11, A→12

Global feature vector layout (G = 45)
--------------------------------------
[0]     current_high_bid / 13
[1]     score[0] / 31
[2]     score[1] / 31
[3]     round_score[0] / 13
[4]     round_score[1] / 13
[5]     round_num / 13
[6]     is_bidding_phase  (1 if trump_suit is None)
[7:11]  agent_idx one-hot  (4)
[11:16] bidder one-hot     (5: players 0-3 + "no bidder")
[16:21] double_by one-hot  (5: players 0-3 + "no double")
[21:45] bids per player    (6 × 4):
          [p*6+21] = bid_value / 13  (0 if no bid)
          [p*6+22 .. p*6+25] = suit one-hot H/D/C/S
          [p*6+26] = suns flag (suit is None)
"""

from __future__ import annotations

import numpy as np

from envs.tarneeb.env import (
    BidAction,
    DeckCard,
    PartialTarneebState,
    Suit,
    TarneebAction,
    TarneebGameActions,
)

# Canonical suit ordering (must match feature_extractor.py)
_SUIT_ORDER: list[Suit] = [Suit.HEARTS, Suit.DIAMONDS, Suit.CLUBS, Suit.SPADES]

ACTION_SIZE: int = 89
GLOBAL_FEATURES_SIZE: int = 45

# ---------------------------------------------------------------------------
# Pre-computed action list
# ---------------------------------------------------------------------------


def _build_action_list() -> list[TarneebAction]:
    actions: list[TarneebAction] = []
    # 0–51: DeckCard  (suit × 13 + rank_idx, rank by value()-1)
    for suit in _SUIT_ORDER:
        for rank in range(13):
            # rank 0..11 → number rank+2; rank 12 → number 1 (Ace)
            number = 1 if rank == 12 else rank + 2
            actions.append(DeckCard(suit, number))
    # 52–79: BidAction with suit
    for value in range(7, 14):
        for suit in _SUIT_ORDER:
            actions.append(BidAction(value, suit))
    # 80–86: BidAction suns
    for value in range(7, 14):
        actions.append(BidAction(value, None))
    # 87: PASS  88: DOUBLE
    actions.append(TarneebGameActions.PASS)
    actions.append(TarneebGameActions.DOUBLE)
    assert len(actions) == ACTION_SIZE
    return actions


ACTION_LIST: list[TarneebAction] = _build_action_list()

# Fast reverse lookup dict built once at import time
_ACTION_TO_INDEX: dict = {}
for _i, _a in enumerate(ACTION_LIST):
    _ACTION_TO_INDEX[_a] = _i


# ---------------------------------------------------------------------------
# Index ↔ action conversion
# ---------------------------------------------------------------------------


def action_to_index(a: TarneebAction) -> int:
    """Return the canonical index (0..88) for *a*.

    Raises ``KeyError`` if *a* is not in the action space.
    """
    return _ACTION_TO_INDEX[a]


def index_to_action(i: int) -> TarneebAction:
    """Return the canonical action for index *i* (0..88)."""
    return ACTION_LIST[i]


# ---------------------------------------------------------------------------
# State encoding
# ---------------------------------------------------------------------------


def partial_state_to_numpy(s: PartialTarneebState) -> np.ndarray:
    """Encode *s* as a ``(4, 13, 3)`` float32 card grid.

    Spatial axes:
      axis 0 – suit index (0=H, 1=D, 2=C, 3=S)
      axis 1 – rank index (rank_idx = value()-1, see module docstring)

    Channels:
      Ch 0 – holding cards one-hot
      Ch 1 – played cards in current trick one-hot
      Ch 2 – trump suit plane (entire trump-suit row = 1, others = 0)
    """
    grid = np.zeros((4, 13, 3), dtype=np.float32)

    for card in s.holding_cards:
        suit_idx = _SUIT_ORDER.index(card.suit)
        rank_idx = card.value() - 1
        grid[suit_idx, rank_idx, 0] = 1.0

    for card in s.played_cards:
        suit_idx = _SUIT_ORDER.index(card.suit)
        rank_idx = card.value() - 1
        grid[suit_idx, rank_idx, 1] = 1.0

    if s.trump_suit is not None:
        trump_idx = _SUIT_ORDER.index(s.trump_suit)
        grid[trump_idx, :, 2] = 1.0

    return grid


def encode_global_features(s: PartialTarneebState, agent_idx: int) -> np.ndarray:
    """Return a ``(45,)`` float32 vector of context features for *agent_idx*.

    See module docstring for the full layout.
    """
    feats = np.zeros(GLOBAL_FEATURES_SIZE, dtype=np.float32)

    feats[0] = s.current_high_bid / 13.0
    feats[1] = s.score[0] / 31.0
    feats[2] = s.score[1] / 31.0
    feats[3] = s.round_score[0] / 13.0
    feats[4] = s.round_score[1] / 13.0
    feats[5] = s.round_num / 13.0
    feats[6] = 1.0 if s.trump_suit is None else 0.0

    # agent_idx one-hot [7:11]
    feats[7 + agent_idx] = 1.0

    # bidder one-hot [11:16] (index 15 = no bidder)
    if s.bidder is not None:
        feats[11 + s.bidder] = 1.0
    else:
        feats[15] = 1.0

    # double_by one-hot [16:21] (index 20 = no double)
    if s.double_by is not None:
        feats[16 + s.double_by] = 1.0
    else:
        feats[20] = 1.0

    # bids per player [21:45]
    for p in range(4):
        base = 21 + p * 6
        bid = s.bids[p]
        if bid is not None:
            value, suit = bid
            feats[base] = value / 13.0
            if suit is not None:
                feats[base + 1 + _SUIT_ORDER.index(suit)] = 1.0
            else:
                feats[base + 5] = 1.0  # suns

    return feats


# ---------------------------------------------------------------------------
# Valid-action mask
# ---------------------------------------------------------------------------


def available_actions_mask(s: PartialTarneebState) -> np.ndarray:
    """Return a binary mask of shape ``(ACTION_SIZE,)`` for actions legal in *s*.

    Bidding phase is inferred from ``s.trump_suit is None`` (proxy used
    consistently with all other agents in this codebase).
    """
    mask = np.zeros(ACTION_SIZE, dtype=np.float32)

    if s.trump_suit is None:
        # ----- Bidding phase -----
        # PASS is always valid during bidding
        mask[87] = 1.0

        # Determine if current top bid is suns
        current_bid_is_suns = (
            s.bidder is not None
            and s.bids[s.bidder] is not None
            and s.bids[s.bidder][1] is None  # type: ignore[index]
        )

        for value in range(7, 14):
            # BidAction with suit: valid when value strictly exceeds high bid
            if value > s.current_high_bid:
                for suit_idx in range(4):
                    mask[52 + (value - 7) * 4 + suit_idx] = 1.0

            # BidAction suns: strictly higher, OR same value if current is not suns
            if value > s.current_high_bid or (
                value == s.current_high_bid
                and s.current_high_bid >= 7
                and not current_bid_is_suns
            ):
                mask[80 + (value - 7)] = 1.0

        # DOUBLE valid if there is a bidder and nobody has doubled yet
        if s.bidder is not None and s.double_by is None:
            mask[88] = 1.0
    else:
        # ----- Playing phase -----
        if s.holding_cards:
            if len(s.played_cards) > 0:
                led_suit = s.played_cards[0].suit
                led_suit_cards = [c for c in s.holding_cards if c.suit == led_suit]
                playable = led_suit_cards if led_suit_cards else s.holding_cards
            else:
                playable = s.holding_cards
            for card in playable:
                mask[action_to_index(card)] = 1.0

    return mask
