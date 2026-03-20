import numpy as np

from .env import BidAction, DeckCard, PartialTarneebState, Suit, TarneebAction, TarneebGameActions

# Canonical ordering for suits used throughout feature encoding
_SUIT_ORDER = [Suit.HEARTS, Suit.DIAMONDS, Suit.CLUBS, Suit.SPADES]

# Feature vector layout
# [  0:  52] holding cards (one-hot over 52 cards)
# [ 52: 104] played cards in current trick (one-hot)
# [104: 109] trump suit (one-hot: H/D/C/S/none)
# [109: 161] action card played (one-hot if action is DeckCard)
# [161]      is_pass flag
# [162]      is_double flag
# [163: 170] bid value one-hot (7-13 → indices 0-6)
# [170: 174] bid suit one-hot (H/D/C/S)
# [174: 176] cumulative team scores normalised by max score (31)
# [176: 178] current round trick scores normalised by max tricks (13)
# [178]      round number normalised by max rounds (13)
# [179]      is_bidding_phase flag
FEATURE_SIZE = 180


def _card_index(card: DeckCard) -> int:
    """Unique index 0-51 for a card (suit * 13 + rank-1)."""
    return _SUIT_ORDER.index(card.suit) * 13 + (card.value() - 1)


def _suit_one_hot(suit: Suit | None) -> np.ndarray:
    """5-dimensional one-hot: indices 0-3 for H/D/C/S, index 4 for 'no trump'."""
    vec = np.zeros(5)
    if suit is not None:
        vec[_SUIT_ORDER.index(suit)] = 1.0
    else:
        vec[4] = 1.0
    return vec


def tarneeb_feature_extractor(s: PartialTarneebState, a: TarneebAction) -> np.ndarray:
    """Build a fixed-length feature vector for a (state, action) pair in Tarneeb."""
    features = np.zeros(FEATURE_SIZE)

    # Holding cards
    for card in s.holding_cards:
        features[_card_index(card)] = 1.0

    # Cards played so far in the current trick
    for card in s.played_cards:
        features[52 + _card_index(card)] = 1.0

    # Trump suit
    features[104:109] = _suit_one_hot(s.trump_suit)

    # Action encoding
    if isinstance(a, DeckCard):
        features[109 + _card_index(a)] = 1.0
    elif isinstance(a, BidAction):
        bid_idx = a.value - 7  # 7→0, …, 13→6
        if 0 <= bid_idx < 7:
            features[163 + bid_idx] = 1.0
        features[170 + _SUIT_ORDER.index(a.suit)] = 1.0
    elif a == TarneebGameActions.PASS:
        features[161] = 1.0
    elif a == TarneebGameActions.DOUBLE:
        features[162] = 1.0

    # Cumulative team scores (normalised)
    features[174] = s.score[0] / 31.0
    features[175] = s.score[1] / 31.0

    # Current round trick scores (normalised by 13 tricks per round)
    features[176] = s.round_score[0] / 13.0
    features[177] = s.round_score[1] / 13.0

    # Round number (normalised by 13)
    features[178] = s.round_num / 13.0

    # Bidding phase flag
    features[179] = 1.0 if s.trump_suit is None else 0.0

    return features
