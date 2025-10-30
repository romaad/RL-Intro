import numpy as np

from easy21.easy21 import Easy21Action, Easy21State

CUSTOM_FEATURES_LEN = 36


def custom_easy21_q_extractor(s: Easy21State, a: Easy21Action) -> np.ndarray:
    """
    Custom feature extractor for Easy21 state-action pairs.
    from the easy21 assignment page 3.
    """
    dealer_first_card = s.dealer_first_card
    player_sum = s.player_sum
    features = np.zeros(CUSTOM_FEATURES_LEN)
    dealer_idx = [[1, 4], [4, 7], [7, 10]]
    player_idx = [[1, 6], [4, 9], [7, 12], [10, 15], [13, 18], [16, 21]]
    idx = 0
    for d_range in dealer_idx:
        for p_range in player_idx:
            if (
                d_range[0] <= dealer_first_card <= d_range[1]
                and p_range[0] <= player_sum <= p_range[1]
            ):
                if a == Easy21Action.HIT:
                    features[idx] = 1.0
                else:
                    features[idx + int(CUSTOM_FEATURES_LEN / 2)] = 1.0
            idx += 1
    return features
