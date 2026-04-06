from base import Agent
from .env import (
    PartialTarneebState,
    TarneebAction,
    TarneebGameActions,
    Suit,
    DeckCard,
    BidAction,
)
from agents.monte_carlo import MonteCarloAgent
from agents.sarsa import SarsaAgent, SarsaLambdaAgent
from agents.value_approx import (
    CNNApproxAgent,
    CNNSeparateHeadsApproxAgent,
    CNNSeparateHeadsValueApproximator,
    CNNValueApproximator,
    LinearApproxAgent,
    LinearValueApproximator,
)
import random


class RandomTarneebAgent(Agent[PartialTarneebState, TarneebAction]):
    AGENT_STATE_T = type(None)

    def __init__(self):
        self._state = None  # No state to save for random agent

    def act(self, s: PartialTarneebState) -> TarneebAction:
        # If suit not selected, bidding phase
        if not s.trump_suit:
            # Can bid higher than current_high_bid (7-13), or pass
            actions = [TarneebGameActions.PASS]
            for bid in range(s.current_high_bid + 1, 14):
                for suit in Suit:
                    actions.append(BidAction(bid, suit))
                # suns (no-trump) is also a valid bid at each higher value
                actions.append(BidAction(bid, None))
            # suns can also overbid a suit bid at the same value
            current_bid_is_suns = (
                s.bidder is not None
                and s.bids[s.bidder] is not None
                and s.bids[s.bidder][1] is None
            )
            if not current_bid_is_suns and s.current_high_bid >= 7:
                actions.append(BidAction(s.current_high_bid, None))
            return random.choice(actions)
        else:
            # Must play a card from holding
            if s.holding_cards:
                # Follow suit if possible
                if len(s.played_cards) > 0:
                    led_suit = s.played_cards[0].suit
                    led_suit_cards = [c for c in s.holding_cards if c.suit == led_suit]
                    if led_suit_cards:
                        return random.choice(led_suit_cards)
                return random.choice(s.holding_cards)
            else:
                # No cards left, but shouldn't happen
                return TarneebGameActions.PASS  # fallback


class HumanTarneebAgent(Agent[PartialTarneebState, TarneebAction]):
    AGENT_STATE_T = type(None)

    def __init__(self, verbose: bool = False):
        self._state = None
        self.verbose = verbose

    def act(self, s: PartialTarneebState) -> TarneebAction:
        print(f"\nYour turn!")
        if s.played_cards:
            played_info = []
            if s.last_player_idx is not None:
                num_cards = len(s.played_cards)
                starting_player = (s.last_player_idx - num_cards + 1) % 4
                for i, card in enumerate(s.played_cards):
                    player = (starting_player + i) % 4
                    played_info.append(f"P{player}: {card}")
            else:
                played_info = [str(card) for card in s.played_cards]
            print(f"Played cards in this trick: {', '.join(played_info)}")
        else:
            print("No cards played yet in this trick.")
        print(
            f"Your cards: {[str(c) for c in sorted(s.holding_cards, key=lambda c: (c.suit.value.value, -c._number))]}"
        )
        print(f"Trump suit: {s.trump_suit}")
        print(f"Score: {s.score}")
        print(f"Round: {s.round_num}, Round score: {s.round_score}")
        if s.double_by is not None:
            print(f"Doubled by P{s.double_by}")
        if not s.trump_suit:
            print(f"Current high bid: {s.current_high_bid}")
            if s.bidder is not None:
                print(f"Current bidder: {s.bidder}")
            action_str = (
                input("Enter action (PASS, BID <7-13> <suit: H/D/C/S>): ")
                .strip()
                .upper()
            )
        else:
            action_str = (
                input(
                    "Enter action (suit: H/D/C/S, card: e.g. H5, HJ, CQ, DK or 5H, PASS, DOUBLE): "
                )
                .strip()
                .upper()
            )

        # Parse suit
        suit_map = {
            "H": Suit.HEARTS,
            "D": Suit.DIAMONDS,
            "C": Suit.CLUBS,
            "S": Suit.SPADES,
        }
        if action_str in suit_map:
            return suit_map[action_str]

        # Parse game actions
        if action_str == "PASS":
            return TarneebGameActions.PASS
        if action_str == "DOUBLE":
            return TarneebGameActions.DOUBLE
        # Parse bid
        if action_str.startswith("BID "):
            parts = action_str[4:].split()
            if len(parts) == 2:
                try:
                    bid = int(parts[0])
                    suit = suit_map.get(parts[1])
                    if 7 <= bid <= 13 and suit:
                        return BidAction(bid, suit)
                except ValueError:
                    pass
        # Parse card
        if len(action_str) >= 2:
            first = action_str[0]
            rest = action_str[1:]
            suit_char = None
            number_str = None
            if first in suit_map:
                suit_char = first
                number_str = rest
            elif len(rest) > 0 and rest[-1] in suit_map:
                suit_char = rest[-1]
                number_str = first + rest[:-1]
            if suit_char and number_str:
                face_to_num = {"J": 11, "Q": 12, "K": 13}
                try:
                    if number_str in face_to_num:
                        number = face_to_num[number_str]
                    else:
                        number = int(number_str)
                    card = DeckCard(suit_map[suit_char], number)
                    if card in s.holding_cards:
                        # Check follow suit
                        if len(s.played_cards) > 0:
                            led_suit = s.played_cards[0].suit
                            has_led_suit = any(
                                c.suit == led_suit for c in s.holding_cards
                            )
                            if has_led_suit and card.suit != led_suit:
                                print("You must follow suit!")
                                return self.act(s)
                        return card
                    else:
                        print("You don't have that card!")
                        return self.act(s)
                except ValueError:
                    pass

        print("Invalid input, try again.")
        return self.act(s)


class NaiveTarneebAgent(Agent[PartialTarneebState, TarneebAction]):
    """A simple rule-based agent for Tarneeb."""

    def act(self, s: PartialTarneebState) -> TarneebAction:
        # For bidding: bid 7 if possible, else pass
        if not s.trump_suit:
            if s.current_high_bid < 7:
                return BidAction(7, Suit.HEARTS)  # arbitrary suit
            else:
                return TarneebGameActions.PASS
        else:
            # For playing: play random valid card
            if s.holding_cards:
                return random.choice(s.holding_cards)
            else:
                return TarneebGameActions.PASS


class _TarneebControlBaseAgent:
    def action_space(self) -> list[TarneebAction]:
        # This is complex, as actions depend on state
        # For simplicity, return empty, agents handle it
        return []

    def state_to_xy(self, s: PartialTarneebState) -> tuple[int, int]:
        # Simple representation: number of cards left, round number
        return (len(s.holding_cards), s.round_num)

    def get_xy_labels(self) -> tuple[str, str]:
        return ("Cards left", "Round")

    def get_states(self) -> list[PartialTarneebState]:
        # Hard to enumerate, return empty
        return []

    def _get_possible_actions(self, s: PartialTarneebState) -> list[TarneebAction]:
        """Get possible actions for the current state."""
        if not s.trump_suit:
            # Bidding phase
            actions = [TarneebGameActions.PASS]
            for value in range(7, 14):
                for suit in Suit:
                    actions.append(BidAction(value, suit))
            return actions
        else:
            # Playing phase
            actions = []
            if s.double_by is None:
                actions.append(TarneebGameActions.DOUBLE)
            actions.extend(s.holding_cards)
            return actions

    def act(self, s: PartialTarneebState) -> TarneebAction:
        # Override to use state-dependent actions
        possible_actions = self._get_possible_actions(s)
        if random.random() < getattr(self, "_epsilon", 0.1):  # default epsilon
            return random.choice(possible_actions)
        else:
            # Greedy action: choose the one with highest Q-value
            best_action = max(possible_actions, key=lambda a: self.q_value(s, a))
            return best_action

    def get_variable_learning_rate(
        self, s: PartialTarneebState, a: TarneebAction | None
    ) -> float:
        if a is None:
            return 1.0
        # Use the number of returns for (s,a) as visit count
        visit_count = len(self._state.returns.get((s, a), []))
        return 1.0 / (visit_count + 1)


class MCTarneebAgent(
    _TarneebControlBaseAgent, MonteCarloAgent[PartialTarneebState, TarneebAction]
):
    """An agent that uses Monte Carlo methods for Tarneeb."""

    pass


class SarsaTarneebAgent(
    _TarneebControlBaseAgent,
    SarsaAgent[PartialTarneebState, TarneebAction],
    MonteCarloAgent[PartialTarneebState, TarneebAction],
):
    """SARSA agent for Tarneeb."""

    pass


class SarsaLambdaTarneebAgent(
    _TarneebControlBaseAgent,
    SarsaLambdaAgent[PartialTarneebState, TarneebAction],
    MonteCarloAgent[PartialTarneebState, TarneebAction],
):
    """SARSA(λ) agent for Tarneeb."""

    pass


class TarneebLinearValueApprox(LinearValueApproximator[PartialTarneebState, TarneebAction]):
    """Linear value function approximator for Tarneeb."""

    def __init__(self) -> None:
        from .feature_extractor import FEATURE_SIZE, tarneeb_feature_extractor

        super().__init__(
            feature_extractor=tarneeb_feature_extractor,
            feature_vector_size=FEATURE_SIZE,
            alpha=0.01,
        )


class SarsaLambdaTarneebLinearApproxAgent(
    _TarneebControlBaseAgent,
    LinearApproxAgent[PartialTarneebState, TarneebAction],
    SarsaLambdaAgent[PartialTarneebState, TarneebAction],
):
    """SARSA(λ) agent for Tarneeb with linear value function approximation."""

    _FIXED_LEARNING_RATE: float = 0.1

    @property
    def name(self) -> str:
        return f"TarneebSarsaLinearApprox(λ={self._lambda})"

    def __init__(self, lambbda: float, gamma: float) -> None:
        _TarneebControlBaseAgent.__init__(self)
        LinearApproxAgent.__init__(self, value_approximator=TarneebLinearValueApprox())
        SarsaLambdaAgent.__init__(self, lambbda=lambbda, gamma=gamma)

    def get_variable_learning_rate(
        self, s: PartialTarneebState, a: TarneebAction | None
    ) -> float:
        # Use a fixed learning rate; the linear approximator manages its own alpha.
        return self._FIXED_LEARNING_RATE

    def update(self, steps: list) -> None:
        # Reset eligibility traces at episode end (standard SARSA(λ) behaviour).
        self._eligibility.clear()


class TarneebCNNValueApprox(CNNValueApproximator[PartialTarneebState, TarneebAction]):
    """CNN + 3-hidden-layer value function approximator for Tarneeb.

    Feature layout from ``tarneeb_feature_extractor`` (FEATURE_SIZE = 180):
      [  0: 52] holding cards one-hot
      [ 52:104] played cards one-hot
      [104:109] trump suit one-hot (5-dim)
      [109:161] action card one-hot
      [161:180] other scalars (pass/double flags, bid, scores, round, phase)

    The CNN block uses 3 channels over 52 card positions:
      channel 0: holding cards  [  0: 52]
      channel 1: played cards   [ 52:104]
      channel 2: action card    [109:161]

    The other block contains the remaining 24 scalar features:
      [104:109] + [161:180]
    """

    # CNN card-channel slices: (start, stop) indexing the full feature vector
    _CNN_CHANNEL_SLICES: list[tuple[int, int]] = [(0, 52), (52, 104), (109, 161)]
    # Non-card scalar feature slices
    _OTHER_SLICES: list[tuple[int, int]] = [(104, 109), (161, 180)]

    def __init__(self) -> None:
        from .feature_extractor import tarneeb_feature_extractor

        super().__init__(
            feature_extractor=tarneeb_feature_extractor,
            cnn_input_len=52,
            cnn_channel_slices=self._CNN_CHANNEL_SLICES,
            other_slices=self._OTHER_SLICES,
            cnn_filters=16,
            cnn_kernel=4,
            fc_hidden=(256, 128, 64),
            alpha=0.001,
        )


class SarsaLambdaTarneebCNNApproxAgent(
    _TarneebControlBaseAgent,
    CNNApproxAgent[PartialTarneebState, TarneebAction],
    SarsaLambdaAgent[PartialTarneebState, TarneebAction],
):
    """SARSA(λ) agent for Tarneeb with a CNN value function approximator."""

    _FIXED_LEARNING_RATE: float = 0.1

    @property
    def name(self) -> str:
        return f"TarneebSarsaCNNApprox(λ={self._lambda})"

    def __init__(self, lambbda: float, gamma: float) -> None:
        _TarneebControlBaseAgent.__init__(self)
        CNNApproxAgent.__init__(self, value_approximator=TarneebCNNValueApprox())
        SarsaLambdaAgent.__init__(self, lambbda=lambbda, gamma=gamma)

    def get_variable_learning_rate(
        self, s: PartialTarneebState, a: TarneebAction | None
    ) -> float:
        # Use a fixed learning rate; the CNN approximator manages its own alpha.
        return self._FIXED_LEARNING_RATE

    def update(self, steps: list) -> None:
        # Reset eligibility traces at episode end (standard SARSA(λ) behaviour).
        self._eligibility.clear()


# ---------------------------------------------------------------------------
# Shared-network agents
# ---------------------------------------------------------------------------

def _is_bid_action(a: TarneebAction) -> bool:
    """Return True for bidding-phase actions (BidAction or PASS)."""
    return isinstance(a, BidAction) or a == TarneebGameActions.PASS


class TarneebSeparateHeadsCNNValueApprox(
    CNNSeparateHeadsValueApproximator[PartialTarneebState, TarneebAction]
):
    """CNN approximator with separate bid/play heads for Tarneeb.

    Feature layout from ``tarneeb_feature_extractor`` (FEATURE_SIZE = 180):
      [  0: 52] holding cards one-hot
      [ 52:104] played cards one-hot
      [104:109] trump suit one-hot (5-dim)
      [109:161] action card one-hot
      [161:180] other scalars (pass/double flags, bid, scores, round, phase)

    The CNN block uses 3 channels over 52 card positions:
      channel 0: holding cards  [  0: 52]
      channel 1: played cards   [ 52:104]
      channel 2: action card    [109:161]

    The other block contains the remaining 24 scalar features:
      [104:109] + [161:180]

    Bidding actions (BidAction / PASS) are routed to ``head_bid``;
    playing actions (DeckCard / DOUBLE) are routed to ``head_play``.
    """

    _CNN_CHANNEL_SLICES: list[tuple[int, int]] = [(0, 52), (52, 104), (109, 161)]
    _OTHER_SLICES: list[tuple[int, int]] = [(104, 109), (161, 180)]

    def __init__(self) -> None:
        from .feature_extractor import tarneeb_feature_extractor

        super().__init__(
            feature_extractor=tarneeb_feature_extractor,
            cnn_input_len=52,
            cnn_channel_slices=self._CNN_CHANNEL_SLICES,
            other_slices=self._OTHER_SLICES,
            bid_head_selector=_is_bid_action,
            cnn_filters=16,
            cnn_kernel=4,
            fc_hidden=(256, 128, 64),
            alpha=0.001,
        )


class SarsaLambdaTarneebSharedCNNApproxAgent(
    _TarneebControlBaseAgent,
    CNNApproxAgent[PartialTarneebState, TarneebAction],
    SarsaLambdaAgent[PartialTarneebState, TarneebAction],
):
    """SARSA(λ) Tarneeb agent where all players share one unified CNN network.

    Instantiate a single :class:`TarneebCNNValueApprox` and pass the same
    instance to every agent so they all train the same network weights.
    """

    _FIXED_LEARNING_RATE: float = 0.1

    @property
    def name(self) -> str:
        return f"TarneebSarsaSharedCNNApprox(λ={self._lambda})"

    def __init__(
        self,
        lambbda: float,
        gamma: float,
        shared_approximator: CNNValueApproximator[PartialTarneebState, TarneebAction],
    ) -> None:
        _TarneebControlBaseAgent.__init__(self)
        CNNApproxAgent.__init__(self, value_approximator=shared_approximator)
        SarsaLambdaAgent.__init__(self, lambbda=lambbda, gamma=gamma)

    def get_variable_learning_rate(
        self, s: PartialTarneebState, a: TarneebAction | None
    ) -> float:
        return self._FIXED_LEARNING_RATE

    def update(self, steps: list) -> None:
        self._eligibility.clear()


class SarsaLambdaTarneebSeparateHeadsCNNApproxAgent(
    _TarneebControlBaseAgent,
    CNNSeparateHeadsApproxAgent[PartialTarneebState, TarneebAction],
    SarsaLambdaAgent[PartialTarneebState, TarneebAction],
):
    """SARSA(λ) Tarneeb agent using a shared CNN with separate bid/play heads.

    All four players share the same :class:`TarneebSeparateHeadsCNNValueApprox`
    instance.  The shared backbone learns a common card representation while
    the two specialised heads independently learn bidding and playing policies.
    """

    _FIXED_LEARNING_RATE: float = 0.1

    @property
    def name(self) -> str:
        return f"TarneebSarsaSepHeadsCNNApprox(λ={self._lambda})"

    def __init__(
        self,
        lambbda: float,
        gamma: float,
        shared_approximator: TarneebSeparateHeadsCNNValueApprox,
    ) -> None:
        _TarneebControlBaseAgent.__init__(self)
        CNNSeparateHeadsApproxAgent.__init__(
            self, value_approximator=shared_approximator
        )
        SarsaLambdaAgent.__init__(self, lambbda=lambbda, gamma=gamma)

    def get_variable_learning_rate(
        self, s: PartialTarneebState, a: TarneebAction | None
    ) -> float:
        return self._FIXED_LEARNING_RATE

    def update(self, steps: list) -> None:
        self._eligibility.clear()
