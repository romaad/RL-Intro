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
from agents.value_approx import LinearApproxAgent, LinearValueApproximator
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
        if random.random() < getattr(self, '_epsilon', 0.1):  # default epsilon
            return random.choice(possible_actions)
        else:
            # Greedy action: choose the one with highest Q-value
            best_action = max(possible_actions, key=lambda a: self.q_value(s, a))
            return best_action

    def get_variable_learning_rate(self, s: PartialTarneebState, a: TarneebAction | None) -> float:
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


# For linear approximation, need feature extractor
# But for simplicity, skip for now


class SarsaLambdaTarneebLinearApproxAgent(
    _TarneebControlBaseAgent,
    SarsaLambdaAgent[PartialTarneebState, TarneebAction],
):
    """SARSA(λ) agent for Tarneeb with linear approximation."""

    @property
    def name(self) -> str:
        return f"SarsaLinearApprox(λ={self._lambda})"

    def __init__(self, lambbda: float, gamma: float) -> None:
        # Need to implement LinearValueApproximator for Tarneeb
        # For now, just inherit
        _TarneebControlBaseAgent.__init__(self)
        SarsaLambdaAgent.__init__(self, lambbda=lambbda, gamma=gamma)
