from base import Agent
from .env import PartialTarneebState, TarneebAction, TarneebGameActions, Suit, DeckCard
import random


class RandomTarneebAgent(Agent[PartialTarneebState, TarneebAction]):
    AGENT_STATE_T = type(None)

    def __init__(self):
        self._state = None  # No state to save for random agent

    def act(self, s: PartialTarneebState) -> TarneebAction:
        # If suit not selected, choose a random action: suit or pass/double
        if not s.trump_suit:
            # Can select suit or pass/double
            actions = [
                Suit.HEARTS,
                Suit.DIAMONDS,
                Suit.CLUBS,
                Suit.SPADES,
                TarneebGameActions.PASS,
                TarneebGameActions.DOUBLE,
            ]
            return random.choice(actions)
        else:
            # Must play a card from holding
            if s.holding_cards:
                return random.choice(s.holding_cards)
            else:
                # No cards left, but shouldn't happen
                return TarneebGameActions.PASS  # fallback


class HumanTarneebAgent(Agent[PartialTarneebState, TarneebAction]):
    AGENT_STATE_T = type(None)

    def __init__(self):
        self._state = None

    def act(self, s: PartialTarneebState) -> TarneebAction:
        print(f"\nYour turn!")
        print(
            f"Played cards: {[str(c) for c in sorted(s.played_cards, key=lambda c: (c.suit.value.value, -c._number))]}"
        )
        print(
            f"Your cards: {[str(c) for c in sorted(s.holding_cards, key=lambda c: (c.suit.value.value, -c._number))]}"
        )
        print(f"Trump suit: {s.trump_suit}")
        print(f"Score: {s.score}")
        print(f"Round score: {s.round_score}")
        if s.double_by is not None:
            print(f"Doubled by player {s.double_by}")
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
                        return card
                    else:
                        print("You don't have that card!")
                        return self.act(s)
                except ValueError:
                    pass

        print("Invalid input, try again.")
        return self.act(s)
