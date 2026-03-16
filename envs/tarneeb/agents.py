from base import Agent
from .env import PartialTarneebState, TarneebAction, TarneebGameActions, Suit, DeckCard
import random


class RandomTarneebAgent(Agent[PartialTarneebState, TarneebAction]):
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
