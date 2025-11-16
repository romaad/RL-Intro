from dataclasses import dataclass
from enum import Enum

from base import MultipleAgentEnv, Outcome


@dataclass(frozen=True)
class SuitConf:
    value: int
    name: str
    abbv: str


class Suit(Enum):
    HEARTS = SuitConf(3, "Hearts", "H")
    DIAMONDS = SuitConf(2, "Diamonds", "D")
    CLUBS = SuitConf(1, "Clubs", "C")
    SPADES = SuitConf(4, "Spades", "S")


@dataclass(frozen=True)
class DeckCard:
    _suit: Suit
    _number: int

    def __str__(self) -> str:
        return f"{self._suit.value.abbv}{self.number}"

    def value(self) -> int:
        return self._number

    def number(self) -> int:
        """Get the number of the card (1-54)."""
        return self._number * self._suit.value.value


@dataclass(frozen=True)
class TarneebSate:
    # previously played cards in the current round
    played_cards: list[DeckCard]
    # cards that the player is holding
    holding_cards: list[DeckCard]
    trump_suit: Suit | None

    def __str__(self) -> str:
        return (
            f"(P:{self.played_cards},"
            f"T:{self.trump_suit},"
            f"H:{self.holding_cards})"
        )


# we just have to play a card from our holding cards
TarneebAction = DeckCard | Suit


class TarneebEnv(MultipleAgentEnv[TarneebSate, TarneebAction]):

    def __init__(self) -> None:
        self.reset()
        super().__init__()

    def step_impl(
        self, s: TarneebSate, action: list[TarneebAction]
    ) -> Outcome[TarneebSate]:
        invalid_move_outcome = Outcome(
            next_state=s,
            reward=-1.0,
            done=True,
        )
        if isinstance(action, Suit):
            if s.trump_suit and s.trump_suit.value.value >= action.value.value:
                # can't select a lower-value trump suit
                return invalid_move_outcome
            # setting trump suit
            return Outcome(
                next_state=TarneebSate(
                    played_cards=s.played_cards,
                    holding_cards=s.holding_cards,
                    trump_suit=action,
                ),
                reward=0.0,
                done=False,
            )

        played_card: DeckCard = action
        if played_card not in s.holding_cards:
            return invalid_move_outcome
        # playing a card
        new_holding = s.holding_cards.copy()
        new_holding.remove(played_card)
        played_cards = s.played_cards + [played_card]
        if len(played_cards) % 4 == 0:
            # round is over
            return Outcome(
                next_state=TarneebSate(
                    played_cards=[],
                    holding_cards=new_holding,
                    trump_suit=None,
                ),
                reward=0.0,
                done=True,
            )

    def init_state(self) -> TarneebSate:
        return TarneebSate(
            played_cards=[],
            holding_cards=[],
            trump_suit=None,
        )
