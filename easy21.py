"""
environment and agent base classes
for easy21 game:
https://davidstarsilver.wordpress.com/wp-content/uploads/2025/04/easy21-assignment.pdf
"""

from dataclasses import dataclass
from enum import Enum, auto
import random

from base import Env, Outcome, Agent


class _Color(Enum):
    BLACK = auto()
    RED = auto()


@dataclass(frozen=True)
class Card:
    number: int
    color: _Color

    def value(self) -> int:
        return self.number if self.color == _Color.BLACK else -self.number

    def __str__(self) -> str:
        color_str = "B" if self.color == _Color.BLACK else "R"
        return f"{self.number}{color_str}"


random.seed(42)  # for reproducibility


def random_card(color: _Color | None = None) -> Card:
    """Draw a random card from the deck."""
    card = random.randint(1, 10)
    ret_color = (
        color
        or random.choices([_Color.BLACK, _Color.RED], weights=[2 / 3, 1 / 3], k=1)[0]
    )

    c = Card(number=card, color=ret_color)
    return c


@dataclass(frozen=True)
class Easy21State:
    player_sum: int
    dealer_sum: int
    is_terminal: bool = False

    def __str__(self) -> str:
        return f"(P:{self.player_sum},D:{self.dealer_sum},T:{self.is_terminal})"


class Easy21Action(Enum):
    HIT = auto()
    STICK = auto()


class Easy21Env(Env[Easy21State, Easy21Action]):

    def init_state(self) -> Easy21State:
        dealer_card = self.draw_card(_Color.BLACK)
        player_card = self.draw_card(_Color.BLACK)
        return Easy21State(
            player_sum=player_card.value(), dealer_sum=dealer_card.value()
        )

    def draw_card(self, color: _Color | None = None) -> Card:
        c = random_card(color)
        return c

    def step_impl(self, s: Easy21State, action: Easy21Action) -> Outcome[Easy21State]:
        if s.is_terminal:
            raise ValueError("Cannot take action in terminal state")

        if action == Easy21Action.HIT:
            card = self.draw_card()
            new_player_sum = s.player_sum + card.value()
            if new_player_sum < 1 or new_player_sum > 21:
                return Outcome(
                    next_state=Easy21State(s.player_sum, s.dealer_sum, True),
                    reward=-1.0,
                    done=True,
                )
            else:
                return Outcome(
                    next_state=Easy21State(new_player_sum, s.dealer_sum, False),
                    reward=0.0,
                    done=False,
                )

        elif action == Easy21Action.STICK:
            dealer_sum = s.dealer_sum
            while 1 <= dealer_sum < 17:
                card = self.draw_card()
                dealer_sum += card.value()
            if dealer_sum < 1 or dealer_sum > 21:
                return Outcome(
                    next_state=Easy21State(s.player_sum, dealer_sum, True),
                    reward=1.0,
                    done=True,
                )
            elif dealer_sum > s.player_sum:
                # dealer wins
                return Outcome(
                    next_state=Easy21State(s.player_sum, dealer_sum, True),
                    reward=-1.0,
                    done=True,
                )
            elif dealer_sum < s.player_sum:
                # player wins
                return Outcome(
                    next_state=Easy21State(s.player_sum, dealer_sum, True),
                    reward=1.0,
                    done=True,
                )
            else:
                # draw
                return Outcome(
                    next_state=Easy21State(s.player_sum, dealer_sum, True),
                    reward=0.0,
                    done=True,
                )

        else:
            raise ValueError("Invalid action")


class NaiveAgent(Agent[Easy21State, Easy21Action]):

    def act(self, s: Easy21State) -> Easy21Action:
        if s.player_sum >= 20:
            return Easy21Action.STICK
        else:
            return Easy21Action.HIT
