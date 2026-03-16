from dataclasses import dataclass, replace
from enum import Enum
import random

from base import MultiAgentOutcome, MultipleAgentEnv
from utils import none_throws


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

    @property
    def suit(self) -> Suit:
        """Get the suit of the card."""
        return self._suit

    def __post_init__(self) -> None:
        if self._number < 1 or self._number > 13:
            raise ValueError("Card number must be between 1 and 13.")


class TarneebGameActions(Enum):
    PASS = 1
    DOUBLE = 2


@dataclass(frozen=True)
class TarneebSate:
    # previously played cards in the current round
    played_cards: list[DeckCard]
    # cards that the player is holding
    # per player (index 0-3)
    holding_cards: list[list[DeckCard]]
    trump_suit: Suit | None
    suit_selected: bool
    passes_count: int
    double_by: int | None
    score: tuple[int, int]  # team 0 and team 1 scores
    round_num: int
    # index of the last player who played a card or selected suit
    last_player_idx: int | None
    # score to track the round score
    round_score: tuple[int, int]

    def __str__(self) -> str:
        return (
            f"(P:{self.played_cards},"
            f"T:{self.trump_suit},"
            f"H:{self.holding_cards})"
        )


@dataclass(frozen=True)
class PartialTarneebState:
    played_cards: list[DeckCard]
    holding_cards: list[DeckCard]
    trump_suit: Suit | None
    double_by: int | None
    score: tuple[int, int]
    round_num: int
    round_score: tuple[int, int]

    def __str__(self) -> str:
        return f"(P:{self.played_cards},H:{self.holding_cards},T:{self.trump_suit})"


# we just have to play a card from our holding cards
TarneebAction = DeckCard | Suit | TarneebGameActions
GAME_OVER_SCORE = 31


def next_agent(current_agent_idx: int) -> int:
    return (current_agent_idx + 1) % 4


class TarneebEnv(MultipleAgentEnv[TarneebSate, PartialTarneebState, TarneebAction]):

    def _create_deck(self) -> list[DeckCard]:
        deck = []
        for suit in Suit:
            for num in range(1, 14):
                deck.append(DeckCard(suit, num))
        return deck

    def __init__(self) -> None:
        self.reset()
        super().__init__()

    def reset(self) -> None:
        pass

    def _no_reward(self) -> list[float]:
        return [0.0, 0.0, 0.0, 0.0]

    def _wrong_move_reward(self, agent_idx: int) -> list[float]:
        rewards = [0.0, 0.0, 0.0, 0.0]
        rewards[agent_idx] = -1.0
        return rewards

    def _calc_score(
        self,
        last_player: int,
        played_cards: list[DeckCard],
        trump_suit: Suit | None,
    ) -> tuple[int, int]:
        # The max card score wins the round, if there's a trump suit, trump cards beat others
        # it takes precedence over non-trump cards
        winning_card = played_cards[0]
        winning_player_idx = (last_player - 3) % 4  # first player in the round
        for i, card in enumerate(played_cards[1:], start=1):
            if (
                card.suit == winning_card.suit and card.value() > winning_card.value()
            ) or (
                trump_suit
                and card.suit == trump_suit
                and winning_card.suit != trump_suit
            ):
                winning_card = card
                winning_player_idx = (last_player - i) % 4
        # winning player gets a point for their team
        if winning_player_idx % 2 == 0:
            return (1, 0)
        else:
            return (0, 1)

    def _calc_reward(self, round_score: tuple[int, int]) -> list[float]:
        return [
            float(num) for num in list(round_score) * 2
        ]  # each team has two players

    def _calc_round_score(self, s: TarneebSate) -> tuple[int, int]:
        return (s.score[0] + s.round_score[0], s.score[1] + s.round_score[1])

    def _invalid_move_outcome(
        self, s: TarneebSate, agent_idx: int
    ) -> MultiAgentOutcome[TarneebSate]:
        return MultiAgentOutcome(
            next_state=s,
            reward_per_agent=self._wrong_move_reward(agent_idx),
            done=True,
            next_agent_idx=0,
        )

    def _select_suit(
        self, s: TarneebSate, suit: Suit, agent_idx: int
    ) -> MultiAgentOutcome[TarneebSate]:
        if not s.suit_selected:
            if s.trump_suit and s.trump_suit.value.value >= suit.value.value:
                # can't select a lower-value trump suit
                return self._invalid_move_outcome(s, agent_idx)
            # upping trump suit
            return MultiAgentOutcome(
                next_state=replace(
                    s,
                    played_cards=s.played_cards,
                    holding_cards=s.holding_cards,
                    trump_suit=suit,
                    last_player_idx=agent_idx,
                ),
                reward_per_agent=self._no_reward(),
                done=False,
                next_agent_idx=next_agent(agent_idx),
            )
        # can't select suit again
        return self._invalid_move_outcome(s, agent_idx)

    def _game_action(
        self, s: TarneebSate, action: TarneebGameActions, agent_idx: int
    ) -> MultiAgentOutcome[TarneebSate]:
        if action == TarneebGameActions.PASS:
            if not s.suit_selected:
                if s.passes_count == 3:
                    new_state = replace(s, suit_selected=True)
                    return MultiAgentOutcome(
                        next_state=new_state,
                        reward_per_agent=self._wrong_move_reward(agent_idx),
                        done=True,
                        # the current agent selected the suit, so they get to start
                        next_agent_idx=agent_idx,
                    )
                if s.passes_count < 3:
                    new_state = replace(s, passes_count=s.passes_count + 1)
                    return MultiAgentOutcome(
                        next_state=new_state,
                        reward_per_agent=self._no_reward(),
                        done=False,
                        next_agent_idx=next_agent(agent_idx),
                    )
            # can't pass after suit is selected
            return self._invalid_move_outcome(s, agent_idx)

        if action == TarneebGameActions.DOUBLE:
            if not s.suit_selected:
                new_state = replace(s, suit_selected=True, double_by=agent_idx)
                return MultiAgentOutcome(
                    next_state=new_state,
                    reward_per_agent=self._no_reward(),
                    done=False,
                    next_agent_idx=next_agent(agent_idx),
                )
            # can't double before suit is selected
            return self._invalid_move_outcome(s, agent_idx)

    def _get_updated_round_score(
        self, curr_score: tuple[int, int], hand_score: tuple[int, int]
    ) -> tuple[int, int]:
        return (
            curr_score[0] + hand_score[0],
            curr_score[1] + hand_score[1],
        )

    def _round_done(self, s: TarneebSate) -> bool:
        # round is done when all players have no cards left
        return all(len(cards) == 0 for cards in s.holding_cards)

    def play_card(
        self, s: TarneebSate, played_card: DeckCard, agent_idx: int
    ) -> MultiAgentOutcome[TarneebSate]:
        # playing a card
        new_holding = s.holding_cards.copy()
        new_holding[agent_idx].remove(played_card)
        played_cards = s.played_cards + [played_card]
        if len(played_cards) % 4 == 0:
            # A hand is over
            hand_score = self._calc_score(agent_idx, played_cards, s.trump_suit)
            new_state = replace(
                s,
                played_cards=[],
                holding_cards=new_holding,
                last_player_idx=agent_idx,
                round_score=self._get_updated_round_score(s.round_score, hand_score),
            )
            if self._round_done(new_state):
                # Might be better to return rewards only at the end of the game
                round_score = self._calc_round_score(new_state)
                reward_per_player = self._calc_reward(round_score)

                if any(team_score >= GAME_OVER_SCORE for team_score in round_score):
                    # game over
                    return MultiAgentOutcome(
                        next_state=new_state,
                        reward_per_agent=reward_per_player,
                        done=True,
                        next_agent_idx=0,
                    )
                # start new round
                new_state = replace(
                    self.init_state(), score=round_score, round_num=s.round_num + 1
                )
                return MultiAgentOutcome(
                    next_state=new_state,
                    reward_per_agent=reward_per_player,
                    done=False,
                    next_agent_idx=0,
                )

            return MultiAgentOutcome(
                next_state=new_state,
                reward_per_agent=self._no_reward(),
                done=False,
                next_agent_idx=agent_idx,
            )
        else:
            # continue the round
            new_state = replace(
                s,
                played_cards=played_cards,
                holding_cards=new_holding,
                last_player_idx=agent_idx,
            )
            return MultiAgentOutcome(
                next_state=new_state,
                reward_per_agent=self._no_reward(),
                done=False,
                next_agent_idx=next_agent(agent_idx),
            )

    def to_partial_state(self, s: TarneebSate, agent_idx: int) -> PartialTarneebState:
        return PartialTarneebState(
            played_cards=s.played_cards,
            holding_cards=s.holding_cards[agent_idx],
            trump_suit=s.trump_suit,
            double_by=s.double_by,
            score=s.score,
            round_num=s.round_num,
            round_score=s.round_score,
        )

    def agent_step(
        self, s: TarneebSate, action: TarneebAction, agent_idx: int
    ) -> MultiAgentOutcome[TarneebSate]:

        if isinstance(action, Suit):
            # stage one - selecting trump suit
            return self._select_suit(s, action, agent_idx)

        if isinstance(action, TarneebGameActions):
            return self._game_action(s, action, agent_idx)

        # action is DeckCard
        # stage two - playing a card
        if not s.suit_selected:
            # can't play a card before selecting trump suit
            return self._invalid_move_outcome(s, agent_idx)
        return self.play_card(s, action, agent_idx)

    def init_state(self) -> TarneebSate:
        deck = self._create_deck()
        random.shuffle(deck)
        holding_cards = [deck[i * 13 : (i + 1) * 13] for i in range(4)]
        return TarneebSate(
            played_cards=[],
            holding_cards=holding_cards,
            trump_suit=None,
            suit_selected=False,
            passes_count=0,
            double_by=None,
            score=(0, 0),
            round_num=1,
            last_player_idx=None,
            round_score=(0, 0),
        )
