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
        face = {11: "J", 12: "Q", 13: "K"}.get(self._number, str(self._number))
        return f"{self._suit.value.abbv}{face}"

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
class BidAction:
    value: int
    suit: Suit


@dataclass(frozen=True)
class TarneebState:
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
    # bidding
    current_high_bid: int
    bidder: int | None
    bids: list[tuple[int, Suit] | None]

    def __str__(self) -> str:
        return (
            f"(P:{self.played_cards},"
            f"T:{self.trump_suit},"
            f"H:{self.holding_cards})"
        )

    def __hash__(self) -> int:
        # Convert lists to tuples for hashing
        return hash(
            (
                tuple(self.played_cards),
                tuple(tuple(cards) for cards in self.holding_cards),
                self.trump_suit,
                self.suit_selected,
                self.passes_count,
                self.double_by,
                self.score,
                self.round_num,
                self.last_player_idx,
                self.round_score,
                self.current_high_bid,
                self.bidder,
                tuple(self.bids),
            )
        )

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, TarneebState):
            return False
        return (
            self.played_cards == other.played_cards
            and self.holding_cards == other.holding_cards
            and self.trump_suit == other.trump_suit
            and self.suit_selected == other.suit_selected
            and self.passes_count == other.passes_count
            and self.double_by == other.double_by
            and self.score == other.score
            and self.round_num == other.round_num
            and self.last_player_idx == other.last_player_idx
            and self.round_score == other.round_score
            and self.current_high_bid == other.current_high_bid
            and self.bidder == other.bidder
            and self.bids == other.bids
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
    current_high_bid: int
    bidder: int | None
    last_player_idx: int | None

    def __str__(self) -> str:
        return f"(P:{self.played_cards},H:{self.holding_cards},T:{self.trump_suit})"

    def __hash__(self) -> int:
        # Convert lists to tuples for hashing
        return hash(
            (
                tuple(self.played_cards),
                tuple(self.holding_cards),
                self.trump_suit,
                self.double_by,
                self.score,
                self.round_num,
                self.round_score,
                self.current_high_bid,
                self.bidder,
                self.last_player_idx,
            )
        )

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, PartialTarneebState):
            return False
        return (
            self.played_cards == other.played_cards
            and self.holding_cards == other.holding_cards
            and self.trump_suit == other.trump_suit
            and self.double_by == other.double_by
            and self.score == other.score
            and self.round_num == other.round_num
            and self.round_score == other.round_score
            and self.current_high_bid == other.current_high_bid
            and self.bidder == other.bidder
            and self.last_player_idx == other.last_player_idx
        )


# we just have to play a card from our holding cards
TarneebAction = DeckCard | TarneebGameActions | BidAction
GAME_OVER_SCORE = 31


def next_agent(current_agent_idx: int) -> int:
    return (current_agent_idx + 1) % 4


class TarneebEnv(MultipleAgentEnv[TarneebState, PartialTarneebState, TarneebAction]):

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
    ) -> tuple[int, tuple[int, int]]:
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
                winning_player_idx = ((last_player - 3) + i) % 4
        # winning player gets a point for their team
        if winning_player_idx % 2 == 0:
            return winning_player_idx, (1, 0)
        else:
            return winning_player_idx, (0, 1)

    def _calc_reward(self, round_score: tuple[int, int]) -> list[float]:
        return [
            float(num) for num in list(round_score) * 2
        ]  # each team has two players

    def _calc_round_score(self, s: TarneebState) -> tuple[int, int]:
        return (s.score[0] + s.round_score[0], s.score[1] + s.round_score[1])

    def _invalid_move_outcome(
        self, s: TarneebState, agent_idx: int
    ) -> MultiAgentOutcome[TarneebState]:
        return MultiAgentOutcome(
            next_state=s,
            reward_per_agent=self._wrong_move_reward(agent_idx),
            done=True,
            next_agent_idx=0,
        )

    def _select_suit(
        self, s: TarneebState, suit: Suit, agent_idx: int
    ) -> MultiAgentOutcome[TarneebState]:
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
        self, s: TarneebState, action: TarneebGameActions | BidAction, agent_idx: int
    ) -> MultiAgentOutcome[TarneebState]:
        if isinstance(action, BidAction):
            if not s.suit_selected:
                bid_value = action.value
                bid_suit = action.suit
                if bid_value > s.current_high_bid and 7 <= bid_value <= 13:
                    new_bids = s.bids.copy()
                    new_bids[agent_idx] = (bid_value, bid_suit)
                    new_state = replace(
                        s,
                        current_high_bid=bid_value,
                        bidder=agent_idx,
                        bids=new_bids,
                        last_player_idx=agent_idx,
                        passes_count=0,  # reset passes on bid
                    )
                    if all(b is not None for b in new_state.bids):
                        new_state = replace(
                            new_state, suit_selected=True, trump_suit=bid_suit
                        )
                    return MultiAgentOutcome(
                        next_state=new_state,
                        reward_per_agent=self._no_reward(),
                        done=False,
                        next_agent_idx=(
                            next_agent(agent_idx)
                            if not new_state.suit_selected
                            else (
                                none_throws(new_state.bidder)
                                if new_state.trump_suit is not None
                                else 0
                            )
                        ),
                    )
                else:
                    return self._invalid_move_outcome(s, agent_idx)
            return self._invalid_move_outcome(s, agent_idx)

        if action == TarneebGameActions.PASS:
            if not s.suit_selected:
                new_passes = s.passes_count + 1
                new_state = replace(
                    s,
                    passes_count=new_passes,
                    last_player_idx=agent_idx,
                )
                if new_passes == 4:
                    new_state = replace(new_state, suit_selected=True, trump_suit=None)
                return MultiAgentOutcome(
                    next_state=new_state,
                    reward_per_agent=self._no_reward(),
                    done=False,
                    next_agent_idx=(
                        next_agent(agent_idx)
                        if not new_state.suit_selected
                        else (
                            none_throws(new_state.bidder)
                            if new_state.trump_suit is not None
                            else 0
                        )
                    ),
                )
            # can't pass after suit is selected
            return self._invalid_move_outcome(s, agent_idx)

        if action == TarneebGameActions.DOUBLE:
            if s.suit_selected and s.double_by is None:
                new_state = replace(s, double_by=agent_idx)
                return MultiAgentOutcome(
                    next_state=new_state,
                    reward_per_agent=self._no_reward(),
                    done=False,
                    next_agent_idx=none_throws(s.bidder),
                )
            # can't double before suit is selected or if already doubled
            return self._invalid_move_outcome(s, agent_idx)

    def _get_updated_round_score(
        self, curr_score: tuple[int, int], hand_score: tuple[int, int]
    ) -> tuple[int, int]:
        return (
            curr_score[0] + hand_score[0],
            curr_score[1] + hand_score[1],
        )

    def _round_done(self, s: TarneebState) -> bool:
        # round is done when all players have no cards left
        return all(len(cards) == 0 for cards in s.holding_cards)

    def play_card(
        self, s: TarneebState, played_card: DeckCard, agent_idx: int
    ) -> MultiAgentOutcome[TarneebState]:
        # playing a card
        new_holding = s.holding_cards.copy()
        new_holding[agent_idx].remove(played_card)
        # Check follow suit rule
        if len(s.played_cards) > 0:
            led_suit = s.played_cards[0].suit
            has_led_suit = any(c.suit == led_suit for c in s.holding_cards[agent_idx])
            if has_led_suit and played_card.suit != led_suit:
                return self._invalid_move_outcome(s, agent_idx)
        played_cards = s.played_cards + [played_card]
        if len(played_cards) % 4 == 0:
            # A hand is over
            hand_winner, hand_score = self._calc_score(
                agent_idx, played_cards, s.trump_suit
            )
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
                next_agent_idx=hand_winner,
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

    def to_partial_state(self, s: TarneebState, agent_idx: int) -> PartialTarneebState:
        return PartialTarneebState(
            played_cards=s.played_cards,
            holding_cards=s.holding_cards[agent_idx],
            trump_suit=s.trump_suit,
            double_by=s.double_by,
            score=s.score,
            round_num=s.round_num,
            round_score=s.round_score,
            current_high_bid=s.current_high_bid,
            bidder=s.bidder,
            last_player_idx=s.last_player_idx,
        )

    def agent_step(
        self, s: TarneebState, action: TarneebAction, agent_idx: int
    ) -> MultiAgentOutcome[TarneebState]:

        if isinstance(action, Suit):
            # stage one - selecting trump suit
            return self._select_suit(s, action, agent_idx)

        if isinstance(action, (TarneebGameActions, BidAction)):
            return self._game_action(s, action, agent_idx)

        # action is DeckCard
        # stage two - playing a card
        if not s.suit_selected:
            # can't play a card before selecting trump suit
            return self._invalid_move_outcome(s, agent_idx)
        return self.play_card(s, action, agent_idx)

    def init_state(self) -> TarneebState:
        deck = self._create_deck()
        random.shuffle(deck)
        holding_cards = [deck[i * 13 : (i + 1) * 13] for i in range(4)]
        return TarneebState(
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
            current_high_bid=6,  # start below 7
            bidder=None,
            bids=[None] * 4,
        )
