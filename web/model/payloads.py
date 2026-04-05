from dataclasses import dataclass
from typing import Any, Mapping

from easy21.easy21 import Easy21State
from envs.tarneeb.env import DeckCard, Suit, TarneebState

_SUIT_SYMBOLS = {
    "HEARTS": "♥",
    "DIAMONDS": "♦",
    "CLUBS": "♣",
    "SPADES": "♠",
}

_FACE_LABELS = {1: "A", 11: "J", 12: "Q", 13: "K"}

JsonDict = dict[str, Any]


@dataclass(frozen=True)
class Easy21StateJson:
    player_sum: int
    dealer_sum: int
    dealer_first_card: int
    is_terminal: bool

    @staticmethod
    def from_state(state: Easy21State) -> "Easy21StateJson":
        return Easy21StateJson(
            player_sum=state.player_sum,
            dealer_sum=state.dealer_sum,
            dealer_first_card=state.dealer_first_card,
            is_terminal=state.is_terminal,
        )

    @staticmethod
    def from_dict(data: Mapping[str, Any]) -> "Easy21StateJson":
        return Easy21StateJson(
            player_sum=int(data["player_sum"]),
            dealer_sum=int(data["dealer_sum"]),
            dealer_first_card=int(data["dealer_first_card"]),
            is_terminal=bool(data["is_terminal"]),
        )

    def to_state(self) -> Easy21State:
        return Easy21State(
            player_sum=self.player_sum,
            dealer_sum=self.dealer_sum,
            dealer_first_card=self.dealer_first_card,
            is_terminal=self.is_terminal,
        )

    def to_dict(self) -> JsonDict:
        return {
            "player_sum": self.player_sum,
            "dealer_sum": self.dealer_sum,
            "dealer_first_card": self.dealer_first_card,
            "is_terminal": self.is_terminal,
        }


@dataclass(frozen=True)
class CardJson:
    suit: str
    number: int
    display: str
    valid: bool | None = None
    player: int | None = None

    @staticmethod
    def from_card(card: DeckCard) -> "CardJson":
        num = card.number()
        label = _FACE_LABELS.get(num, str(num))
        suit_name = card.suit.name
        return CardJson(
            suit=suit_name,
            number=num,
            display=label + _SUIT_SYMBOLS[suit_name],
        )

    @staticmethod
    def from_dict(data: Mapping[str, Any]) -> "CardJson":
        return CardJson(
            suit=str(data["suit"]),
            number=int(data["number"]),
            display=str(data.get("display", "")),
            valid=(None if "valid" not in data else bool(data["valid"])),
            player=(None if "player" not in data else int(data["player"])),
        )

    def to_card(self) -> DeckCard:
        return DeckCard(Suit[self.suit], self.number)

    def with_valid(self, valid: bool) -> "CardJson":
        return CardJson(
            suit=self.suit,
            number=self.number,
            display=self.display,
            valid=valid,
            player=self.player,
        )

    def with_player(self, player: int) -> "CardJson":
        return CardJson(
            suit=self.suit,
            number=self.number,
            display=self.display,
            valid=self.valid,
            player=player,
        )

    def to_dict(self) -> JsonDict:
        data: JsonDict = {
            "suit": self.suit,
            "number": self.number,
            "display": self.display,
        }
        if self.valid is not None:
            data["valid"] = self.valid
        if self.player is not None:
            data["player"] = self.player
        return data


@dataclass(frozen=True)
class BidJson:
    value: int
    suit: str

    def to_tuple(self) -> tuple[int, Suit]:
        return (self.value, Suit[self.suit])

    @staticmethod
    def from_tuple(bid: tuple[int, Suit]) -> "BidJson":
        value, suit = bid
        return BidJson(value=value, suit=suit.name)

    @staticmethod
    def from_dict(data: Mapping[str, Any]) -> "BidJson":
        return BidJson(value=int(data["value"]), suit=str(data["suit"]))

    def to_dict(self) -> JsonDict:
        return {"value": self.value, "suit": self.suit}


@dataclass(frozen=True)
class Easy21NewResponseJson:
    state: Easy21StateJson
    wins: int
    losses: int
    draws: int

    def to_dict(self) -> JsonDict:
        return {
            "state": self.state.to_dict(),
            "wins": self.wins,
            "losses": self.losses,
            "draws": self.draws,
        }


@dataclass(frozen=True)
class Easy21ActionResponseJson:
    state: Easy21StateJson
    dealer_first_card: int | None
    result: str | None
    reward: float
    wins: int
    losses: int
    draws: int

    def to_dict(self) -> JsonDict:
        return {
            "state": self.state.to_dict(),
            "dealer_first_card": self.dealer_first_card,
            "result": self.result,
            "reward": self.reward,
            "wins": self.wins,
            "losses": self.losses,
            "draws": self.draws,
        }


@dataclass(frozen=True)
class TarneebSessionStateJson:
    played_cards: list[CardJson]
    holding_cards: list[list[CardJson]]
    trump_suit: str | None
    suit_selected: bool
    passes_count: int
    double_by: int | None
    score: list[int]
    round_num: int
    last_player_idx: int | None
    round_score: list[int]
    current_high_bid: int
    bidder: int | None
    bids: list[BidJson | None]

    @staticmethod
    def from_state(state: TarneebState) -> "TarneebSessionStateJson":
        return TarneebSessionStateJson(
            played_cards=[CardJson.from_card(c) for c in state.played_cards],
            holding_cards=[
                [CardJson.from_card(c) for c in hand] for hand in state.holding_cards
            ],
            trump_suit=state.trump_suit.name if state.trump_suit else None,
            suit_selected=state.suit_selected,
            passes_count=state.passes_count,
            double_by=state.double_by,
            score=list(state.score),
            round_num=state.round_num,
            last_player_idx=state.last_player_idx,
            round_score=list(state.round_score),
            current_high_bid=state.current_high_bid,
            bidder=state.bidder,
            bids=[BidJson.from_tuple(b) if b is not None else None for b in state.bids],
        )

    @staticmethod
    def from_dict(data: Mapping[str, Any]) -> "TarneebSessionStateJson":
        return TarneebSessionStateJson(
            played_cards=[CardJson.from_dict(c) for c in data["played_cards"]],
            holding_cards=[
                [CardJson.from_dict(c) for c in hand] for hand in data["holding_cards"]
            ],
            trump_suit=(
                None if data["trump_suit"] is None else str(data["trump_suit"])
            ),
            suit_selected=bool(data["suit_selected"]),
            passes_count=int(data["passes_count"]),
            double_by=(None if data["double_by"] is None else int(data["double_by"])),
            score=[int(x) for x in data["score"]],
            round_num=int(data["round_num"]),
            last_player_idx=(
                None
                if data["last_player_idx"] is None
                else int(data["last_player_idx"])
            ),
            round_score=[int(x) for x in data["round_score"]],
            current_high_bid=int(data["current_high_bid"]),
            bidder=(None if data["bidder"] is None else int(data["bidder"])),
            bids=[None if b is None else BidJson.from_dict(b) for b in data["bids"]],
        )

    def to_state(self) -> TarneebState:
        return TarneebState(
            played_cards=[c.to_card() for c in self.played_cards],
            holding_cards=[[c.to_card() for c in hand] for hand in self.holding_cards],
            trump_suit=Suit[self.trump_suit] if self.trump_suit else None,
            suit_selected=self.suit_selected,
            passes_count=self.passes_count,
            double_by=self.double_by,
            score=(self.score[0], self.score[1]),
            round_num=self.round_num,
            last_player_idx=self.last_player_idx,
            round_score=(self.round_score[0], self.round_score[1]),
            current_high_bid=self.current_high_bid,
            bidder=self.bidder,
            bids=[b.to_tuple() if b is not None else None for b in self.bids],
        )

    def to_dict(self) -> JsonDict:
        return {
            "played_cards": [c.to_dict() for c in self.played_cards],
            "holding_cards": [
                [c.to_dict() for c in hand] for hand in self.holding_cards
            ],
            "trump_suit": self.trump_suit,
            "suit_selected": self.suit_selected,
            "passes_count": self.passes_count,
            "double_by": self.double_by,
            "score": self.score,
            "round_num": self.round_num,
            "last_player_idx": self.last_player_idx,
            "round_score": self.round_score,
            "current_high_bid": self.current_high_bid,
            "bidder": self.bidder,
            "bids": [b.to_dict() if b is not None else None for b in self.bids],
        }


@dataclass(frozen=True)
class TarneebClientStateJson:
    hand: list[CardJson]
    trump_suit: str | None
    trump_suit_display: str | None
    current_high_bid_suit: str | None
    current_high_bid_suit_display: str | None
    phase: str
    score: list[int]
    round_score: list[int]
    round_num: int
    current_high_bid: int
    bidder: int | None
    bidder_label: str | None
    current_player: int
    player_card_counts: list[int]
    trick: list[CardJson]
    last_trick: list[CardJson]
    player_labels: list[str]
    player_names: list[str]
    team_names: list[str]
    double_by: int | None
    suit_selected: bool

    def to_dict(self) -> JsonDict:
        return {
            "hand": [c.to_dict() for c in self.hand],
            "trump_suit": self.trump_suit,
            "trump_suit_display": self.trump_suit_display,
            "current_high_bid_suit": self.current_high_bid_suit,
            "current_high_bid_suit_display": self.current_high_bid_suit_display,
            "phase": self.phase,
            "score": self.score,
            "round_score": self.round_score,
            "round_num": self.round_num,
            "current_high_bid": self.current_high_bid,
            "bidder": self.bidder,
            "bidder_label": self.bidder_label,
            "current_player": self.current_player,
            "player_card_counts": self.player_card_counts,
            "trick": [c.to_dict() for c in self.trick],
            "last_trick": [c.to_dict() for c in self.last_trick],
            "player_labels": self.player_labels,
            "player_names": self.player_names,
            "team_names": self.team_names,
            "double_by": self.double_by,
            "suit_selected": self.suit_selected,
        }


@dataclass(frozen=True)
class TarneebApiResponseJson:
    state: TarneebClientStateJson
    done: bool
    result: str | None
    wins: int
    losses: int
    play_events: list[CardJson] | None = None
    bid_events: list[JsonDict] | None = None
    trick_before: list[CardJson] | None = None

    def to_dict(self) -> JsonDict:
        data = self.state.to_dict()
        data["done"] = self.done
        data["result"] = self.result
        data["wins"] = self.wins
        data["losses"] = self.losses
        if self.play_events is not None:
            data["play_events"] = [c.to_dict() for c in self.play_events]
        if self.bid_events is not None:
            data["bid_events"] = self.bid_events
        if self.trick_before is not None:
            data["trick_before"] = [c.to_dict() for c in self.trick_before]
        return data
