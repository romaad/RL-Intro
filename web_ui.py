"""
Web UI for playing RL games against reinforcement learning agents.

Run with:
    python web_ui.py

Then open http://localhost:5000 in your browser.
"""

import argparse
import os
import sys
from typing import Any, Mapping, cast

from flask import Flask, jsonify, render_template, request, session

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from easy21.easy21 import Easy21Action, Easy21Env, Easy21State  # noqa: E402
from envs.tarneeb.agents import RandomTarneebAgent  # noqa: E402
from envs.tarneeb.env import (  # noqa: E402
    BidAction,
    DeckCard,
    Suit,
    TarneebEnv,
    TarneebGameActions,
    TarneebState,
)
from web.model import (  # noqa: E402
    CardJson,
    Easy21ActionResponseJson,
    Easy21NewResponseJson,
    Easy21StateJson,
    JsonDict,
    TarneebApiResponseJson,
    TarneebClientStateJson,
    TarneebSessionStateJson,
)

app = Flask(__name__)
app.secret_key = os.environ.get("SECRET_KEY", os.urandom(24))

# ---------------------------------------------------------------------------
# Tarneeb constants
# ---------------------------------------------------------------------------

_SUIT_SYMBOLS = {
    "HEARTS": "♥",
    "DIAMONDS": "♦",
    "CLUBS": "♣",
    "SPADES": "♠",
}


# ---------------------------------------------------------------------------
# Easy21 helpers
# ---------------------------------------------------------------------------


def _state_to_dict(state: Easy21State) -> JsonDict:
    return Easy21StateJson.from_state(state).to_dict()


def _dict_to_state(data: Mapping[str, Any]) -> Easy21State:
    return Easy21StateJson.from_dict(data).to_state()


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/easy21")
def easy21_game():
    return render_template("easy21.html")


@app.route("/api/easy21/new", methods=["POST"])
def easy21_new():
    env = Easy21Env()
    state = env.init_state()
    session["easy21_state"] = _state_to_dict(state)
    session["easy21_dealer_first_card"] = state.dealer_first_card
    session["easy21_done"] = False

    payload = Easy21NewResponseJson(
        state=Easy21StateJson.from_state(state),
        wins=int(session.get("wins", 0)),
        losses=int(session.get("losses", 0)),
        draws=int(session.get("draws", 0)),
    )
    return jsonify(payload.to_dict())


@app.route("/api/easy21/action", methods=["POST"])
def easy21_action():
    raw = request.get_json()
    if not isinstance(raw, dict) or "action" not in raw:
        return jsonify({"error": "Action is required"}), 400

    data = cast(Mapping[str, Any], raw)
    action_str = data["action"]
    if action_str == "hit":
        action = Easy21Action.HIT
    elif action_str == "stick":
        action = Easy21Action.STICK
    else:
        return jsonify({"error": f"Invalid action: {action_str}"}), 400

    state_raw = session.get("easy21_state")
    if not isinstance(state_raw, dict):
        return jsonify({"error": "No active game. Start a new game first."}), 400

    if bool(session.get("easy21_done", False)):
        return jsonify({"error": "Game is already over. Start a new game."}), 400

    state = _dict_to_state(cast(Mapping[str, Any], state_raw))
    env = Easy21Env()
    outcome = env.step(state, action)
    new_state = outcome.next_state
    session["easy21_state"] = _state_to_dict(new_state)

    result: str | None = None
    if outcome.done:
        if outcome.reward > 0:
            result = "win"
            session["wins"] = int(session.get("wins", 0)) + 1
        elif outcome.reward < 0:
            result = "lose"
            session["losses"] = int(session.get("losses", 0)) + 1
        else:
            result = "draw"
            session["draws"] = int(session.get("draws", 0)) + 1
        session["easy21_done"] = True

    dealer_first_card_raw = session.get("easy21_dealer_first_card")
    dealer_first_card = (
        None if dealer_first_card_raw is None else int(dealer_first_card_raw)
    )

    payload = Easy21ActionResponseJson(
        state=Easy21StateJson.from_state(new_state),
        dealer_first_card=dealer_first_card,
        result=result,
        reward=float(outcome.reward),
        wins=int(session.get("wins", 0)),
        losses=int(session.get("losses", 0)),
        draws=int(session.get("draws", 0)),
    )
    return jsonify(payload.to_dict())


# ---------------------------------------------------------------------------
# Tarneeb helpers
# ---------------------------------------------------------------------------


def _tarneeb_state_to_session(state: TarneebState) -> JsonDict:
    return TarneebSessionStateJson.from_state(state).to_dict()


def _session_to_tarneeb_state(data: Mapping[str, Any]) -> TarneebState:
    return TarneebSessionStateJson.from_dict(data).to_state()


def _build_tarneeb_client_state(
    state: TarneebState,
    current_player: int,
    last_trick: list[JsonDict] | None = None,
) -> TarneebClientStateJson:
    suit_name = state.trump_suit.name if state.trump_suit else None

    hand = sorted(
        state.holding_cards[0],
        key=lambda c: (c.suit.value.value, -c.number()),
    )

    valid_card_set: set[DeckCard] | None = None
    if state.suit_selected and state.played_cards:
        led_suit = state.played_cards[0].suit
        same_suit = [c for c in state.holding_cards[0] if c.suit == led_suit]
        if same_suit:
            valid_card_set = set(same_suit)

    trick = _build_trick_from_state(state)
    last_trick_cards = [CardJson.from_dict(c) for c in (last_trick or [])]

    player_labels = ["You", "Right", "Partner", "Left"]

    def card_info(c: DeckCard) -> CardJson:
        return CardJson.from_card(c).with_valid(
            valid_card_set is None or c in valid_card_set
        )

    return TarneebClientStateJson(
        hand=[card_info(c) for c in hand],
        trump_suit=suit_name,
        trump_suit_display=_SUIT_SYMBOLS[suit_name] if suit_name else None,
        phase="bidding" if not state.suit_selected else "playing",
        score=list(state.score),
        round_score=list(state.round_score),
        round_num=state.round_num,
        current_high_bid=state.current_high_bid,
        bidder=state.bidder,
        bidder_label=(
            player_labels[state.bidder] if state.bidder is not None else None
        ),
        current_player=current_player,
        player_card_counts=[len(h) for h in state.holding_cards],
        trick=trick,
        last_trick=last_trick_cards,
        player_labels=player_labels,
        double_by=state.double_by,
        suit_selected=state.suit_selected,
    )


def _build_trick_from_state(state: TarneebState) -> list[CardJson]:
    trick: list[CardJson] = []
    if state.played_cards and state.last_player_idx is not None:
        n = len(state.played_cards)
        starting = (state.last_player_idx - n + 1) % 4
        for i, card in enumerate(state.played_cards):
            player = (starting + i) % 4
            trick.append(CardJson.from_card(card).with_player(player))
    return trick


def _play_event(
    action: DeckCard | TarneebGameActions | BidAction, agent_idx: int
) -> CardJson | None:
    if isinstance(action, DeckCard):
        return CardJson.from_card(action).with_player(agent_idx)
    return None


def _maybe_completed_trick(
    prev_state: TarneebState,
    action: DeckCard | TarneebGameActions | BidAction,
    agent_idx: int,
    next_state: TarneebState,
) -> list[CardJson] | None:
    if not isinstance(action, DeckCard):
        return None
    if len(prev_state.played_cards) != 3:
        return None
    if next_state.played_cards:
        return None

    full_trick = prev_state.played_cards + [action]
    starting = (agent_idx - 3) % 4
    trick: list[CardJson] = []
    for i, card in enumerate(full_trick):
        trick.append(CardJson.from_card(card).with_player((starting + i) % 4))
    return trick


def _advance_ai_turns(
    env: TarneebEnv,
    state: TarneebState,
    current_player: int,
) -> tuple[
    TarneebState,
    int,
    bool,
    list[float],
    list[CardJson] | None,
    list[CardJson],
]:
    ai_agents = [RandomTarneebAgent() for _ in range(4)]
    done = False
    rewards: list[float] = [0.0, 0.0, 0.0, 0.0]
    last_completed_trick: list[CardJson] | None = None
    play_events: list[CardJson] = []

    while current_player != 0 and not done:
        prev_state = state
        partial = env.to_partial_state(state, current_player)
        action = ai_agents[current_player].act(partial)
        outcome = env.agent_step(state, action, current_player)
        event = _play_event(action, current_player)
        if event:
            play_events.append(event)
        state = outcome.next_state
        maybe_trick = _maybe_completed_trick(prev_state, action, current_player, state)
        if maybe_trick:
            last_completed_trick = maybe_trick
        done = outcome.done
        current_player = outcome.next_agent_idx
        if done:
            rewards = outcome.reward_per_agent

    return state, current_player, done, rewards, last_completed_trick, play_events


# ---------------------------------------------------------------------------
# Tarneeb routes
# ---------------------------------------------------------------------------


@app.route("/tarneeb")
def tarneeb_game():
    return render_template("tarneeb.html")


@app.route("/api/tarneeb/new", methods=["POST"])
def tarneeb_new():
    env = TarneebEnv()
    state = env.init_state()
    current_player = 0

    session["tarneeb_state"] = _tarneeb_state_to_session(state)
    session["tarneeb_current_player"] = current_player
    session["tarneeb_done"] = False
    session["tarneeb_last_trick"] = []

    payload = TarneebApiResponseJson(
        state=_build_tarneeb_client_state(state, current_player, []),
        play_events=[],
        trick_before=[],
        done=False,
        result=None,
        wins=int(session.get("tarneeb_wins", 0)),
        losses=int(session.get("tarneeb_losses", 0)),
    )
    return jsonify(payload.to_dict())


@app.route("/api/tarneeb/action", methods=["POST"])
def tarneeb_action():
    raw = request.get_json()
    if not isinstance(raw, dict) or "action" not in raw:
        return jsonify({"error": "Action is required"}), 400
    request_data = cast(Mapping[str, Any], raw)

    state_raw = session.get("tarneeb_state")
    if not isinstance(state_raw, dict):
        return jsonify({"error": "No active game. Start a new game first."}), 400

    if bool(session.get("tarneeb_done", False)):
        return jsonify({"error": "Game is already over. Start a new game."}), 400

    current_player = int(session.get("tarneeb_current_player", 0))
    last_trick_data_raw = session.get("tarneeb_last_trick", [])
    last_trick_data = cast(list[Mapping[str, Any]], last_trick_data_raw)
    last_trick: list[CardJson] = [CardJson.from_dict(c) for c in last_trick_data]

    state = _session_to_tarneeb_state(cast(Mapping[str, Any], state_raw))
    env = TarneebEnv()

    action_data_raw = request_data["action"]
    if not isinstance(action_data_raw, dict):
        return jsonify({"error": "Invalid action payload"}), 400

    action_data = cast(Mapping[str, Any], action_data_raw)
    try:
        action = _parse_tarneeb_action(action_data, state)
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400

    prev_state = state
    trick_before = _build_trick_from_state(prev_state)
    play_events: list[CardJson] = []

    outcome = env.agent_step(state, action, current_player)
    event = _play_event(action, current_player)
    if event:
        play_events.append(event)

    state = outcome.next_state
    maybe_trick = _maybe_completed_trick(prev_state, action, current_player, state)
    if maybe_trick:
        last_trick = maybe_trick

    done = outcome.done
    current_player = outcome.next_agent_idx
    rewards = outcome.reward_per_agent if done else [0.0, 0.0, 0.0, 0.0]

    if not done:
        state, current_player, done, rewards, ai_last_trick, ai_play_events = (
            _advance_ai_turns(env, state, current_player)
        )
        play_events.extend(ai_play_events)
        if ai_last_trick:
            last_trick = ai_last_trick

    session["tarneeb_state"] = _tarneeb_state_to_session(state)
    session["tarneeb_current_player"] = current_player
    session["tarneeb_done"] = done
    session["tarneeb_last_trick"] = [c.to_dict() for c in last_trick]

    result: str | None = None
    if done:
        player_reward = rewards[0]
        if player_reward > 0:
            result = "win"
            session["tarneeb_wins"] = int(session.get("tarneeb_wins", 0)) + 1
        elif player_reward < 0:
            result = "lose"
            session["tarneeb_losses"] = int(session.get("tarneeb_losses", 0)) + 1
        else:
            result = "draw"

    payload = TarneebApiResponseJson(
        state=_build_tarneeb_client_state(
            state,
            current_player,
            [c.to_dict() for c in last_trick],
        ),
        play_events=play_events,
        trick_before=trick_before,
        done=done,
        result=result,
        wins=int(session.get("tarneeb_wins", 0)),
        losses=int(session.get("tarneeb_losses", 0)),
    )
    return jsonify(payload.to_dict())


def _parse_tarneeb_action(action_data: Mapping[str, Any], state: TarneebState):
    kind_raw = action_data.get("kind")
    kind = str(kind_raw) if kind_raw is not None else ""

    if kind == "pass":
        return TarneebGameActions.PASS

    if kind == "double":
        return TarneebGameActions.DOUBLE

    if kind == "bid":
        value = action_data.get("value")
        suit_name = action_data.get("suit")
        if not isinstance(value, int) or not isinstance(suit_name, str):
            raise ValueError("Invalid bid action")
        if suit_name not in _SUIT_SYMBOLS:
            raise ValueError("Invalid bid action")
        return BidAction(value=value, suit=Suit[suit_name])

    if kind == "card":
        suit_name = action_data.get("suit")
        number = action_data.get("number")
        if not isinstance(suit_name, str) or not isinstance(number, int):
            raise ValueError("Invalid card action")
        if suit_name not in _SUIT_SYMBOLS:
            raise ValueError("Invalid card action")
        card = DeckCard(Suit[suit_name], number)
        if card not in state.holding_cards[0]:
            raise ValueError("Card not in hand")
        return card

    raise ValueError(f"Unknown action kind: {kind}")


if __name__ == "__main__":
    debug = os.environ.get("FLASK_DEBUG", "0") == "1"
    parser = argparse.ArgumentParser("Web server")
    parser.add_argument(
        "-p", "--port", type=int, default=5001, help="Port to use for the web server"
    )
    args = parser.parse_args()
    app.run(debug=debug, port=args.port)
