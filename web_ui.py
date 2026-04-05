"""
Web UI for playing RL games against reinforcement learning agents.

Run with:
    python web_ui.py

Then open http://localhost:5000 in your browser.
"""

import argparse
import json
import os
import random
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, cast
from uuid import uuid4

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

_AI_NAME_POOL = [
    "Alice",
    "Bob",
    "Mary",
    "Curie",
    "Mark",
    "Eissa",
    "Mazen",
    "Saied",
    "Amr",
    "Abdo",
    "Andrew",
]


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _ui_log_dir() -> Path:
    configured = os.environ.get("UI_GAME_LOG_DIR", "logs/ui_games")
    base = Path(configured)
    if not base.is_absolute():
        base = Path(__file__).resolve().parent / base
    return base


def _game_log_path(log_id: str) -> Path:
    return _ui_log_dir() / f"{log_id}.json"


def _write_game_log(data: JsonDict, log_id: str) -> None:
    log_path = _game_log_path(log_id)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def _read_game_log(log_id: str) -> JsonDict | None:
    log_path = _game_log_path(log_id)
    if not log_path.exists():
        return None
    with log_path.open("r", encoding="utf-8") as f:
        raw = json.load(f)
    return cast(JsonDict, raw)


def _start_game_log(
    game_type: str,
    initial_state: Mapping[str, Any],
    metadata: Mapping[str, Any] | None = None,
) -> str:
    log_id = f"{game_type}_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S')}_{uuid4().hex[:8]}"
    payload: JsonDict = {
        "log_id": log_id,
        "game_type": game_type,
        "started_at": _now_iso(),
        "status": "in_progress",
        "metadata": dict(metadata or {}),
        "initial_state": dict(initial_state),
        "events": [],
    }
    _write_game_log(payload, log_id)
    return log_id


def _append_game_log_event(log_id: str, event: Mapping[str, Any]) -> None:
    payload = _read_game_log(log_id)
    if payload is None:
        return
    events_raw = payload.get("events", [])
    events: list[JsonDict] = (
        cast(list[JsonDict], events_raw) if isinstance(events_raw, list) else []
    )
    events.append(dict(event))
    payload["events"] = events
    payload["updated_at"] = _now_iso()
    _write_game_log(payload, log_id)


def _finalize_game_log(log_id: str, result: str) -> None:
    payload = _read_game_log(log_id)
    if payload is None:
        return
    payload["status"] = "done"
    payload["result"] = result
    payload["finished_at"] = _now_iso()
    _write_game_log(payload, log_id)


def _normalize_player_name(name: str | None) -> str:
    if name is None:
        return "Player"
    cleaned = name.strip()
    return cleaned if cleaned else "Player"


def _build_player_names(player_name: str | None) -> list[str]:
    human_name = _normalize_player_name(player_name)
    ai_names = random.sample(_AI_NAME_POOL, 3)
    return [human_name, ai_names[0], ai_names[1], ai_names[2]]


def _team_names(player_names: list[str]) -> list[str]:
    return [
        f"{player_names[0]} & {player_names[2]}",
        f"{player_names[1]} & {player_names[3]}",
    ]


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
    try:
        easy21_log_id = _start_game_log(
            game_type="easy21",
            initial_state=_state_to_dict(state),
            metadata={"dealer_first_card": state.dealer_first_card},
        )
        session["easy21_log_id"] = easy21_log_id
    except Exception:
        session["easy21_log_id"] = None

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

    easy21_log_id_raw = session.get("easy21_log_id")
    if isinstance(easy21_log_id_raw, str):
        try:
            _append_game_log_event(
                easy21_log_id_raw,
                {
                    "timestamp": _now_iso(),
                    "action": action_str,
                    "state_before": _state_to_dict(state),
                    "state_after": _state_to_dict(new_state),
                    "reward": float(outcome.reward),
                    "done": bool(outcome.done),
                    "result": result,
                },
            )
            if result is not None:
                _finalize_game_log(easy21_log_id_raw, result)
        except Exception:
            pass

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
    player_names: list[str] | None = None,
) -> TarneebClientStateJson:
    suit_name = state.trump_suit.name if state.trump_suit else None
    current_high_bid_suit_name: str | None = None
    if state.bidder is not None:
        winning_bid = state.bids[state.bidder]
        if winning_bid is not None:
            current_high_bid_suit_name = winning_bid[1].name

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

    if player_names is None or len(player_names) != 4:
        player_names = ["Player", "Right", "Partner", "Left"]

    player_labels = [
        f"You ({player_names[0]})",
        player_names[1],
        player_names[2],
        player_names[3],
    ]
    team_names = _team_names(player_names)

    def card_info(c: DeckCard) -> CardJson:
        return CardJson.from_card(c).with_valid(
            valid_card_set is None or c in valid_card_set
        )

    return TarneebClientStateJson(
        hand=[card_info(c) for c in hand],
        trump_suit=suit_name,
        trump_suit_display=_SUIT_SYMBOLS[suit_name] if suit_name else None,
        current_high_bid_suit=current_high_bid_suit_name,
        current_high_bid_suit_display=(
            _SUIT_SYMBOLS[current_high_bid_suit_name]
            if current_high_bid_suit_name
            else None
        ),
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
        player_names=player_names,
        team_names=team_names,
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


def _bid_event(
    action: DeckCard | TarneebGameActions | BidAction, agent_idx: int
) -> JsonDict | None:
    if isinstance(action, BidAction):
        return {
            "player": agent_idx,
            "kind": "bid",
            "value": action.value,
            "suit": action.suit.name,
        }
    if action == TarneebGameActions.PASS:
        return {"player": agent_idx, "kind": "pass"}
    if action == TarneebGameActions.DOUBLE:
        return {"player": agent_idx, "kind": "double"}
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
    list[JsonDict],
]:
    ai_agents = [RandomTarneebAgent() for _ in range(4)]
    done = False
    rewards: list[float] = [0.0, 0.0, 0.0, 0.0]
    last_completed_trick: list[CardJson] | None = None
    play_events: list[CardJson] = []
    bid_events: list[JsonDict] = []

    while current_player != 0 and not done:
        prev_state = state
        partial = env.to_partial_state(state, current_player)
        action = ai_agents[current_player].act(partial)
        outcome = env.agent_step(state, action, current_player)
        event = _play_event(action, current_player)
        if event:
            play_events.append(event)
        bid_event = _bid_event(action, current_player)
        if bid_event:
            bid_events.append(bid_event)
        state = outcome.next_state
        maybe_trick = _maybe_completed_trick(prev_state, action, current_player, state)
        if maybe_trick:
            last_completed_trick = maybe_trick
        done = outcome.done
        current_player = outcome.next_agent_idx
        if done:
            rewards = outcome.reward_per_agent

    return (
        state,
        current_player,
        done,
        rewards,
        last_completed_trick,
        play_events,
        bid_events,
    )


# ---------------------------------------------------------------------------
# Tarneeb routes
# ---------------------------------------------------------------------------


@app.route("/tarneeb")
def tarneeb_game():
    return render_template("tarneeb.html")


@app.route("/api/tarneeb/new", methods=["POST"])
def tarneeb_new():
    raw = request.get_json(silent=True)
    request_data: Mapping[str, Any] = (
        cast(Mapping[str, Any], raw) if isinstance(raw, dict) else {}
    )
    player_name_raw = request_data.get("player_name")
    player_name = player_name_raw if isinstance(player_name_raw, str) else None

    env = TarneebEnv()
    state = env.init_state()
    current_player = 0
    player_names = _build_player_names(player_name)

    session["tarneeb_state"] = _tarneeb_state_to_session(state)
    session["tarneeb_current_player"] = current_player
    session["tarneeb_done"] = False
    session["tarneeb_last_trick"] = []
    session["tarneeb_player_names"] = player_names

    tarneeb_initial = _build_tarneeb_client_state(
        state, current_player, [], player_names
    ).to_dict()
    try:
        tarneeb_log_id = _start_game_log(
            game_type="tarneeb",
            initial_state=tarneeb_initial,
            metadata={
                "player_names": player_names,
                "team_names": _team_names(player_names),
            },
        )
        session["tarneeb_log_id"] = tarneeb_log_id
    except Exception:
        session["tarneeb_log_id"] = None

    payload = TarneebApiResponseJson(
        state=_build_tarneeb_client_state(state, current_player, [], player_names),
        play_events=[],
        bid_events=[],
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
    player_names_raw_obj = session.get("tarneeb_player_names", [])
    player_names_raw: list[Any] = (
        cast(list[Any], player_names_raw_obj)
        if isinstance(player_names_raw_obj, list)
        else []
    )
    player_names = (
        [str(x) for x in player_names_raw]
        if len(player_names_raw) == 4
        else ["Player", "Right", "Partner", "Left"]
    )
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
    bid_events: list[JsonDict] = []

    outcome = env.agent_step(state, action, current_player)
    event = _play_event(action, current_player)
    if event:
        play_events.append(event)
    bid_event = _bid_event(action, current_player)
    if bid_event:
        bid_events.append(bid_event)

    state = outcome.next_state
    maybe_trick = _maybe_completed_trick(prev_state, action, current_player, state)
    if maybe_trick:
        last_trick = maybe_trick

    done = outcome.done
    current_player = outcome.next_agent_idx
    rewards = outcome.reward_per_agent if done else [0.0, 0.0, 0.0, 0.0]

    if not done:
        (
            state,
            current_player,
            done,
            rewards,
            ai_last_trick,
            ai_play_events,
            ai_bid_events,
        ) = _advance_ai_turns(env, state, current_player)
        play_events.extend(ai_play_events)
        bid_events.extend(ai_bid_events)
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
            player_names,
        ),
        play_events=play_events,
        bid_events=bid_events,
        trick_before=trick_before,
        done=done,
        result=result,
        wins=int(session.get("tarneeb_wins", 0)),
        losses=int(session.get("tarneeb_losses", 0)),
    )

    tarneeb_log_id_raw = session.get("tarneeb_log_id")
    if isinstance(tarneeb_log_id_raw, str):
        action_log: JsonDict
        if isinstance(action, DeckCard):
            action_log = {
                "kind": "card",
                "suit": action.suit.name,
                "number": action.number(),
            }
        elif isinstance(action, BidAction):
            action_log = {
                "kind": "bid",
                "value": action.value,
                "suit": action.suit.name,
            }
        elif action == TarneebGameActions.PASS:
            action_log = {"kind": "pass"}
        else:
            action_log = {"kind": "double"}

        try:
            _append_game_log_event(
                tarneeb_log_id_raw,
                {
                    "timestamp": _now_iso(),
                    "action": action_log,
                    "player": 0,
                    "bid_events": bid_events,
                    "play_events": [c.to_dict() for c in play_events],
                    "done": done,
                    "result": result,
                    "state_after": payload.state.to_dict(),
                },
            )
            if result is not None:
                _finalize_game_log(tarneeb_log_id_raw, result)
        except Exception:
            pass

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
