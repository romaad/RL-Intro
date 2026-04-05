"""
Web UI for playing RL games against reinforcement learning agents.

Run with:
    python web_ui.py

Then open http://localhost:5000 in your browser.
"""

import os
import sys

from flask import Flask, jsonify, render_template, request, session

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from easy21.easy21 import Easy21Action, Easy21Env, Easy21State  # noqa: E402
from envs.tarneeb.env import (  # noqa: E402
    BidAction,
    DeckCard,
    PartialTarneebState,
    Suit,
    TarneebEnv,
    TarneebGameActions,
    TarneebState,
)
from envs.tarneeb.agents import RandomTarneebAgent  # noqa: E402

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

_FACE_LABELS = {1: "A", 11: "J", 12: "Q", 13: "K"}


# ---------------------------------------------------------------------------
# Easy21 helpers
# ---------------------------------------------------------------------------

def _state_to_dict(state: Easy21State) -> dict:
    return {
        "player_sum": state.player_sum,
        "dealer_sum": state.dealer_sum,
        "dealer_first_card": state.dealer_first_card,
        "is_terminal": state.is_terminal,
    }


def _dict_to_state(d: dict) -> Easy21State:
    return Easy21State(
        player_sum=d["player_sum"],
        dealer_sum=d["dealer_sum"],
        dealer_first_card=d["dealer_first_card"],
        is_terminal=d["is_terminal"],
    )


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
    # Preserve the true first card value; easy21.py's step constructors
    # overwrite dealer_first_card with a boolean via positional arg ordering.
    session["easy21_dealer_first_card"] = state.dealer_first_card
    session["easy21_done"] = False
    state_dict = _state_to_dict(state)
    return jsonify(
        {
            "state": state_dict,
            "wins": session.get("wins", 0),
            "losses": session.get("losses", 0),
            "draws": session.get("draws", 0),
        }
    )


@app.route("/api/easy21/action", methods=["POST"])
def easy21_action():
    data = request.get_json()
    if not data or "action" not in data:
        return jsonify({"error": "Action is required"}), 400

    action_str = data["action"]
    if action_str == "hit":
        action = Easy21Action.HIT
    elif action_str == "stick":
        action = Easy21Action.STICK
    else:
        return jsonify({"error": f"Invalid action: {action_str}"}), 400

    state_dict = session.get("easy21_state")
    if not state_dict:
        return jsonify({"error": "No active game. Start a new game first."}), 400

    if session.get("easy21_done", False):
        return jsonify({"error": "Game is already over. Start a new game."}), 400

    state = _dict_to_state(state_dict)
    env = Easy21Env()
    outcome = env.step(state, action)
    new_state = outcome.next_state
    session["easy21_state"] = _state_to_dict(new_state)

    result = None
    if outcome.done:
        if outcome.reward > 0:
            result = "win"
            session["wins"] = session.get("wins", 0) + 1
        elif outcome.reward < 0:
            result = "lose"
            session["losses"] = session.get("losses", 0) + 1
        else:
            result = "draw"
            session["draws"] = session.get("draws", 0) + 1
        session["easy21_done"] = True

    return jsonify(
        {
            "state": _state_to_dict(new_state),
            "dealer_first_card": session.get("easy21_dealer_first_card"),
            "result": result,
            "reward": outcome.reward,
            "wins": session.get("wins", 0),
            "losses": session.get("losses", 0),
            "draws": session.get("draws", 0),
        }
    )


# ---------------------------------------------------------------------------
# Tarneeb helpers
# ---------------------------------------------------------------------------

def _card_to_dict(card: DeckCard) -> dict:
    num = card.value()
    label = _FACE_LABELS.get(num, str(num))
    suit_name = card.suit.name
    return {
        "suit": suit_name,
        "number": num,
        "display": label + _SUIT_SYMBOLS[suit_name],
    }


def _dict_to_card(d: dict) -> DeckCard:
    return DeckCard(Suit[d["suit"]], d["number"])


def _bid_to_dict(bid: tuple | None) -> dict | None:
    if bid is None:
        return None
    value, suit = bid
    return {"value": value, "suit": suit.name}


def _tarneeb_state_to_session(state: TarneebState) -> dict:
    """Serialize TarneebState to a JSON-compatible dict for session storage."""
    return {
        "played_cards": [_card_to_dict(c) for c in state.played_cards],
        "holding_cards": [
            [_card_to_dict(c) for c in hand] for hand in state.holding_cards
        ],
        "trump_suit": state.trump_suit.name if state.trump_suit else None,
        "suit_selected": state.suit_selected,
        "passes_count": state.passes_count,
        "double_by": state.double_by,
        "score": list(state.score),
        "round_num": state.round_num,
        "last_player_idx": state.last_player_idx,
        "round_score": list(state.round_score),
        "current_high_bid": state.current_high_bid,
        "bidder": state.bidder,
        "bids": [_bid_to_dict(b) for b in state.bids],
    }


def _session_to_tarneeb_state(d: dict) -> TarneebState:
    """Deserialize a TarneebState from session storage."""
    bids = []
    for b in d["bids"]:
        if b is None:
            bids.append(None)
        else:
            bids.append((b["value"], Suit[b["suit"]]))
    return TarneebState(
        played_cards=[_dict_to_card(c) for c in d["played_cards"]],
        holding_cards=[
            [_dict_to_card(c) for c in hand] for hand in d["holding_cards"]
        ],
        trump_suit=Suit[d["trump_suit"]] if d["trump_suit"] else None,
        suit_selected=d["suit_selected"],
        passes_count=d["passes_count"],
        double_by=d["double_by"],
        score=tuple(d["score"]),
        round_num=d["round_num"],
        last_player_idx=d["last_player_idx"],
        round_score=tuple(d["round_score"]),
        current_high_bid=d["current_high_bid"],
        bidder=d["bidder"],
        bids=bids,
    )


def _build_tarneeb_client_state(state: TarneebState, current_player: int) -> dict:
    """Build the JSON response payload shown to the human player (player 0)."""
    suit_name = state.trump_suit.name if state.trump_suit else None

    # Sort player 0's hand by suit value then card number descending (high cards first)
    hand = sorted(
        state.holding_cards[0],
        key=lambda c: (c.suit.value.value, -c.value()),
    )

    # Determine which cards in player 0's hand are valid plays
    valid_card_set: set[DeckCard] | None = None
    if state.suit_selected and state.played_cards:
        led_suit = state.played_cards[0].suit
        same_suit = [c for c in state.holding_cards[0] if c.suit == led_suit]
        if same_suit:
            valid_card_set = set(same_suit)
    # If no restriction (leading a trick or no led suit cards), all cards valid

    # Map played cards to the player who played them within the current trick
    trick: list[dict] = []
    if state.played_cards and state.last_player_idx is not None:
        n = len(state.played_cards)
        starting = (state.last_player_idx - n + 1) % 4
        for i, card in enumerate(state.played_cards):
            player = (starting + i) % 4
            trick.append({**_card_to_dict(card), "player": player})

    player_labels = ["You", "Right", "Partner", "Left"]

    def card_info(c: DeckCard) -> dict:
        d = _card_to_dict(c)
        d["valid"] = valid_card_set is None or c in valid_card_set
        return d

    return {
        "hand": [card_info(c) for c in hand],
        "trump_suit": suit_name,
        "trump_suit_display": _SUIT_SYMBOLS[suit_name] if suit_name else None,
        "phase": "bidding" if not state.suit_selected else "playing",
        "score": list(state.score),
        "round_score": list(state.round_score),
        "round_num": state.round_num,
        "current_high_bid": state.current_high_bid,
        "bidder": state.bidder,
        "bidder_label": player_labels[state.bidder] if state.bidder is not None else None,
        "current_player": current_player,
        "player_card_counts": [len(h) for h in state.holding_cards],
        "trick": trick,
        "player_labels": player_labels,
        "double_by": state.double_by,
        "suit_selected": state.suit_selected,
    }


def _advance_ai_turns(
    env: TarneebEnv,
    state: TarneebState,
    current_player: int,
) -> tuple[TarneebState, int, bool, list[float]]:
    """Run AI agents (players 1-3) until it is player 0's turn or the game is done."""
    ai_agents = [RandomTarneebAgent() for _ in range(4)]
    done = False
    rewards: list[float] = [0.0, 0.0, 0.0, 0.0]

    while current_player != 0 and not done:
        partial = env.to_partial_state(state, current_player)
        action = ai_agents[current_player].act(partial)
        outcome = env.agent_step(state, action, current_player)
        state = outcome.next_state
        done = outcome.done
        current_player = outcome.next_agent_idx
        if done:
            rewards = outcome.reward_per_agent

    return state, current_player, done, rewards


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
    current_player = 0  # bidding starts with player 0

    session["tarneeb_state"] = _tarneeb_state_to_session(state)
    session["tarneeb_current_player"] = current_player
    session["tarneeb_done"] = False

    return jsonify(
        {
            **_build_tarneeb_client_state(state, current_player),
            "done": False,
            "result": None,
            "wins": session.get("tarneeb_wins", 0),
            "losses": session.get("tarneeb_losses", 0),
        }
    )


@app.route("/api/tarneeb/action", methods=["POST"])
def tarneeb_action():
    data = request.get_json()
    if not data or "action" not in data:
        return jsonify({"error": "Action is required"}), 400

    state_data = session.get("tarneeb_state")
    if not state_data:
        return jsonify({"error": "No active game. Start a new game first."}), 400

    if session.get("tarneeb_done", False):
        return jsonify({"error": "Game is already over. Start a new game."}), 400

    current_player = session.get("tarneeb_current_player", 0)
    state = _session_to_tarneeb_state(state_data)
    env = TarneebEnv()

    # Parse the action from the request
    action_data = data["action"]
    try:
        action = _parse_tarneeb_action(action_data, state)
    except ValueError as e:
        return jsonify({"error": str(e)}), 400

    # Apply the human player's action
    outcome = env.agent_step(state, action, current_player)
    state = outcome.next_state
    done = outcome.done
    current_player = outcome.next_agent_idx
    rewards = outcome.reward_per_agent if done else [0.0, 0.0, 0.0, 0.0]

    # Auto-advance AI turns
    if not done:
        state, current_player, done, rewards = _advance_ai_turns(
            env, state, current_player
        )

    session["tarneeb_state"] = _tarneeb_state_to_session(state)
    session["tarneeb_current_player"] = current_player
    session["tarneeb_done"] = done

    result = None
    if done:
        # Player 0 is on team 0 (players 0 and 2)
        player_reward = rewards[0]
        if player_reward > 0:
            result = "win"
            session["tarneeb_wins"] = session.get("tarneeb_wins", 0) + 1
        elif player_reward < 0:
            result = "lose"
            session["tarneeb_losses"] = session.get("tarneeb_losses", 0) + 1
        else:
            result = "draw"

    return jsonify(
        {
            **_build_tarneeb_client_state(state, current_player),
            "done": done,
            "result": result,
            "wins": session.get("tarneeb_wins", 0),
            "losses": session.get("tarneeb_losses", 0),
        }
    )


def _parse_tarneeb_action(action_data: dict, state: TarneebState):
    """Parse an action dict from the client into a TarneebAction."""
    kind = action_data.get("kind")

    if kind == "pass":
        return TarneebGameActions.PASS

    if kind == "double":
        return TarneebGameActions.DOUBLE

    if kind == "bid":
        value = action_data.get("value")
        suit_name = action_data.get("suit")
        if not isinstance(value, int) or suit_name not in _SUIT_SYMBOLS:
            raise ValueError("Invalid bid action")
        return BidAction(value=value, suit=Suit[suit_name])

    if kind == "card":
        suit_name = action_data.get("suit")
        number = action_data.get("number")
        if suit_name not in _SUIT_SYMBOLS or not isinstance(number, int):
            raise ValueError("Invalid card action")
        card = DeckCard(Suit[suit_name], number)
        # Verify the card is in the player's hand
        if card not in state.holding_cards[0]:
            raise ValueError("Card not in hand")
        return card

    raise ValueError(f"Unknown action kind: {kind}")


if __name__ == "__main__":
    # Set the SECRET_KEY environment variable for a persistent session key.
    # Without it a new random key is generated on every restart (invalidating sessions).
    debug = os.environ.get("FLASK_DEBUG", "0") == "1"
    app.run(debug=debug, port=5000)