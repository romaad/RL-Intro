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

app = Flask(__name__)
app.secret_key = os.environ.get("SECRET_KEY", os.urandom(24))


# ---------------------------------------------------------------------------
# Helpers
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


if __name__ == "__main__":
    # Set the SECRET_KEY environment variable for a persistent session key.
    # Without it a new random key is generated on every restart (invalidating sessions).
    debug = os.environ.get("FLASK_DEBUG", "0") == "1"
    app.run(debug=debug, port=5000)
