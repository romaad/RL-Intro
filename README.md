Implementation of the algorithms defined in https://davidstarsilver.wordpress.com/wp-content/uploads/2025/04
With multiple problems for training

To Run:

- create a venv using `venv <name>`
- `source <name>/bin/activate`
- `pyhon main.py --episodes 10000 --no-plot`

check lastrun.md for latest results produced

UI game logging:

- Web UI games are now logged as one JSON file per game under `logs/ui_games/`.
- Override folder with `UI_GAME_LOG_DIR`, for example:
	`UI_GAME_LOG_DIR=/tmp/rl-ui-logs /home/ramadan/code/rl-intro/rlvenv/bin/python web_ui.py`
