import json
import os
import tempfile
import unittest
from pathlib import Path

from web_ui import app


class UiGameLoggingTests(unittest.TestCase):
    def test_easy21_creates_per_game_log_file(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            previous = os.environ.get("UI_GAME_LOG_DIR")
            os.environ["UI_GAME_LOG_DIR"] = tmpdir
            try:
                with app.test_client() as client:
                    new_res = client.post("/api/easy21/new")
                    self.assertEqual(new_res.status_code, 200)

                    with client.session_transaction() as sess:
                        log_id = sess.get("easy21_log_id")
                    self.assertIsInstance(log_id, str)

                    action_res = client.post("/api/easy21/action", json={"action": "hit"})
                    self.assertEqual(action_res.status_code, 200)

                    log_path = Path(tmpdir) / f"{log_id}.json"
                    self.assertTrue(log_path.exists())

                    data = json.loads(log_path.read_text(encoding="utf-8"))
                    self.assertEqual(data["game_type"], "easy21")
                    self.assertIn("initial_state", data)
                    self.assertIn("events", data)
                    self.assertGreaterEqual(len(data["events"]), 1)
                    self.assertIn("action", data["events"][0])
            finally:
                if previous is None:
                    os.environ.pop("UI_GAME_LOG_DIR", None)
                else:
                    os.environ["UI_GAME_LOG_DIR"] = previous

    def test_tarneeb_creates_per_game_log_file(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            previous = os.environ.get("UI_GAME_LOG_DIR")
            os.environ["UI_GAME_LOG_DIR"] = tmpdir
            try:
                with app.test_client() as client:
                    new_res = client.post(
                        "/api/tarneeb/new", json={"player_name": "Ramadan"}
                    )
                    self.assertEqual(new_res.status_code, 200)

                    with client.session_transaction() as sess:
                        log_id = sess.get("tarneeb_log_id")
                    self.assertIsInstance(log_id, str)

                    action_res = client.post(
                        "/api/tarneeb/action",
                        json={
                            "action": {
                                "kind": "bid",
                                "value": 7,
                                "suit": "SPADES",
                            }
                        },
                    )
                    self.assertEqual(action_res.status_code, 200)

                    log_path = Path(tmpdir) / f"{log_id}.json"
                    self.assertTrue(log_path.exists())

                    data = json.loads(log_path.read_text(encoding="utf-8"))
                    self.assertEqual(data["game_type"], "tarneeb")
                    self.assertIn("metadata", data)
                    self.assertIn("events", data)
                    self.assertGreaterEqual(len(data["events"]), 1)
                    self.assertIn("action", data["events"][0])
                    self.assertIn("state_after", data["events"][0])
            finally:
                if previous is None:
                    os.environ.pop("UI_GAME_LOG_DIR", None)
                else:
                    os.environ["UI_GAME_LOG_DIR"] = previous


if __name__ == "__main__":
    unittest.main()
