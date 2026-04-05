import unittest

from web_ui import app


class TarneebApiTests(unittest.TestCase):
    def test_new_game_includes_round_total_and_names(self) -> None:
        with app.test_client() as client:
            res = client.post("/api/tarneeb/new", json={"player_name": "Ramadan"})
            self.assertEqual(res.status_code, 200)
            data = res.get_json()

            self.assertIn("score", data)
            self.assertIn("round_score", data)
            self.assertEqual(data["score"], [0, 0])
            self.assertEqual(data["round_score"], [0, 0])

            self.assertIn("player_names", data)
            self.assertIn("team_names", data)
            self.assertEqual(len(data["player_names"]), 4)
            self.assertEqual(len(data["team_names"]), 2)
            self.assertEqual(data["player_names"][0], "Ramadan")
            self.assertIn("Ramadan", data["team_names"][0])

            self.assertIn("last_trick", data)
            self.assertIn("play_events", data)
            self.assertIn("bid_events", data)
            self.assertIn("trick_before", data)

    def test_action_response_keeps_score_and_name_metadata(self) -> None:
        with app.test_client() as client:
            new_data = client.post(
                "/api/tarneeb/new", json={"player_name": "Ramadan"}
            ).get_json()

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
            data = action_res.get_json()

            self.assertIn("score", data)
            self.assertIn("round_score", data)
            self.assertIn("player_names", data)
            self.assertIn("team_names", data)
            self.assertIn("bid_events", data)
            self.assertIsInstance(data["bid_events"], list)
            self.assertEqual(data["player_names"][0], "Ramadan")
            self.assertEqual(data["team_names"], new_data["team_names"])


if __name__ == "__main__":
    unittest.main()
