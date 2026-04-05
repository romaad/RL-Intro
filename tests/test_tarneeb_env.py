from dataclasses import replace
import unittest

from envs.tarneeb.env import BidAction, Suit, TarneebEnv, TarneebGameActions


class TarneebEnvRulesTests(unittest.TestCase):
    def setUp(self) -> None:
        self.env = TarneebEnv()

    def test_double_only_during_bidding(self) -> None:
        state = self.env.init_state()

        # Before any bid, double is invalid.
        out_pre_bid = self.env.agent_step(state, TarneebGameActions.DOUBLE, 0)
        self.assertTrue(out_pre_bid.done)

        # After a bid while still in bidding stage, double is valid.
        out_bid = self.env.agent_step(state, BidAction(7, Suit.SPADES), 0)
        bidding_state = out_bid.next_state
        out_bidding_double = self.env.agent_step(
            bidding_state, TarneebGameActions.DOUBLE, 1
        )
        self.assertFalse(out_bidding_double.done)
        self.assertEqual(out_bidding_double.next_state.double_by, 1)

        # During playing stage, double is invalid.
        playing_state = replace(bidding_state, suit_selected=True)
        out_playing_double = self.env.agent_step(
            playing_state, TarneebGameActions.DOUBLE, 2
        )
        self.assertTrue(out_playing_double.done)

    def test_round_scoring_caller_rules_without_double(self) -> None:
        base = self.env.init_state()

        # Caller team 0, called 8, collected 9 -> caller gets collected tricks.
        made_call = replace(
            base,
            score=(0, 0),
            round_score=(9, 4),
            bidder=0,
            current_high_bid=8,
            double_by=None,
        )
        self.assertEqual(self.env._calc_round_score(made_call), (9, 4))

        # Caller team 0, called 8, collected 6 -> caller gets -8, other gets their tricks.
        failed_call = replace(
            base,
            score=(0, 0),
            round_score=(6, 7),
            bidder=0,
            current_high_bid=8,
            double_by=None,
        )
        self.assertEqual(self.env._calc_round_score(failed_call), (-8, 7))

    def test_round_scoring_with_double(self) -> None:
        base = self.env.init_state()

        # Caller makes call with double: caller collected score is doubled.
        made_call_double = replace(
            base,
            score=(0, 0),
            round_score=(9, 4),
            bidder=0,
            current_high_bid=8,
            double_by=1,
        )
        self.assertEqual(self.env._calc_round_score(made_call_double), (18, 4))

        # Caller fails with double: caller loses double call, winners get double tricks.
        failed_call_double = replace(
            base,
            score=(0, 0),
            round_score=(6, 7),
            bidder=0,
            current_high_bid=8,
            double_by=1,
        )
        self.assertEqual(self.env._calc_round_score(failed_call_double), (-16, 14))

    def test_bidder_starts_play_after_three_passes(self) -> None:
        state = self.env.init_state()

        # Player 1 makes the winning bid.
        out_bid = self.env.agent_step(state, BidAction(7, Suit.SPADES), 1)
        self.assertFalse(out_bid.next_state.suit_selected)
        self.assertEqual(out_bid.next_agent_idx, 2)

        # Three consecutive passes should close bidding.
        out_pass_2 = self.env.agent_step(
            out_bid.next_state, TarneebGameActions.PASS, 2
        )
        out_pass_3 = self.env.agent_step(
            out_pass_2.next_state, TarneebGameActions.PASS, 3
        )
        out_pass_0 = self.env.agent_step(
            out_pass_3.next_state, TarneebGameActions.PASS, 0
        )

        self.assertTrue(out_pass_0.next_state.suit_selected)
        self.assertEqual(out_pass_0.next_state.trump_suit, Suit.SPADES)
        self.assertEqual(out_pass_0.next_agent_idx, 1)


if __name__ == "__main__":
    unittest.main()
