"""Smoke tests for the AlphaZero Tarneeb implementation.

Tests verify:
1. State encoding produces the correct shape.
2. Action encoding / decoding round-trips correctly.
3. Available-actions mask has the right number of True entries per phase.
4. TarneebSENet forward pass produces the correct output shapes.
5. NeuralNetwork.predict returns a valid policy and 4 values in [-1, 1].
6. TarneebMCTS runs N simulations without crashing.
7. AlphaZeroTarneebAgent.act() returns a valid TarneebAction in bidding and playing phases.
"""

import unittest
from dataclasses import replace

import numpy as np

from envs.tarneeb.env import (
    BidAction,
    DeckCard,
    PartialTarneebState,
    Suit,
    TarneebEnv,
    TarneebGameActions,
    TarneebState,
)
from envs.tarneeb.alphazero_encoding import (
    ACTION_LIST,
    ACTION_SIZE,
    GLOBAL_FEATURES_SIZE,
    action_to_index,
    available_actions_mask,
    encode_global_features,
    index_to_action,
    partial_state_to_numpy,
)
from envs.tarneeb.alphazero_model import TarneebSENet
from agents.alphazero.neural_network import NeuralNetwork
from agents.alphazero.tarneeb_mcts import TarneebMCTS
from envs.tarneeb.alphazero_agent import AlphaZeroTarneebAgent


def _fresh_partial(env: TarneebEnv, agent_idx: int = 0) -> PartialTarneebState:
    state = env.init_state()
    return env.to_partial_state(state, agent_idx)


def _fresh_full(env: TarneebEnv) -> TarneebState:
    return env.init_state()


def _fresh_nn() -> NeuralNetwork:
    model = TarneebSENet()
    return NeuralNetwork(model)


# ---------------------------------------------------------------------------
# Encoding tests
# ---------------------------------------------------------------------------


class TestStateEncoding(unittest.TestCase):
    def setUp(self) -> None:
        self.env = TarneebEnv()

    def test_card_grid_shape(self) -> None:
        partial = _fresh_partial(self.env)
        grid = partial_state_to_numpy(partial)
        self.assertEqual(grid.shape, (4, 13, 3))
        self.assertEqual(grid.dtype, np.float32)

    def test_holding_cards_encoded_in_channel_0(self) -> None:
        partial = _fresh_partial(self.env)
        grid = partial_state_to_numpy(partial)
        # Each agent holds 13 cards → exactly 13 ones in channel 0
        self.assertEqual(int(grid[:, :, 0].sum()), 13)

    def test_no_played_cards_at_start(self) -> None:
        partial = _fresh_partial(self.env)
        grid = partial_state_to_numpy(partial)
        self.assertEqual(int(grid[:, :, 1].sum()), 0)

    def test_trump_channel_zero_during_bidding(self) -> None:
        partial = _fresh_partial(self.env)
        grid = partial_state_to_numpy(partial)
        self.assertEqual(int(grid[:, :, 2].sum()), 0)

    def test_trump_channel_full_row_after_trump_set(self) -> None:
        partial = _fresh_partial(self.env)
        partial_with_trump = replace(partial, trump_suit=Suit.HEARTS)
        grid = partial_state_to_numpy(partial_with_trump)
        # Entire HEARTS row (index 0, all 13 ranks) should be 1 in channel 2
        self.assertEqual(int(grid[0, :, 2].sum()), 13)
        self.assertEqual(int(grid[1:, :, 2].sum()), 0)

    def test_global_features_shape(self) -> None:
        partial = _fresh_partial(self.env)
        feats = encode_global_features(partial, agent_idx=0)
        self.assertEqual(feats.shape, (GLOBAL_FEATURES_SIZE,))
        self.assertEqual(feats.dtype, np.float32)

    def test_global_features_agent_one_hot(self) -> None:
        partial = _fresh_partial(self.env)
        for i in range(4):
            feats = encode_global_features(partial, agent_idx=i)
            self.assertEqual(feats[7 + i], 1.0)
            for j in range(4):
                if j != i:
                    self.assertEqual(feats[7 + j], 0.0)


# ---------------------------------------------------------------------------
# Action encoding round-trip
# ---------------------------------------------------------------------------


class TestActionEncoding(unittest.TestCase):
    def test_action_size(self) -> None:
        self.assertEqual(len(ACTION_LIST), ACTION_SIZE)
        self.assertEqual(ACTION_SIZE, 89)

    def test_round_trip_all_actions(self) -> None:
        for i, a in enumerate(ACTION_LIST):
            self.assertEqual(action_to_index(a), i)
            self.assertEqual(index_to_action(i), a)

    def test_deck_card_indices(self) -> None:
        # All 52 cards should map to indices 0..51
        from itertools import product
        _SUIT_ORDER = [Suit.HEARTS, Suit.DIAMONDS, Suit.CLUBS, Suit.SPADES]
        for suit_idx, suit in enumerate(_SUIT_ORDER):
            for number in range(1, 14):
                card = DeckCard(suit, number)
                idx = action_to_index(card)
                self.assertGreaterEqual(idx, 0)
                self.assertLess(idx, 52)
                self.assertEqual(index_to_action(idx), card)

    def test_pass_double_indices(self) -> None:
        self.assertEqual(action_to_index(TarneebGameActions.PASS), 87)
        self.assertEqual(action_to_index(TarneebGameActions.DOUBLE), 88)

    def test_suns_bid_indices(self) -> None:
        for v in range(7, 14):
            idx = action_to_index(BidAction(v, None))
            self.assertEqual(idx, 80 + (v - 7))

    def test_suited_bid_indices(self) -> None:
        _SUIT_ORDER = [Suit.HEARTS, Suit.DIAMONDS, Suit.CLUBS, Suit.SPADES]
        for v in range(7, 14):
            for s_idx, suit in enumerate(_SUIT_ORDER):
                idx = action_to_index(BidAction(v, suit))
                self.assertEqual(idx, 52 + (v - 7) * 4 + s_idx)


# ---------------------------------------------------------------------------
# Available-actions mask tests
# ---------------------------------------------------------------------------


class TestAvailableActionsMask(unittest.TestCase):
    def setUp(self) -> None:
        self.env = TarneebEnv()

    def test_bidding_phase_pass_always_valid(self) -> None:
        partial = _fresh_partial(self.env)
        mask = available_actions_mask(partial)
        self.assertEqual(mask[87], 1.0, "PASS should be valid at start of bidding")

    def test_bidding_phase_double_invalid_before_bid(self) -> None:
        partial = _fresh_partial(self.env)
        mask = available_actions_mask(partial)
        self.assertEqual(mask[88], 0.0, "DOUBLE invalid before any bid")

    def test_bidding_phase_double_valid_after_bid(self) -> None:
        partial = _fresh_partial(self.env)
        partial_with_bid = replace(partial, bidder=0, current_high_bid=7)
        mask = available_actions_mask(partial_with_bid)
        self.assertEqual(mask[88], 1.0, "DOUBLE valid after a bid is placed")

    def test_bidding_phase_no_card_actions(self) -> None:
        partial = _fresh_partial(self.env)
        mask = available_actions_mask(partial)
        self.assertEqual(mask[:52].sum(), 0.0, "No card actions during bidding")

    def test_playing_phase_only_card_actions(self) -> None:
        partial = _fresh_partial(self.env)
        partial_playing = replace(partial, trump_suit=Suit.SPADES)
        mask = available_actions_mask(partial_playing)
        # Bid actions and PASS should be 0
        self.assertEqual(mask[52:88].sum(), 0.0, "No bid/pass during play")
        # Cards in hand should be valid
        self.assertEqual(int(mask[:52].sum()), len(partial_playing.holding_cards))

    def test_playing_phase_follow_suit(self) -> None:
        # Build a state where the agent definitely has hearts and must follow suit
        env = TarneebEnv()
        partial = _fresh_partial(env)
        hearts_in_hand = [c for c in partial.holding_cards if c.suit == Suit.HEARTS]
        if not hearts_in_hand:
            return  # skip if this particular deal has no hearts for agent 0

        # Led card is a heart not in agent's hand (so it stays in played_cards legally)
        hearts_not_in_hand = [
            DeckCard(Suit.HEARTS, n)
            for n in range(1, 14)
            if DeckCard(Suit.HEARTS, n) not in partial.holding_cards
        ]
        if not hearts_not_in_hand:
            return  # extremely unlikely; skip

        partial_must_follow = replace(
            partial,
            trump_suit=Suit.SPADES,
            played_cards=[hearts_not_in_hand[0]],
        )
        mask = available_actions_mask(partial_must_follow)

        # Non-heart cards in hand must NOT be valid
        non_heart_valid = any(
            mask[action_to_index(c)] == 1.0
            for c in partial_must_follow.holding_cards
            if c.suit != Suit.HEARTS
        )
        self.assertFalse(non_heart_valid, "Must follow led suit when holding one")

        # All hearts in hand must be valid
        for c in hearts_in_hand:
            self.assertEqual(
                mask[action_to_index(c)], 1.0, f"{c} should be playable (follow suit)"
            )


# ---------------------------------------------------------------------------
# SENet model tests
# ---------------------------------------------------------------------------


class TestTarneebSENet(unittest.TestCase):
    def setUp(self) -> None:
        import torch
        self.model = TarneebSENet()
        self.model.eval()

    def test_output_shapes(self) -> None:
        import torch
        x = torch.zeros(2, 4, 13, 3)
        g = torch.zeros(2, GLOBAL_FEATURES_SIZE)
        with torch.no_grad():
            p, v = self.model(x, g)
        self.assertEqual(tuple(p.shape), (2, ACTION_SIZE))
        self.assertEqual(tuple(v.shape), (2, 4))

    def test_value_head_tanh_range(self) -> None:
        import torch
        x = torch.randn(4, 4, 13, 3)
        g = torch.randn(4, GLOBAL_FEATURES_SIZE)
        with torch.no_grad():
            _, v = self.model(x, g)
        self.assertTrue((v >= -1.0).all())
        self.assertTrue((v <= 1.0).all())


# ---------------------------------------------------------------------------
# NeuralNetwork wrapper tests
# ---------------------------------------------------------------------------


class TestNeuralNetwork(unittest.TestCase):
    def setUp(self) -> None:
        self.nn = _fresh_nn()
        self.env = TarneebEnv()

    def test_predict_output_shapes(self) -> None:
        partial = _fresh_partial(self.env)
        card_grid = partial_state_to_numpy(partial)
        global_feats = encode_global_features(partial, 0)
        policy, values = self.nn.predict(card_grid, global_feats)
        self.assertEqual(policy.shape, (ACTION_SIZE,))
        self.assertEqual(values.shape, (4,))

    def test_predict_policy_is_distribution(self) -> None:
        partial = _fresh_partial(self.env)
        policy, _ = self.nn.predict(
            partial_state_to_numpy(partial),
            encode_global_features(partial, 0),
        )
        self.assertAlmostEqual(float(policy.sum()), 1.0, places=5)

    def test_predict_values_in_range(self) -> None:
        partial = _fresh_partial(self.env)
        _, values = self.nn.predict(
            partial_state_to_numpy(partial),
            encode_global_features(partial, 0),
        )
        self.assertTrue(np.all(values >= -1.0))
        self.assertTrue(np.all(values <= 1.0))

    def test_train_returns_scalar_loss(self) -> None:
        partial = _fresh_partial(self.env)
        card_grid = partial_state_to_numpy(partial)
        global_feats = encode_global_features(partial, 0)
        policy_target = np.ones(ACTION_SIZE, dtype=np.float32) / ACTION_SIZE
        value_target = np.zeros(4, dtype=np.float32)
        buffer = [(card_grid, global_feats, policy_target, value_target)] * 4
        loss = self.nn.train(buffer)
        self.assertIsInstance(loss, float)
        self.assertGreater(loss, 0.0)


# ---------------------------------------------------------------------------
# MCTS tests
# ---------------------------------------------------------------------------


class TestTarneebMCTS(unittest.TestCase):
    def setUp(self) -> None:
        self.env = TarneebEnv()
        self.nn = _fresh_nn()
        self.mcts = TarneebMCTS(self.env, self.nn, num_simulations=10, cpuct=1.0)

    def test_simulations_run_without_crash(self) -> None:
        state = _fresh_full(self.env)
        dist = self.mcts.get_action_distribution(state, agent_idx=0, temperature=1.0)
        self.assertEqual(dist.shape, (ACTION_SIZE,))

    def test_distribution_is_valid_probability(self) -> None:
        state = _fresh_full(self.env)
        dist = self.mcts.get_action_distribution(state, 0, temperature=1.0)
        self.assertAlmostEqual(float(dist.sum()), 1.0, places=5)

    def test_reset_clears_tree(self) -> None:
        state = _fresh_full(self.env)
        self.mcts.get_action_distribution(state, 0, temperature=1.0)
        self.mcts.reset()
        self.assertEqual(len(self.mcts._Ns), 0)


# ---------------------------------------------------------------------------
# Agent tests
# ---------------------------------------------------------------------------


class TestAlphaZeroAgent(unittest.TestCase):
    def setUp(self) -> None:
        self.env = TarneebEnv()
        self.nn = _fresh_nn()
        replay_buffer: list = []
        self.mcts = TarneebMCTS(self.env, self.nn, num_simulations=5)
        self.agent = AlphaZeroTarneebAgent(
            agent_idx=0,
            nn=self.nn,
            mcts=self.mcts,
            replay_buffer=replay_buffer,
            temperature=1.0,
        )
        self.replay_buffer = replay_buffer

    def test_act_bidding_phase(self) -> None:
        full_state = _fresh_full(self.env)
        partial = self.env.to_partial_state(full_state, 0)
        self.agent.set_full_state(full_state)
        action = self.agent.act(partial)
        valid_types = (BidAction, TarneebGameActions)
        self.assertIsInstance(action, valid_types)

    def test_act_playing_phase(self) -> None:
        full_state = _fresh_full(self.env)
        # Force playing phase by setting suit_selected
        playing_state = replace(full_state, suit_selected=True, trump_suit=Suit.SPADES)
        partial = self.env.to_partial_state(playing_state, 0)
        self.agent.set_full_state(playing_state)
        action = self.agent.act(partial)
        self.assertIsInstance(action, DeckCard)

    def test_act_raises_without_full_state(self) -> None:
        full_state = _fresh_full(self.env)
        partial = self.env.to_partial_state(full_state, 0)
        with self.assertRaises(AssertionError):
            self.agent.act(partial)

    def test_episode_buffer_filled(self) -> None:
        full_state = _fresh_full(self.env)
        partial = self.env.to_partial_state(full_state, 0)
        self.agent.set_full_state(full_state)
        self.agent.act(partial)
        self.assertEqual(len(self.agent._episode_buffer), 1)

    def test_on_episode_end_fills_replay_buffer(self) -> None:
        full_state = _fresh_full(self.env)
        partial = self.env.to_partial_state(full_state, 0)
        self.agent.set_full_state(full_state)
        self.agent.act(partial)
        self.agent.on_episode_end([9.0, 4.0, 9.0, 4.0])
        self.assertEqual(len(self.replay_buffer), 1)
        card_grid, global_feats, policy, value_targets = self.replay_buffer[0]
        self.assertEqual(card_grid.shape, (4, 13, 3))
        self.assertEqual(global_feats.shape, (GLOBAL_FEATURES_SIZE,))
        self.assertEqual(policy.shape, (ACTION_SIZE,))
        self.assertEqual(value_targets.shape, (4,))
        # Values should be clipped to [-1, 1]
        self.assertTrue(np.all(value_targets >= -1.0))
        self.assertTrue(np.all(value_targets <= 1.0))

    def test_episode_buffer_cleared_after_episode_end(self) -> None:
        full_state = _fresh_full(self.env)
        partial = self.env.to_partial_state(full_state, 0)
        self.agent.set_full_state(full_state)
        self.agent.act(partial)
        self.agent.on_episode_end([9.0, 4.0, 9.0, 4.0])
        self.assertEqual(len(self.agent._episode_buffer), 0)


if __name__ == "__main__":
    unittest.main()
