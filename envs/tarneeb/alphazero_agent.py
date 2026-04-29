"""AlphaZero agent for the 4-player Tarneeb card game.

The agent uses MCTS guided by a shared SENet to choose actions. Because
MCTS requires the *full* game state (to simulate opponent moves) while the
base :class:`Agent` interface only provides a *partial* state, the runner
calls :meth:`set_full_state` with the complete :class:`TarneebState` before
each :meth:`act` call.

Episode lifecycle
-----------------
1. Runner calls ``agent.set_full_state(state)`` → stores full state.
2. Runner calls ``agent.act(partial_state)`` →
   a. runs MCTS simulations from the full state,
   b. records ``(card_grid, global_feats, policy_dist)`` in the episode buffer,
   c. samples an action from the MCTS policy.
3. Runner calls ``agent.on_episode_end(final_rewards_all_agents)`` →
   fills in value targets and pushes complete tuples to the shared replay buffer.
"""

from __future__ import annotations

import numpy as np

from base import Agent, Step
from envs.tarneeb.alphazero_encoding import (
    ACTION_SIZE,
    available_actions_mask,
    encode_global_features,
    index_to_action,
    partial_state_to_numpy,
)
from envs.tarneeb.env import PartialTarneebState, TarneebAction, TarneebState

from agents.alphazero.neural_network import NeuralNetwork
from agents.alphazero.tarneeb_mcts import TarneebMCTS

# Normalisation constant for value targets
_REWARD_NORM = 31.0


class AlphaZeroTarneebAgent(Agent[PartialTarneebState, TarneebAction]):
    """AlphaZero agent for Tarneeb — MCTS + shared SENet.

    Args:
        agent_idx:      player index (0–3)
        nn:             shared :class:`NeuralNetwork` instance
        mcts:           :class:`TarneebMCTS` instance (one per agent to keep
                        separate visit-count trees)
        replay_buffer:  shared list appended to at episode end; pass the same
                        object to all agents and the trainer
        temperature:    MCTS policy temperature; 1.0 for training (exploration),
                        near 0 for evaluation (greedy)
    """

    AGENT_STATE_T = type(None)

    def __init__(
        self,
        agent_idx: int,
        nn: NeuralNetwork,
        mcts: TarneebMCTS,
        replay_buffer: list,
        temperature: float = 1.0,
    ) -> None:
        self._agent_idx = agent_idx
        self._nn = nn
        self._mcts = mcts
        self._replay_buffer = replay_buffer
        self._temperature = temperature

        # Set by the runner before each act() call
        self._current_full_state: TarneebState | None = None
        # Per-episode buffer: list of (card_grid, global_feats, policy_dist)
        self._episode_buffer: list[tuple] = []

    # ------------------------------------------------------------------
    # Runner hooks
    # ------------------------------------------------------------------

    def set_full_state(self, state: TarneebState) -> None:
        """Called by :class:`AlphaZeroMultiAgentRunner` before :meth:`act`."""
        self._current_full_state = state

    def on_episode_end(self, final_rewards: list[float]) -> None:
        """Fill in value targets and push training tuples to the replay buffer.

        Args:
            final_rewards: per-agent cumulative rewards at game end (length 4)
        """
        value_targets = np.clip(
            np.array(final_rewards, dtype=np.float32) / _REWARD_NORM, -1.0, 1.0
        )
        for card_grid, global_feats, policy_dist in self._episode_buffer:
            self._replay_buffer.append(
                (card_grid, global_feats, policy_dist, value_targets.copy())
            )
        self._episode_buffer.clear()
        # Reset MCTS tree for the next episode
        self._mcts.reset()

    # ------------------------------------------------------------------
    # Agent interface
    # ------------------------------------------------------------------

    def act(self, s: PartialTarneebState) -> TarneebAction:
        assert (
            self._current_full_state is not None
        ), "set_full_state() must be called before act()"
        full_state = self._current_full_state

        # MCTS policy from the full state
        policy_dist = self._mcts.get_action_distribution(
            full_state, self._agent_idx, temperature=self._temperature
        )

        # Record state for training (encoded from partial state — no info leak)
        card_grid = partial_state_to_numpy(s)
        global_feats = encode_global_features(s, self._agent_idx)
        self._episode_buffer.append((card_grid, global_feats, policy_dist.copy()))

        # Sample action from masked MCTS policy
        valid = available_actions_mask(s)
        masked = policy_dist * valid
        total = masked.sum()
        if total > 0:
            masked /= total
            action_idx = int(np.random.choice(ACTION_SIZE, p=masked))
        elif valid.sum() > 0:
            valid_indices = np.where(valid)[0]
            action_idx = int(np.random.choice(valid_indices))
        else:
            action_idx = 0  # fallback (should never happen in practice)

        return index_to_action(action_idx)

    def update(self, steps: list[Step]) -> None:
        """No-op — AlphaZero uses :meth:`on_episode_end` for learning."""

    def checkpoint(self) -> None:
        """No-op — checkpointing is handled by the trainer via NeuralNetwork."""

    def restore(self) -> None:
        """No-op — model loading is handled by the trainer via NeuralNetwork."""
