"""Tarneeb-specific Monte Carlo Tree Search for AlphaZero.

The tree is keyed by ``hash(TarneebState)`` (full game state). At each
node we store:

* ``Ns[s]``      – total visit count for state *s*
* ``Nsa[(s, a)]`` – visit count for (state, action index) pair
* ``Qsa[(s, a)]`` – running-mean Q-value for the *acting player* at *s*
* ``Psa[s]``     – prior policy (shape ``(ACTION_SIZE,)``) from the NN
* ``valid[s]``   – valid-action mask (shape ``(ACTION_SIZE,)``)

Selection uses the PUCT formula::

    U(s, a) = Q(s, a) + c_puct * P(s,a) * sqrt(N(s)) / (1 + N(s, a))

Leaf expansion evaluates the position with the neural network (using the
acting player's partial state as NN input so that no information is leaked).
Values are backed up from the perspective of the acting player at each level.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import numpy as np

from envs.tarneeb.alphazero_encoding import (
    ACTION_SIZE,
    available_actions_mask,
    encode_global_features,
    index_to_action,
    partial_state_to_numpy,
)

if TYPE_CHECKING:
    from agents.alphazero.neural_network import NeuralNetwork
    from envs.tarneeb.env import TarneebEnv, TarneebState

# Normalisation constant for terminal rewards (max cumulative game score)
_REWARD_NORM = 31.0


class TarneebMCTS:
    """MCTS engine for Tarneeb, interfacing directly with :class:`TarneebEnv`."""

    def __init__(
        self,
        env: "TarneebEnv",
        nn: "NeuralNetwork",
        num_simulations: int = 50,
        cpuct: float = 1.0,
    ) -> None:
        self._env = env
        self._nn = nn
        self._num_simulations = num_simulations
        self._cpuct = cpuct

        self._Ns: dict[int, int] = {}
        self._Nsa: dict[tuple[int, int], int] = {}
        self._Qsa: dict[tuple[int, int], float] = {}
        self._Psa: dict[int, np.ndarray] = {}
        self._valid: dict[int, np.ndarray] = {}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def reset(self) -> None:
        """Clear all tree statistics (call at the start of each episode)."""
        self._Ns.clear()
        self._Nsa.clear()
        self._Qsa.clear()
        self._Psa.clear()
        self._valid.clear()

    def get_action_distribution(
        self,
        state: "TarneebState",
        agent_idx: int,
        temperature: float = 1.0,
    ) -> np.ndarray:
        """Run simulations from *state* and return a visit-count policy.

        Args:
            state:       full game state (available inside the episode runner)
            agent_idx:   index of the agent that will act
            temperature: controls exploration; 0 gives argmax (greedy)

        Returns:
            policy distribution of shape ``(ACTION_SIZE,)``
        """
        for _ in range(self._num_simulations):
            self._simulate(state, agent_idx)

        s_hash = hash(state)
        counts = np.array(
            [self._Nsa.get((s_hash, a), 0) for a in range(ACTION_SIZE)],
            dtype=np.float32,
        )

        if temperature == 0:
            best = int(np.argmax(counts))
            pi = np.zeros(ACTION_SIZE, dtype=np.float32)
            pi[best] = 1.0
            return pi

        counts = counts ** (1.0 / temperature)
        total = counts.sum()
        if total > 0:
            return counts / total

        # Fallback: uniform over valid actions
        valid = self._valid.get(s_hash, np.ones(ACTION_SIZE, dtype=np.float32))
        s = valid.sum()
        return valid / s if s > 0 else np.ones(ACTION_SIZE, dtype=np.float32) / ACTION_SIZE

    # ------------------------------------------------------------------
    # Internal simulation
    # ------------------------------------------------------------------

    def _simulate(
        self, state: "TarneebState", agent_idx: int
    ) -> list[float]:
        """Run one simulation from *state* acting as *agent_idx*.

        Returns:
            list of 4 floats — per-player values in [-1, 1]
        """
        s_hash = hash(state)

        if s_hash not in self._Psa:
            # ---- Leaf expansion ----
            partial = self._env.to_partial_state(state, agent_idx)
            card_grid = partial_state_to_numpy(partial)
            global_feats = encode_global_features(partial, agent_idx)
            policy, values = self._nn.predict(card_grid, global_feats)

            # Mask and normalise prior
            valid = available_actions_mask(partial)
            policy = policy * valid
            p_sum = policy.sum()
            if p_sum > 0:
                policy /= p_sum
            else:
                policy = valid / valid.sum() if valid.sum() > 0 else (
                    np.ones(ACTION_SIZE, dtype=np.float32) / ACTION_SIZE
                )

            self._Psa[s_hash] = policy
            self._valid[s_hash] = valid
            self._Ns[s_hash] = 0
            return list(values)

        # ---- Selection via PUCT ----
        valid = self._valid[s_hash]
        policy = self._Psa[s_hash]
        ns = self._Ns[s_hash]

        best_score = -float("inf")
        best_a = -1
        for a in range(ACTION_SIZE):
            if valid[a] == 0:
                continue
            nsa = self._Nsa.get((s_hash, a), 0)
            qsa = self._Qsa.get((s_hash, a), 0.0)
            u = qsa + self._cpuct * policy[a] * math.sqrt(ns + 1e-8) / (1 + nsa)
            if u > best_score:
                best_score = u
                best_a = a

        if best_a < 0:
            return [0.0] * 4  # no valid actions (shouldn't happen)

        action = index_to_action(best_a)
        outcome = self._env.agent_step(state, action, agent_idx)

        if outcome.done:
            values = [
                max(-1.0, min(1.0, r / _REWARD_NORM))
                for r in outcome.reward_per_agent
            ]
        else:
            values = self._simulate(outcome.next_state, outcome.next_agent_idx)

        # ---- Backpropagation ----
        self._Ns[s_hash] = ns + 1
        key = (s_hash, best_a)
        old_n = self._Nsa.get(key, 0)
        new_n = old_n + 1
        self._Nsa[key] = new_n
        old_q = self._Qsa.get(key, 0.0)
        # Running mean Q from acting player's perspective
        self._Qsa[key] = (old_q * old_n + values[agent_idx]) / new_n

        return values
