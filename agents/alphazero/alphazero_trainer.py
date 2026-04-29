"""AlphaZero self-play trainer for Tarneeb.

Workflow per iteration
-----------------------
1. Run *episodes_per_iter* self-play episodes using
   :class:`AlphaZeroMultiAgentRunner` (which feeds the full state to each
   agent before ``act()``).
2. Agents push ``(card_grid, global_feats, policy_dist, value_target)``
   tuples into the shared ``replay_buffer``.
3. If the buffer has enough entries, run *train_steps_per_iter* gradient
   updates on randomly-sampled mini-batches.
4. Save a checkpoint every *checkpoint_every* iterations.
"""

from __future__ import annotations

import os
import random

import numpy as np

from base import AlphaZeroMultiAgentRunner
from envs.tarneeb.alphazero_agent import AlphaZeroTarneebAgent
from envs.tarneeb.env import TarneebEnv

from .neural_network import NeuralNetwork


class AlphaZeroTrainer:
    """Runs the AlphaZero self-play / training loop for Tarneeb.

    Args:
        env:                 :class:`TarneebEnv` instance
        nn:                  shared :class:`NeuralNetwork`
        agents:              list of 4 :class:`AlphaZeroTarneebAgent` instances
                             all sharing the same *nn* and *replay_buffer*
        replay_buffer:       shared list populated by agents; also passed to
                             ``agents`` at construction time
        replay_buffer_max_size: maximum number of tuples to retain (FIFO)
        batch_size:          mini-batch size for gradient updates
        checkpoint_path:     directory for ``.pt`` checkpoints
    """

    def __init__(
        self,
        env: TarneebEnv,
        nn: NeuralNetwork,
        agents: list[AlphaZeroTarneebAgent],
        replay_buffer: list,
        replay_buffer_max_size: int = 10_000,
        batch_size: int = 256,
        checkpoint_path: str = "agents_pickes",
    ) -> None:
        self._env = env
        self._nn = nn
        self._agents = agents
        self._replay_buffer = replay_buffer
        self._replay_buffer_max_size = replay_buffer_max_size
        self._batch_size = batch_size
        self._checkpoint_path = checkpoint_path
        self._runner = AlphaZeroMultiAgentRunner()

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------

    def run(
        self,
        num_iterations: int = 100,
        episodes_per_iter: int = 10,
        train_steps_per_iter: int = 20,
        checkpoint_every: int = 10,
    ) -> None:
        """Run the full policy-iteration loop.

        Args:
            num_iterations:       total training iterations
            episodes_per_iter:    self-play episodes per iteration
            train_steps_per_iter: gradient update steps per iteration
            checkpoint_every:     save a checkpoint every N iterations
        """
        for iteration in range(num_iterations):
            # ---- Self-play ----
            for _ in range(episodes_per_iter):
                self._runner.run_episode(self._env, self._agents)

            # Trim replay buffer
            if len(self._replay_buffer) > self._replay_buffer_max_size:
                del self._replay_buffer[
                    : len(self._replay_buffer) - self._replay_buffer_max_size
                ]

            # ---- Training ----
            avg_loss = 0.0
            if len(self._replay_buffer) >= self._batch_size:
                for _ in range(train_steps_per_iter):
                    batch = random.sample(self._replay_buffer, self._batch_size)
                    avg_loss += self._nn.train(batch)
                avg_loss /= train_steps_per_iter

            print(
                f"[AlphaZero] iter {iteration:4d}  "
                f"buf={len(self._replay_buffer)}  "
                f"loss={avg_loss:.4f}"
            )

            # ---- Checkpoint ----
            if checkpoint_every > 0 and (iteration + 1) % checkpoint_every == 0:
                path = os.path.join(
                    self._checkpoint_path,
                    f"alphazero_tarneeb_{iteration}.pt",
                )
                self._nn.save(path)
                print(f"[AlphaZero] checkpoint saved → {path}")

        # Always save a final checkpoint as "latest"
        latest = os.path.join(self._checkpoint_path, "alphazero_tarneeb_latest.pt")
        self._nn.save(latest)
        print(f"[AlphaZero] training complete. Latest checkpoint → {latest}")

    # ------------------------------------------------------------------
    # Convenience helpers
    # ------------------------------------------------------------------

    def _sample_batch(self) -> list:
        return random.sample(self._replay_buffer, self._batch_size)
