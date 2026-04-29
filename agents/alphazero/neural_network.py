import os

import numpy as np
import torch
import torch.nn as nn

from .model import Model


class NeuralNetwork:
    """Wrapper around a :class:`Model` providing a predict / train interface.

    The model is expected to accept two tensor arguments in ``forward``::

        p_logits, v = model(card_grid, global_features)

    where
    * ``card_grid``       has shape ``(N, H, W, C)`` (NHWC)
    * ``global_features`` has shape ``(N, G)``
    * ``p_logits``        has shape ``(N, action_size)``
    * ``v``               has shape ``(N, num_players)``
    """

    def __init__(
        self,
        model: Model,
        lr: float = 0.001,
        device: str = "cpu",
    ) -> None:
        self._model = model.to(device)
        self._device = device
        self._optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        self._value_loss_fn = nn.MSELoss()

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def predict(
        self, card_grid: np.ndarray, global_features: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """Predict the policy distribution and per-player values for a single state.

        Args:
            card_grid:       shape (H, W, C) float32 — e.g. (4, 13, 3) for Tarneeb
            global_features: shape (G,) float32

        Returns:
            policy_dist: shape (action_size,) softmax probabilities
            values:      shape (num_players,) tanh values in [-1, 1]
        """
        self._model.eval()
        with torch.no_grad():
            x = (
                torch.tensor(card_grid, dtype=torch.float32)
                .unsqueeze(0)
                .to(self._device)
            )
            g = (
                torch.tensor(global_features, dtype=torch.float32)
                .unsqueeze(0)
                .to(self._device)
            )
            p_logits, v = self._model(x, g)
            policy = torch.softmax(p_logits[0], dim=0).cpu().numpy()
            values = v[0].cpu().numpy()
        return policy, values

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def train(
        self,
        buffer: list[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]],
    ) -> float:
        """Run one gradient update step on a mini-batch.

        Args:
            buffer: list of (card_grid, global_features, policy_target, value_target)
                    where policy_target has shape (action_size,) and
                    value_target has shape (num_players,).

        Returns:
            scalar total loss for this update.
        """
        self._model.train()
        card_grids = torch.tensor(
            np.stack([b[0] for b in buffer]), dtype=torch.float32
        ).to(self._device)
        global_feats = torch.tensor(
            np.stack([b[1] for b in buffer]), dtype=torch.float32
        ).to(self._device)
        policy_targets = torch.tensor(
            np.stack([b[2] for b in buffer]), dtype=torch.float32
        ).to(self._device)
        value_targets = torch.tensor(
            np.stack([b[3] for b in buffer]), dtype=torch.float32
        ).to(self._device)

        self._optimizer.zero_grad()
        p_logits, v = self._model(card_grids, global_feats)

        # Policy loss: cross-entropy with soft targets
        policy_loss = -(
            policy_targets * torch.log_softmax(p_logits, dim=1)
        ).sum(dim=1).mean()

        # Value loss: MSE per player
        value_loss = self._value_loss_fn(v, value_targets)

        total_loss = policy_loss + value_loss
        total_loss.backward()
        self._optimizer.step()
        return float(total_loss.item())

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: str) -> None:
        """Save model and optimiser state to *path*."""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save(
            {
                "model_state_dict": self._model.state_dict(),
                "optimizer_state_dict": self._optimizer.state_dict(),
            },
            path,
        )

    def load(self, path: str) -> None:
        """Restore model and optimiser state from *path*."""
        checkpoint = torch.load(path, weights_only=True)
        self._model.load_state_dict(checkpoint["model_state_dict"])
        self._optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
