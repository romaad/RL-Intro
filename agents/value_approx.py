import os
from typing import Callable, Generic
from agents.q_agent import QAgent
from base import Action, State
import numpy as np
import torch
import torch.nn as nn

from pickle_utils import save_pickle


class ValueApproximator(Generic[State, Action]):
    """An abstract base class for value function approximators."""

    def predict(self, s: State, a: Action) -> float:
        """Predict the Q-value for a given state-action pair."""
        raise NotImplementedError

    def update(self, s: State, a: Action, target: float) -> None:
        """Update the approximator based on the target Q-value."""
        raise NotImplementedError


class LinearValueApproximator(ValueApproximator[State, Action]):
    """A simple linear value function approximator."""

    _weights: np.ndarray
    _feature_extractor: Callable[[State, Action], np.ndarray]
    _alpha: float  # learning rate

    def __init__(
        self,
        feature_extractor: Callable[[State, Action], np.ndarray],
        feature_vector_size: int,
        initial_weights: np.ndarray | None = None,
        alpha: float = 0.01,
    ) -> None:
        self._feature_extractor = feature_extractor
        if initial_weights is not None:
            self._weights = initial_weights
        else:
            self._weights = np.zeros(feature_vector_size)
        assert (
            self._weights.shape[0] == feature_vector_size
        ), "Initial weights size does not match feature vector size."
        self._alpha = alpha

    def predict(self, s: State, a: Action) -> float:
        features = self._feature_extractor(s, a)
        return float(np.dot(self._weights, features))

    def update(self, s: State, a: Action, target: float) -> None:
        features = self._feature_extractor(s, a)
        prediction = self.predict(s, a)
        error = target - prediction
        self._weights += (
            (error * features) * self._alpha * features
        )  # Simple gradient descent update

    def get_state(self) -> object:
        return self._weights.copy()


class LinearApproxAgent(QAgent[State, Action]):
    """An agent that uses linear function approximation for value function."""

    _approximator: LinearValueApproximator[State, Action]
    _epsilon: float  # exploration rate

    def __init__(
        self,
        value_approximator: LinearValueApproximator[State, Action],
        epsilon: float = 0.05,
    ) -> None:
        self._approximator = value_approximator
        self._epsilon = epsilon

    def q_value(self, s: State, a: Action) -> float:
        return self._approximator.predict(s, a)

    def update_q_value(self, s: State, a: Action, value: float) -> None:
        self._approximator.update(s, a, value)

    def get_epsilon(self, s: State) -> float:
        return self._epsilon

    def get_state(self) -> object:
        return self._approximator.get_state()

    def checkpoint(self) -> None:
        save_pickle(
            self.get_state(), f"{self.PICKLE_PATH}/{self.name}_w_checkpoint.pkl"
        )


class _CNNNetwork(nn.Module):
    """PyTorch module for the CNN + 3-FC value network.

    Architecture::

        CNN block  (1, C, L)
            ↓  Conv1d(C→F, kernel_size=K)  →  (1, F, L-K+1)
            ↓  ReLU
            ↓  Flatten                      →  F*(L-K+1)
            │
        Other block  (D,)
            │
            ├── concatenate ────────────── →  F*(L-K+1) + D
            ↓
           Linear(combined_dim → hidden1) + ReLU
            ↓
           Linear(hidden1 → hidden2) + ReLU
            ↓
           Linear(hidden2 → hidden3) + ReLU
            ↓
           Linear(hidden3 → 1)  →  scalar Q-value
    """

    def __init__(
        self,
        cnn_input_len: int,
        num_channels: int,
        other_dim: int,
        cnn_filters: int,
        cnn_kernel: int,
        fc_hidden: tuple[int, int, int],
    ) -> None:
        super().__init__()
        cnn_out_dim = cnn_filters * (cnn_input_len - cnn_kernel + 1)
        combined_dim = cnn_out_dim + other_dim
        h1, h2, h3 = fc_hidden

        self.conv = nn.Conv1d(
            in_channels=num_channels,
            out_channels=cnn_filters,
            kernel_size=cnn_kernel,
        )
        self.fc1 = nn.Linear(combined_dim, h1)
        self.fc2 = nn.Linear(h1, h2)
        self.fc3 = nn.Linear(h2, h3)
        self.fc_out = nn.Linear(h3, 1)
        self.relu = nn.ReLU()

        # He (Kaiming) initialization to match ReLU activations throughout the network
        nn.init.kaiming_uniform_(self.conv.weight, nonlinearity="relu")
        nn.init.zeros_(self.conv.bias)
        for layer in (self.fc1, self.fc2, self.fc3, self.fc_out):
            nn.init.kaiming_uniform_(layer.weight, nonlinearity="relu")
            nn.init.zeros_(layer.bias)

    def forward(
        self, cnn_block: torch.Tensor, other_block: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            cnn_block:   shape (1, C, L)  – batched channel-first card features
            other_block: shape (1, D)     – batched scalar features

        Returns:
            shape (1, 1) – Q-value estimate
        """
        x = self.relu(self.conv(cnn_block))       # (1, F, L-K+1)
        x = x.flatten(start_dim=1)                # (1, cnn_out_dim)
        x = torch.cat([x, other_block], dim=1)    # (1, combined_dim)
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.relu(self.fc3(x))
        return self.fc_out(x)                      # (1, 1)


class CNNValueApproximator(ValueApproximator[State, Action]):
    """CNN + 3-hidden-layer value function approximator backed by PyTorch.

    The feature vector produced by *feature_extractor* is split into two parts:

    1. **CNN block** – assembled from *cnn_channel_slices*.  Each element is a
       ``(start, stop)`` index pair selecting a slice of the feature vector.
       All slices must have the same length (*cnn_input_len*) and are stacked
       to form a tensor of shape ``(num_channels, cnn_input_len)``.

    2. **Other block** – features collected from *other_slices* and
       concatenated into a 1-D vector.

    The underlying network is a :class:`_CNNNetwork` (``torch.nn.Module``),
    trained with :class:`torch.optim.Adam` and MSE loss.
    """

    def __init__(
        self,
        feature_extractor: Callable[[State, Action], np.ndarray],
        cnn_input_len: int,
        cnn_channel_slices: list[tuple[int, int]],
        other_slices: list[tuple[int, int]],
        cnn_filters: int = 16,
        cnn_kernel: int = 4,
        fc_hidden: tuple[int, int, int] = (256, 128, 64),
        alpha: float = 0.001,
        seed: int = 42,
    ) -> None:
        torch.manual_seed(seed)
        self._feature_extractor = feature_extractor
        self._cnn_channel_slices = cnn_channel_slices
        self._other_slices = other_slices

        num_channels = len(cnn_channel_slices)
        other_dim = sum(stop - start for start, stop in other_slices)

        self._net = _CNNNetwork(
            cnn_input_len=cnn_input_len,
            num_channels=num_channels,
            other_dim=other_dim,
            cnn_filters=cnn_filters,
            cnn_kernel=cnn_kernel,
            fc_hidden=fc_hidden,
        )
        self._optimizer = torch.optim.Adam(self._net.parameters(), lr=alpha)
        self._loss_fn = nn.MSELoss()

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _build_tensors(
        self, features: np.ndarray
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Split the feature vector into the CNN block and the other block."""
        # CNN block: stack channels → (num_channels, cnn_input_len) → (1, C, L)
        cnn_block = torch.tensor(
            np.stack([features[s:e] for s, e in self._cnn_channel_slices], axis=0),
            dtype=torch.float32,
        ).unsqueeze(0)
        # Other block: (D,) → (1, D)
        other_block = torch.tensor(
            np.concatenate([features[s:e] for s, e in self._other_slices]),
            dtype=torch.float32,
        ).unsqueeze(0)
        return cnn_block, other_block

    # ------------------------------------------------------------------
    # ValueApproximator interface
    # ------------------------------------------------------------------

    def predict(self, s: State, a: Action) -> float:
        features = self._feature_extractor(s, a)
        cnn_block, other_block = self._build_tensors(features)
        with torch.no_grad():
            q = self._net(cnn_block, other_block)
        return float(q.item())

    def update(self, s: State, a: Action, target: float) -> None:
        features = self._feature_extractor(s, a)
        cnn_block, other_block = self._build_tensors(features)
        target_t = torch.tensor([[target]], dtype=torch.float32)

        self._optimizer.zero_grad()
        q = self._net(cnn_block, other_block)
        loss = self._loss_fn(q, target_t)
        loss.backward()
        self._optimizer.step()

    def get_state(self) -> object:
        """Return the network state dict (PyTorch tensors)."""
        return self._net.state_dict()

    def load_state(self, state: object) -> None:
        """Restore network weights from a previously saved state dict."""
        self._net.load_state_dict(state)  # type: ignore[arg-type]


class CNNApproxAgent(QAgent[State, Action]):
    """An agent that uses a CNN-based value function approximator."""

    _approximator: CNNValueApproximator[State, Action]
    _epsilon: float

    def __init__(
        self,
        value_approximator: CNNValueApproximator[State, Action],
        epsilon: float = 0.05,
    ) -> None:
        self._approximator = value_approximator
        self._epsilon = epsilon

    def q_value(self, s: State, a: Action) -> float:
        return self._approximator.predict(s, a)

    def update_q_value(self, s: State, a: Action, value: float) -> None:
        self._approximator.update(s, a, value)

    def get_epsilon(self, s: State) -> float:
        return self._epsilon

    def get_state(self) -> object:
        return self._approximator.get_state()

    def checkpoint(self) -> None:
        path = f"{self.PICKLE_PATH}/{self.name}_cnn_checkpoint.pt"
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save(self.get_state(), path)

    def restore(self) -> None:
        path = f"{self.PICKLE_PATH}/{self.name}_cnn_checkpoint.pt"
        state = torch.load(path, weights_only=True)
        self._approximator.load_state(state)
