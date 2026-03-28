from typing import Callable, Generic
from agents.q_agent import QAgent
from base import Action, State
import numpy as np

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


class CNNValueApproximator(ValueApproximator[State, Action]):
    """CNN + 3-hidden-layer value function approximator implemented in NumPy.

    The feature vector produced by *feature_extractor* is split into two parts:

    1. **CNN block** – assembled from *cnn_channel_slices*.  Each element is a
       ``(start, stop)`` index pair selecting a slice of the feature vector.
       All slices must have the same length (*cnn_input_len*) and are stacked
       column-wise to form a matrix of shape ``(cnn_input_len, num_channels)``.

    2. **Other block** – features collected from *other_slices* and
       concatenated into a 1-D vector.

    Architecture::

        CNN block  (cnn_input_len, C)
            ↓  Conv1D(num_filters, kernel_size)  →  (cnn_input_len-kernel_size+1, F)
            ↓  ReLU
            ↓  Flatten                           →  cnn_out_dim
            │
        Other block  (D,)
            │
            ├── concatenate ──────────────────── →  cnn_out_dim + D
            ↓
           FC1 (hidden1) + ReLU
            ↓
           FC2 (hidden2) + ReLU
            ↓
           FC3 (hidden3) + ReLU
            ↓
           Output (1)  →  scalar Q-value
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
        self._feature_extractor = feature_extractor
        self._alpha = alpha
        self._cnn_input_len = cnn_input_len
        self._cnn_channel_slices = cnn_channel_slices
        self._other_slices = other_slices
        self._cnn_filters = cnn_filters
        self._cnn_kernel = cnn_kernel

        num_channels = len(cnn_channel_slices)
        cnn_out_len = cnn_input_len - cnn_kernel + 1
        cnn_out_dim = cnn_out_len * cnn_filters
        other_dim = sum(stop - start for start, stop in other_slices)
        combined_dim = cnn_out_dim + other_dim
        h1, h2, h3 = fc_hidden

        self._cnn_out_len = cnn_out_len
        self._cnn_out_dim = cnn_out_dim

        rng = np.random.default_rng(seed)

        # CNN weights: (num_filters, kernel_size, in_channels)
        self._conv_w: np.ndarray = rng.standard_normal(
            (cnn_filters, cnn_kernel, num_channels)
        ) * np.sqrt(2.0 / (cnn_kernel * num_channels))
        self._conv_b: np.ndarray = np.zeros(cnn_filters)

        # FC layer weights
        self._W1: np.ndarray = rng.standard_normal((h1, combined_dim)) * np.sqrt(
            2.0 / combined_dim
        )
        self._b1: np.ndarray = np.zeros(h1)
        self._W2: np.ndarray = rng.standard_normal((h2, h1)) * np.sqrt(2.0 / h1)
        self._b2: np.ndarray = np.zeros(h2)
        self._W3: np.ndarray = rng.standard_normal((h3, h2)) * np.sqrt(2.0 / h2)
        self._b3: np.ndarray = np.zeros(h3)
        self._W_out: np.ndarray = rng.standard_normal((1, h3)) * np.sqrt(2.0 / h3)
        self._b_out: np.ndarray = np.zeros(1)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _relu(x: np.ndarray) -> np.ndarray:
        return np.maximum(0.0, x)

    @staticmethod
    def _relu_grad(x: np.ndarray) -> np.ndarray:
        return (x > 0.0).astype(float)

    def _build_inputs(
        self, features: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """Split the feature vector into the CNN block and the other block."""
        cnn_block = np.stack(
            [features[s:e] for s, e in self._cnn_channel_slices], axis=1
        )  # (cnn_input_len, num_channels)
        other_block = np.concatenate(
            [features[s:e] for s, e in self._other_slices]
        )
        return cnn_block, other_block

    def _conv1d_forward(self, x: np.ndarray) -> np.ndarray:
        """Vectorised 1-D convolution: (L, C) → (L-K+1, F)."""
        # sliding_window_view returns (out_len, 1, K, C) – squeeze the extra dim
        windows = np.lib.stride_tricks.sliding_window_view(
            x, (self._cnn_kernel, x.shape[1])
        )[:, 0, :, :]  # (out_len, K, C)
        return (
            np.einsum("okc,fkc->of", windows, self._conv_w) + self._conv_b
        )  # (out_len, F)

    def _conv1d_backward(
        self, x: np.ndarray, d_out: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Backward pass for Conv1D.

        Returns:
            d_x       – gradient w.r.t. input  (L, C)
            d_conv_w  – gradient w.r.t. weights (F, K, C)
            d_conv_b  – gradient w.r.t. bias    (F,)
        """
        windows = np.lib.stride_tricks.sliding_window_view(
            x, (self._cnn_kernel, x.shape[1])
        )[:, 0, :, :]  # (out_len, K, C)
        d_conv_w = np.einsum("of,okc->fkc", d_out, windows)
        d_conv_b = d_out.sum(axis=0)
        d_x = np.zeros_like(x)
        for k in range(self._cnn_kernel):
            # d_x[k : k+out_len, :] += d_out @ conv_w[:, k, :]
            d_x[k : k + d_out.shape[0], :] += d_out @ self._conv_w[:, k, :]
        return d_x, d_conv_w, d_conv_b

    # ------------------------------------------------------------------
    # Forward / backward
    # ------------------------------------------------------------------

    def _forward(self, features: np.ndarray) -> tuple[float, tuple]:
        """Full forward pass.  Returns (prediction, cache)."""
        cnn_block, other_block = self._build_inputs(features)

        # CNN
        conv_out = self._conv1d_forward(cnn_block)  # (out_len, F)
        conv_relu = self._relu(conv_out)
        conv_flat = conv_relu.flatten()  # (cnn_out_dim,)

        # Combine
        combined = np.concatenate([conv_flat, other_block])

        # FC layers
        z1 = self._W1 @ combined + self._b1
        h1 = self._relu(z1)
        z2 = self._W2 @ h1 + self._b2
        h2 = self._relu(z2)
        z3 = self._W3 @ h2 + self._b3
        h3 = self._relu(z3)

        out = float((self._W_out @ h3 + self._b_out)[0])

        cache = (cnn_block, conv_out, conv_flat, combined, z1, h1, z2, h2, z3, h3)
        return out, cache

    def _backward(self, target: float, prediction: float, cache: tuple) -> None:
        """Backpropagation and in-place gradient descent update."""
        cnn_block, conv_out, conv_flat, combined, z1, h1, z2, h2, z3, h3 = cache

        error = target - prediction
        # dL/d_out where L = 0.5*(target-out)^2  →  dL/d_out = -(target-out) = -error
        d_out_val = -error  # scalar

        # Output layer
        d_W_out = d_out_val * h3[np.newaxis, :]
        d_b_out = np.array([d_out_val])
        d_h3 = self._W_out.T @ np.array([d_out_val])  # (h3,)

        # FC3
        d_z3 = d_h3 * self._relu_grad(z3)
        d_W3 = d_z3[:, np.newaxis] @ h2[np.newaxis, :]
        d_b3 = d_z3
        d_h2 = self._W3.T @ d_z3

        # FC2
        d_z2 = d_h2 * self._relu_grad(z2)
        d_W2 = d_z2[:, np.newaxis] @ h1[np.newaxis, :]
        d_b2 = d_z2
        d_h1 = self._W2.T @ d_z2

        # FC1
        d_z1 = d_h1 * self._relu_grad(z1)
        d_W1 = d_z1[:, np.newaxis] @ combined[np.newaxis, :]
        d_b1 = d_z1
        d_combined = self._W1.T @ d_z1

        # CNN ReLU + conv backward
        d_conv_flat = d_combined[: self._cnn_out_dim]
        d_conv_relu = d_conv_flat.reshape(self._cnn_out_len, self._cnn_filters)
        d_conv_out = d_conv_relu * self._relu_grad(conv_out)
        _, d_conv_w, d_conv_b = self._conv1d_backward(cnn_block, d_conv_out)

        # Gradient descent updates
        self._W_out -= self._alpha * d_W_out
        self._b_out -= self._alpha * d_b_out
        self._W3 -= self._alpha * d_W3
        self._b3 -= self._alpha * d_b3
        self._W2 -= self._alpha * d_W2
        self._b2 -= self._alpha * d_b2
        self._W1 -= self._alpha * d_W1
        self._b1 -= self._alpha * d_b1
        self._conv_w -= self._alpha * d_conv_w
        self._conv_b -= self._alpha * d_conv_b

    # ------------------------------------------------------------------
    # ValueApproximator interface
    # ------------------------------------------------------------------

    def predict(self, s: State, a: Action) -> float:
        features = self._feature_extractor(s, a)
        out, _ = self._forward(features)
        return out

    def update(self, s: State, a: Action, target: float) -> None:
        features = self._feature_extractor(s, a)
        prediction, cache = self._forward(features)
        self._backward(target, prediction, cache)

    def get_state(self) -> object:
        return {
            "conv_w": self._conv_w.copy(),
            "conv_b": self._conv_b.copy(),
            "W1": self._W1.copy(),
            "b1": self._b1.copy(),
            "W2": self._W2.copy(),
            "b2": self._b2.copy(),
            "W3": self._W3.copy(),
            "b3": self._b3.copy(),
            "W_out": self._W_out.copy(),
            "b_out": self._b_out.copy(),
        }


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
        save_pickle(
            self.get_state(), f"{self.PICKLE_PATH}/{self.name}_cnn_checkpoint.pkl"
        )
