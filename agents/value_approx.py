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
