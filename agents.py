from dataclasses import dataclass
import random
from typing import Generic
from base import Action, Agent, State, Step
from plot import plot_value_function

PICKLES_FOLDER = "pickles"


@dataclass(frozen=True)
class _MC_STATE(Generic[State, Action]):
    returns: dict[tuple[State, Action], list[float]]
    q: dict[tuple[State, Action], float]
    s_cnt: dict[State, int]


class MonteCarloAgent(Agent[State, Action]):
    """An agent that uses Monte Carlo methods to learn the value function."""

    AGENT_STATE_T = _MC_STATE[State, Action]
    _state: _MC_STATE[State, Action]
    EPS_CUSHION = 100

    def __init__(self) -> None:
        self._state = _MC_STATE(
            returns={},
            q={},
            s_cnt={},
        )

    def action_space(self) -> list[Action]:
        raise NotImplementedError

    def act(self, s: State) -> Action:
        # Epsilon-greedy policy

        cnt = self._state.s_cnt.get(s, 0)
        self._state.s_cnt[s] = cnt + 1
        epsilon = self.EPS_CUSHION / (self.EPS_CUSHION + cnt)
        if random.uniform(0, 1) < epsilon:
            return random.choice(self.action_space())
        else:
            # MC ARGMAX action selection
            return max(
                self.action_space(),
                key=lambda a: self._state.q.get((s, a), 0.0),
            )

    def update(self, steps: list[Step[State, Action]]) -> None:
        curr_reward = 0.0
        visited: set[tuple[State, Action]] = set()
        for i in range(len(steps) - 1, -1, -1):
            a = steps[i].action
            r = steps[i].outcome.reward
            s = steps[i - 1].outcome.next_state
            if a is None:
                # this is the initial step with no action
                continue
            curr_reward += r
            if (s, a) not in visited:
                visited.add((s, a))
                if (s, a) not in self._state.returns:
                    self._state.returns[(s, a)] = []
                self._state.returns[(s, a)].append(curr_reward)
                self._state.q[(s, a)] = sum(self._state.returns[(s, a)]) / len(
                    self._state.returns[(s, a)]
                )

    def state_to_xy(self, s: State) -> tuple[int, int]:
        raise NotImplementedError

    def get_xy_labels(self) -> tuple[str, str]:
        raise NotImplementedError

    def on_train_end(self) -> None:
        v_star = [
            (
                *self.state_to_xy(s),
                max(self._state.q.get((s, a), 0.0) for a in self.action_space()),
            )
            for s in self._state.s_cnt.keys()
        ]
        labelx, labely = self.get_xy_labels()
        plot_value_function(
            v_star,
            title="State-Value Function V* after Training",
            xlabel=labelx,
            ylabel=labely,
        )
