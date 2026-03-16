from dataclasses import dataclass
from typing import Generic
from agents.q_agent import QAgent
from base import Action, State, Step

PICKLES_FOLDER = "pickles"


@dataclass(frozen=True)
class _MC_STATE(Generic[State, Action]):
    returns: dict[tuple[State, Action], list[float]]
    q: dict[tuple[State, Action], float]
    s_cnt: dict[State, int]


class MonteCarloAgent(QAgent[State, Action]):
    """An agent that uses Monte Carlo methods to learn the value function."""

    AGENT_STATE_T = _MC_STATE[State, Action]
    _state: _MC_STATE[State, Action]
    EPS_CUSHION: int = 100

    def __init__(self) -> None:
        self._state = _MC_STATE(
            returns={},
            q={},
            s_cnt={},
        )

    def visit(self, s: State) -> None:
        cnt = self._state.s_cnt.get(s, 0)
        self._state.s_cnt[s] = cnt + 1

    def get_epsilon(self, s: State) -> float:
        return self.EPS_CUSHION / (self.EPS_CUSHION + self._state.s_cnt.get(s, 0))

    def q_value(self, s: State, a: Action) -> float:
        return self._state.q.get((s, a), 0.0)

    def update_q_value(self, s: State, a: Action, value: float) -> None:
        self._state.q[(s, a)] = value

    def get_variable_learning_rate(self, s: State, a: Action | None) -> float:
        if a is None:
            return 1.0 / len(self._state.returns.get((s, a), [0]))
        return 1.0 / (
            sum(
                len(self._state.returns.get((s, aa), [0])) for aa in self.action_space()
            )
        )

    def get_states(self) -> list[State]:
        return list(self._state.s_cnt.keys())

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
