import random
from base import Action, Agent, State, Step


class MonteCarloAgent(Agent[State, Action]):
    """An agent that uses Monte Carlo methods to learn the value function."""

    EPS_CUSHION = 100

    def __init__(self) -> None:
        self._returns: dict[tuple[State, Action], list[float]] = {}
        self._Q: dict[tuple[State, Action], float] = {}
        self._cnt_state: dict[State, int] = {}

    def action_space(self) -> list[Action]:
        raise NotImplementedError

    def act(self, s: State) -> Action:
        # Epsilon-greedy policy
        cnt = self._cnt_state.get(s, 0)
        self._cnt_state[s] = cnt + 1
        epsilon = self.EPS_CUSHION / (self.EPS_CUSHION + cnt)
        if random.uniform(0, 1) < epsilon:
            return random.choice(self.action_space())
        else:
            # MC ARGMAX action selection
            return max(
                self.action_space(),
                key=lambda a: self._Q.get((s, a), 0.0),
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
                if (s, a) not in self._returns:
                    self._returns[(s, a)] = []
                self._returns[(s, a)].append(curr_reward)
                self._Q[(s, a)] = sum(self._returns[(s, a)]) / len(
                    self._returns[(s, a)]
                )
