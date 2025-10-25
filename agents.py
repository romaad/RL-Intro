import random
from base import Action, Agent, State, Step


class MonteCarloAgent(Agent[State, Action]):
    """An agent that uses Monte Carlo methods to learn the value function."""

    def __init__(self) -> None:
        self.returns: dict[tuple[State, Action], list[float]] = {}
        self.Q: dict[tuple[State, Action], float] = {}

    def action_space(self) -> list[Action]:
        raise NotImplementedError

    def act(self, s: State) -> Action:
        # Epsilon-greedy policy
        epsilon = 0.1
        if random.uniform(0, 1) < epsilon:
            return random.choice(self.action_space())
        else:
            # MC ARGMAX action selection
            return max(
                self.action_space(),
                key=lambda a: self.Q.get((s, a), 0.0),
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
                if (s, a) not in self.returns:
                    self.returns[(s, a)] = []
                self.returns[(s, a)].append(curr_reward)
                self.Q[(s, a)] = sum(self.returns[(s, a)]) / len(self.returns[(s, a)])
