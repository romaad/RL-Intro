from dataclasses import dataclass
import random
from typing import Generic, TypeVar


State = TypeVar("State")
Action = TypeVar("Action")


@dataclass
class Outcome(Generic[State]):
    """Represents the outcome of taking an action in a given state."""

    next_state: State
    reward: float
    done: bool

    def __str__(self) -> str:
        return f"O(NS:{self.next_state},R:{self.reward},D?:{self.done})"


@dataclass
class Step(Generic[State, Action]):
    action: Action | None
    outcome: Outcome[State]

    def __str__(self) -> str:
        return f"S(A:{self.action},O:{self.outcome})"


class Env(Generic[State, Action]):
    """Base class for an environment."""

    def reset(self) -> None:
        pass

    def step(self, s: State, action: Action) -> Outcome[State]:
        return self.step_impl(s, action)

    def step_impl(self, s: State, action: Action) -> Outcome[State]:
        raise NotImplementedError

    def init_state(self) -> State:
        raise NotImplementedError


class Agent(Generic[State, Action]):
    """Base class for an agent."""

    def act(self, s: State) -> Action:
        raise NotImplementedError

    # Optional method to update the agent's knowledge after an episode
    def update(self, steps: list[Step[State, Action]]) -> None:
        pass

    @property
    def name(self) -> str:
        return self.__class__.__name__


class Runner(Generic[State, Action]):
    """Base class for running
    an agent in an environment.
    """

    def run_episode(
        self,
        env: Env[State, Action],
        agent: Agent[State, Action],
        print_game: bool = False,
    ) -> float:
        env.reset()
        s = env.init_state()
        done = False
        steps: list[Step[State, Action]] = [
            Step(action=None, outcome=Outcome(s, 0.0, False))
        ]
        r = 0.0
        while not done:
            action = agent.act(s)
            outcome = env.step(s, action)
            s = outcome.next_state
            r += outcome.reward
            done = outcome.done
            steps.append(Step(action=action, outcome=outcome))
        if print_game:
            print("Game history:", ",\n".join([str(step) for step in steps]))
        agent.update(steps)
        return r

    def run_episodes(
        self,
        env: Env[State, Action],
        agent: Agent[State, Action],
        num_episodes: int,
        record_cnt: int = 0,
    ) -> float:
        total_reward = 0.0
        for epi in range(num_episodes):
            print_game = (
                record_cnt > 0 and random.randint(1, num_episodes) <= record_cnt
            )
            total_reward += self.run_episode(env, agent, print_game)
            if epi % 10000 == 0 and epi > 0:
                print(
                    f"Completed {epi} episodes, avg reward so far: {total_reward/epi}"
                )
        return total_reward / num_episodes
