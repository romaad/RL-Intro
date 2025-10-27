from dataclasses import dataclass
import random
from typing import Generic, TypeVar

from pickle_utils import load_pickle, save_pickle


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

    AGENT_STATE_T: type
    _state: object

    def act(self, s: State) -> Action:
        raise NotImplementedError

    # Optional method to update the agent's knowledge after an episode
    def update(self, steps: list[Step[State, Action]]) -> None:
        pass

    @property
    def name(self) -> str:
        return self.__class__.__name__

    def checkpoint(self) -> None:
        """Optional method to checkpoint the agent's state."""
        save_pickle(self._state, f"{self.name}_checkpoint.pkl")

    def restore(self) -> None:
        """Optional method to restore the agent's state."""

        self._state = load_pickle(self.name + "_checkpoint.pkl", self.AGENT_STATE_T)

    def on_train_end(self) -> None:
        """
        Optional method called at the end of training.
        Can be used for any final processing or cleanup or plots.
        """
        pass

    def update_step(
        self,
        s: State,
        a: Action,
        r: float,
        s_next: State,
        a_next: Action,
    ) -> None:
        """
        Performs the agent's update for a single step.
        Optional as not all agents need per-step updates.
        """
        pass


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
        state = env.init_state()
        done = False
        steps: list[Step[State, Action]] = [
            Step(action=None, outcome=Outcome(state, 0.0, False))
        ]
        r = 0.0
        action = agent.act(state)
        while not done:
            prev_state = state
            outcome = env.step(state, action)
            prev_action = action
            state = outcome.next_state
            r += outcome.reward
            done = outcome.done
            action = agent.act(state)
            steps.append(Step(action=action, outcome=outcome))
            agent.update_step(prev_state, prev_action, outcome.reward, state, action)
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
        agent.on_train_end()
        return total_reward / num_episodes
