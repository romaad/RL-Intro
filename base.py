from dataclasses import dataclass
import random
from typing import Generic, TypeVar

from pickle_utils import load_pickle, save_pickle


State = TypeVar("State")
Action = TypeVar("Action")
PartialState = TypeVar("PartialState")


@dataclass
class Outcome(Generic[State]):
    """Represents the outcome of taking an action in a given state."""

    next_state: State
    reward: float
    done: bool
    # only for multi-agent envs

    def __str__(self) -> str:
        return f"O(NS:{self.next_state},R:{self.reward},D?:{self.done})"


@dataclass
class MultiAgentOutcome(Generic[State]):
    """
    Represents the outcome of taking an action
    in a multi-agent environment.
    1. Agents perform actions in turns in a round.
    2. After each action, the environment returns the next agent to act,
       the next state, the reward for the acting agent, and whether the episode is done.
    3. The rewards are given after all agents have taken their actions in a round.
    """

    next_agent_idx: int
    next_state: State
    reward_per_agent: list[float]
    done: bool


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
    PICKLE_PATH: str = "agents_pickes"

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
        save_pickle(
            self._state,
            f"{self.PICKLE_PATH}/{self.name}_checkpoint.pkl",
        )

    def restore(self) -> None:
        """Optional method to restore the agent's state."""

        self._state = load_pickle(
            self.PICKLE_PATH + "/" + self.name + "_checkpoint.pkl",
            self.AGENT_STATE_T,
        )

    def on_train_end(self) -> None:
        """
        Optional method called at the end of training.
        Can be used for any final processing or cleanup or plots.
        """
        pass

    def get_variable_learning_rate(self, s: State, a: Action | None) -> float:
        """Optional method to get the learning rate for a given state and action."""
        return 0.1

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


class MultipleAgentEnv(Env[State, list[Action]], Generic[State, PartialState, Action]):
    """Base class for an environment with multiple agents."""

    def agent_step(
        self, s: State, action: Action, agent_idx: int
    ) -> MultiAgentOutcome[State]:
        raise NotImplementedError

    def init_state(self) -> State:
        raise NotImplementedError

    def to_partial_state(self, s: State, agent_idx: int) -> PartialState:
        raise NotImplementedError

    def step_impl(self, s: State, action: list[Action]) -> Outcome[State]:
        raise NotSupportedError("Use agent_step for multi-agent environments.")


class MultiAgentRunner(Generic[State, PartialState, Action]):
    """Base class for running multiple agents in a multi-agent environment."""

    def run_episode(
        self,
        env: MultipleAgentEnv[State, PartialState, Action],
        agents: list[Agent[PartialState, Action]],
        print_game: bool = False,
    ) -> list[float]:
        env.reset()
        state = env.init_state()
        done = False
        agent_idx = 0  # start with agent 0
        total_rewards = [0.0] * len(agents)
        steps_per_agent: list[list[Step[PartialState, Action]]] = [[] for _ in agents]
        while not done:
            partial_state = env.to_partial_state(state, agent_idx)
            action = agents[agent_idx].act(partial_state)
            outcome = env.agent_step(state, action, agent_idx)
            reward = outcome.reward_per_agent[agent_idx]
            total_rewards[agent_idx] += reward
            state = outcome.next_state
            done = outcome.done
            agent_idx = outcome.next_agent_idx
            # For simplicity, update each agent with their own steps
            prev_partial_state = partial_state
            steps_per_agent[agent_idx].append(
                Step(action=action, outcome=Outcome(state, reward, done))
            )
            agents[agent_idx].update_step(
                prev_partial_state, action, reward, partial_state, action
            )
        if print_game:
            for i, steps in enumerate(steps_per_agent):
                print(f"Agent {i} history:", ",\n".join([str(step) for step in steps]))
        for agent, steps in zip(agents, steps_per_agent):
            agent.update(steps)
        return total_rewards

    def run_episodes(
        self,
        env: MultipleAgentEnv[State, PartialState, Action],
        agents: list[Agent[PartialState, Action]],
        num_episodes: int,
        record_cnt: int = 0,
        checkpoint_every: int = 0,
    ) -> list[float]:
        total_rewards = [0.0] * len(agents)
        for epi in range(num_episodes):
            print_game = (
                record_cnt > 0 and random.randint(1, num_episodes) <= record_cnt
            )
            episode_rewards = self.run_episode(env, agents, print_game)
            for i in range(len(agents)):
                total_rewards[i] += episode_rewards[i]
            if epi % 10000 == 0 and epi > 0:
                avg_rewards = [total / (epi + 1) for total in total_rewards]
                print(f"Completed {epi} episodes, avg rewards so far: {avg_rewards}")
            if checkpoint_every > 0 and epi % checkpoint_every == 0 and epi > 0:
                for agent in agents:
                    agent.checkpoint()
                print(f"Checkpoint saved at episode {epi}")
        for agent in agents:
            agent.on_train_end()
        return [total / num_episodes for total in total_rewards]


class NotSupportedError(Exception):
    """Exception raised for unsupported operations in the environment."""

    pass
