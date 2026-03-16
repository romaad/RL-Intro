from dataclasses import dataclass
from decimal import Decimal
from typing import Sequence
from base import Agent, Runner, MultiAgentRunner
from easy21.easy21 import (
    Easy21Env,
    Easy21State,
    Easy21Action,
)
from easy21.easy21_agents import (
    MCEasy21Agent,
    SarsaEasy21Agent,
    SarsaLambdaEasy21Agent,
    SarsaLambdaEasy21LinearApproxAgent,
)
from envs.tarneeb.env import TarneebEnv, PartialTarneebState, TarneebAction, TarneebSate
from envs.tarneeb.agents import RandomTarneebAgent, HumanTarneebAgent
from plot import turn_plot_off
from utils import drange
import argparse


@dataclass
class _Args:
    episodes: int
    record_cnt: int
    show_plot: bool
    mode: str
    human_players: int


def run_easy21(args: _Args) -> None:
    agents: Sequence[Agent[Easy21State, Easy21Action]] = (
        [
            # NaiveAgent(),
            MCEasy21Agent(),
            SarsaEasy21Agent(),
        ]
        + [
            SarsaLambdaEasy21Agent(lambbda=lamb, gamma=1.0)
            for lamb in list(drange(Decimal(0.0), Decimal(1.1), Decimal(0.4)))
        ]
        + [
            SarsaLambdaEasy21LinearApproxAgent(lambbda=lamb, gamma=1.0)
            for lamb in list(drange(Decimal(0.0), Decimal(1.1), Decimal(0.4)))
        ]
    )
    # n_episodes = 10_000  # 10k episodes
    # n_episodes = 100_000  # 100k episodes
    # n_episodes = 1000_000  # 1 million episodes

    runner = Runner[Easy21State, Easy21Action]()
    if not args.show_plot:
        turn_plot_off()
    for agent in agents:
        env = Easy21Env()
        print(f"Running agent: {agent.name}")
        r = runner.run_episodes(env, agent, args.episodes, record_cnt=args.record_cnt)
        print(f"Avg reward {agent.name} over {args.episodes} episodes: {r}")
        print("Storing agent data")
        agent.checkpoint()


def run_tarneeb(args: _Args) -> None:
    agents: list[Agent[PartialTarneebState, TarneebAction]] = []
    for _ in range(args.human_players):
        agents.append(HumanTarneebAgent())
    for _ in range(4 - args.human_players):
        agent = RandomTarneebAgent()
        agent.env_name = "tarneeb"
        if args.mode == "play":
            try:
                agent.restore()
                print(f"Loaded saved state for {agent.name}")
            except FileNotFoundError:
                print(f"No saved state found for {agent.name}, using fresh agent")
        agents.append(agent)
    for agent in agents:
        agent.env_name = "tarneeb"
    runner = MultiAgentRunner[TarneebSate, PartialTarneebState, TarneebAction]()
    if not args.show_plot:
        turn_plot_off()
    env = TarneebEnv()
    avg_rewards = runner.run_episodes(
        env, agents, args.episodes, record_cnt=args.record_cnt
    )
    print(
        f"Avg rewards for Tarneeb agents over {args.episodes} episodes: {avg_rewards}"
    )
    if args.mode == "train":
        for agent in agents:
            if isinstance(agent, RandomTarneebAgent):
                agent.checkpoint()


def create_parser():
    parser = argparse.ArgumentParser(description="Run RL agents on environments.")
    parser.add_argument(
        "--game",
        type=str,
        default="easy21",
        choices=["easy21", "tarneeb"],
        help="The game/environment to run agents on.",
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="train",
        choices=["train", "play"],
        help="Mode: train to train agents, play to load saved agents and run.",
    )
    parser.add_argument(
        "--human-players",
        type=int,
        default=0,
        help="Number of human players (for play mode).",
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=10_000,
        help="Number of episodes to run for each agent.",
    )
    parser.add_argument(
        "--record-cnt",
        type=int,
        default=0,
        help="Number of episodes to show the game history for.",
    )
    parser.add_argument(
        "--no-plot",
        action="store_false",
        dest="show_plot",
        help="Disable plotting the value function after training.",
    )

    return parser


def main() -> None:
    parser = create_parser()
    args = parser.parse_args()
    args_dataclass = _Args(
        episodes=args.episodes,
        record_cnt=args.record_cnt,
        show_plot=args.show_plot,
        mode=args.mode,
        human_players=args.human_players,
    )
    if args.game == "easy21":
        run_easy21(args_dataclass)
    elif args.game == "tarneeb":
        run_tarneeb(args_dataclass)


if __name__ == "__main__":
    main()
