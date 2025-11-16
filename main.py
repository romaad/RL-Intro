from dataclasses import dataclass
from decimal import Decimal
from typing import Sequence
from base import Agent, Runner
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
from plot import turn_plot_off
from utils import drange
import argparse


@dataclass
class _Args:
    episodes: int
    record_cnt: int
    show_plot: bool


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


def create_parser():
    parser = argparse.ArgumentParser(
        description="Run RL agents on the Easy21 environment."
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
    )
    run_easy21(args_dataclass)


if __name__ == "__main__":
    main()
