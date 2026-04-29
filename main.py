from dataclasses import dataclass
from decimal import Decimal
from typing import Sequence
from base import Agent, Runner, MultiAgentRunner, AlphaZeroMultiAgentRunner
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
from envs.tarneeb.env import (
    TarneebEnv,
    PartialTarneebState,
    TarneebAction,
    TarneebState,
)
from envs.tarneeb.agents import (
    HumanTarneebAgent,
    MCTarneebAgent,
    SarsaTarneebAgent,
    SarsaLambdaTarneebAgent,
    SarsaLambdaTarneebLinearApproxAgent,
    SarsaLambdaTarneebCNNApproxAgent,
    SarsaLambdaTarneebSharedCNNApproxAgent,
    SarsaLambdaTarneebSeparateHeadsCNNApproxAgent,
    TarneebCNNValueApprox,
    TarneebSeparateHeadsCNNValueApprox,
)
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
    verbose: bool
    agent: str
    checkpoint_every: int


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


def _build_alphazero_agents(
    args: _Args, env: TarneebEnv
) -> tuple[list, object]:
    """Instantiate shared SENet, 4 AlphaZero agents, and a trainer.

    Returns ``(agents, trainer)``; *trainer* is ``None`` in play mode.
    """
    from agents.alphazero.neural_network import NeuralNetwork
    from agents.alphazero.tarneeb_mcts import TarneebMCTS
    from agents.alphazero.alphazero_trainer import AlphaZeroTrainer
    from envs.tarneeb.alphazero_model import TarneebSENet
    from envs.tarneeb.alphazero_agent import AlphaZeroTarneebAgent

    is_train = args.mode == "train"
    temperature = 1.0 if is_train else 0.1
    num_simulations = 50 if is_train else 200

    model = TarneebSENet()
    shared_nn = NeuralNetwork(model, lr=0.001)
    replay_buffer: list = []

    mcts_instances = [
        TarneebMCTS(env, shared_nn, num_simulations=num_simulations)
        for _ in range(4)
    ]
    az_agents = [
        AlphaZeroTarneebAgent(
            agent_idx=i,
            nn=shared_nn,
            mcts=mcts_instances[i],
            replay_buffer=replay_buffer,
            temperature=temperature,
        )
        for i in range(4)
    ]

    checkpoint_path = "agents_pickes/alphazero_tarneeb_latest.pt"
    if not is_train:
        try:
            shared_nn.load(checkpoint_path)
            print(f"Loaded AlphaZero checkpoint from {checkpoint_path}")
        except FileNotFoundError:
            print("No AlphaZero checkpoint found; using untrained network.")

    trainer = None
    if is_train:
        trainer = AlphaZeroTrainer(
            env=env,
            nn=shared_nn,
            agents=az_agents,
            replay_buffer=replay_buffer,
            checkpoint_path="agents_pickes",
        )

    return az_agents, trainer


def run_tarneeb(args: _Args) -> None:
    env = TarneebEnv()

    # AlphaZero has its own training loop
    if args.agent == "alphazero":
        if not args.show_plot:
            turn_plot_off()
        az_agents, trainer = _build_alphazero_agents(args, env)
        if args.mode == "train" and trainer is not None:
            trainer.run(
                num_iterations=args.episodes,
                episodes_per_iter=10,
                train_steps_per_iter=20,
                checkpoint_every=max(1, args.checkpoint_every) if args.checkpoint_every > 0 else 10,
            )
        else:
            az_runner = AlphaZeroMultiAgentRunner()
            avg_rewards = az_runner.run_episodes(
                env, az_agents, args.episodes, record_cnt=args.record_cnt
            )
            print(
                f"Avg rewards for AlphaZero agents over {args.episodes} episodes: {avg_rewards}"
            )
        return

    if args.human_players > 0:
        # Humans + AI agents
        agents: list[Agent[PartialTarneebState, TarneebAction]] = [
            HumanTarneebAgent(verbose=args.verbose) for _ in range(args.human_players)
        ]
        # AI agents
        ai_count = 4 - args.human_players
        if args.agent == "mc":
            ai_agents = [MCTarneebAgent() for _ in range(ai_count)]
        elif args.agent == "sarsa":
            ai_agents = [SarsaTarneebAgent() for _ in range(ai_count)]
        elif args.agent == "sarsa-lambda":
            ai_agents = [
                SarsaLambdaTarneebAgent(lambbda=0.5, gamma=1.0) for _ in range(ai_count)
            ]
        elif args.agent == "value-approx":
            ai_agents = [
                SarsaLambdaTarneebLinearApproxAgent(lambbda=0.5, gamma=1.0)
                for _ in range(ai_count)
            ]
        elif args.agent == "cnn-approx":
            ai_agents = [
                SarsaLambdaTarneebCNNApproxAgent(lambbda=0.5, gamma=1.0)
                for _ in range(ai_count)
            ]
        elif args.agent == "shared-cnn":
            shared_approx = TarneebCNNValueApprox()
            ai_agents = [
                SarsaLambdaTarneebSharedCNNApproxAgent(0.5, 1.0, shared_approx)
                for _ in range(ai_count)
            ]
        elif args.agent == "separate-heads-cnn":
            shared_approx = TarneebSeparateHeadsCNNValueApprox()
            ai_agents = [
                SarsaLambdaTarneebSeparateHeadsCNNApproxAgent(0.5, 1.0, shared_approx)
                for _ in range(ai_count)
            ]
        else:
            raise ValueError(f"Unknown agent: {args.agent}")
        agents.extend(ai_agents)
    else:
        # RL agents - use selected agent for all players
        if args.agent == "mc":
            agents = [MCTarneebAgent() for _ in range(4)]
        elif args.agent == "sarsa":
            agents = [SarsaTarneebAgent() for _ in range(4)]
        elif args.agent == "sarsa-lambda":
            agents = [
                SarsaLambdaTarneebAgent(lambbda=0.5, gamma=1.0) for _ in range(4)
            ]  # default lambda
        elif args.agent == "value-approx":
            agents = [
                SarsaLambdaTarneebLinearApproxAgent(lambbda=0.5, gamma=1.0)
                for _ in range(4)
            ]
        elif args.agent == "cnn-approx":
            agents = [
                SarsaLambdaTarneebCNNApproxAgent(lambbda=0.5, gamma=1.0)
                for _ in range(4)
            ]
        elif args.agent == "shared-cnn":
            shared_approx = TarneebCNNValueApprox()
            agents = [
                SarsaLambdaTarneebSharedCNNApproxAgent(0.5, 1.0, shared_approx)
                for _ in range(4)
            ]
        elif args.agent == "separate-heads-cnn":
            shared_approx = TarneebSeparateHeadsCNNValueApprox()
            agents = [
                SarsaLambdaTarneebSeparateHeadsCNNApproxAgent(0.5, 1.0, shared_approx)
                for _ in range(4)
            ]
        else:
            raise ValueError(f"Unknown agent: {args.agent}")
    for agent in agents:
        agent.env_name = "tarneeb"
        if args.mode == "play" and not isinstance(agent, HumanTarneebAgent):
            try:
                agent.restore()
                print(f"Loaded saved state for {agent.name}")
            except (FileNotFoundError, TypeError):
                print(
                    f"No saved state found or type mismatch for {agent.name}, using fresh agent"
                )
    runner = MultiAgentRunner[TarneebState, PartialTarneebState, TarneebAction]()
    if not args.show_plot:
        turn_plot_off()
    avg_rewards = runner.run_episodes(
        env, agents, args.episodes, record_cnt=args.record_cnt,
        checkpoint_every=args.checkpoint_every,
    )
    print(
        f"Avg rewards for Tarneeb agents over {args.episodes} episodes: {avg_rewards}"
    )
    if args.mode == "train":
        for agent in agents:
            if not isinstance(agent, HumanTarneebAgent):
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
    parser.add_argument(
        "--agent",
        type=str,
        default="mc",
        choices=["mc", "sarsa", "sarsa-lambda", "value-approx", "cnn-approx",
                 "shared-cnn", "separate-heads-cnn", "alphazero"],
        help="The RL agent to use for AI players in Tarneeb.",
    )
    parser.add_argument(
        "--checkpoint-every",
        type=int,
        default=0,
        help="Save a checkpoint every N episodes during training (0 = disabled).",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose output, including detailed game state information.",
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
        verbose=args.verbose,
        agent=args.agent,
        checkpoint_every=args.checkpoint_every,
    )
    if args.game == "easy21":
        run_easy21(args_dataclass)
    elif args.game == "tarneeb":
        run_tarneeb(args_dataclass)


if __name__ == "__main__":
    main()
