from base import Agent, Runner
from easy21 import Easy21Env, Easy21State, Easy21Action, MCEasy21Agent, NaiveAgent


def run_easy21() -> None:
    agents: list[Agent[Easy21State, Easy21Action]] = [
        # NaiveAgent(),
        MCEasy21Agent(),
    ]
    # n_episodes = 100_000  # 100k episodes
    n_episodes = 1000_000  # 1 million episodes
    runner = Runner[Easy21State, Easy21Action]()
    for agent in agents:
        env = Easy21Env()
        r = runner.run_episodes(env, agent, n_episodes, record_cnt=10)
        print(f"Avg reward {agent.name} over {n_episodes} episodes: {r}")


def main() -> None:
    run_easy21()


if __name__ == "__main__":
    main()
