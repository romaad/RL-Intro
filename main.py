from base import Runner
from easy21 import Easy21Env, Easy21State, Easy21Action, NaiveAgent


def run_easy21() -> None:
    n_episodes = 10000
    runner = Runner[Easy21State, Easy21Action]()
    env = Easy21Env()
    agent = NaiveAgent()
    r = runner.run_episodes(env, agent, n_episodes, record_cnt=10)
    print(f"Avg reward over {n_episodes} episodes: {r}")


def main() -> None:
    run_easy21()


if __name__ == "__main__":
    main()
