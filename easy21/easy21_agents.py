from agents.monte_carlo import MonteCarloAgent
from agents.sarsa import SarsaAgent, SarsaLambdaAgent
from agents.value_approx import LinearApproxAgent, LinearValueApproximator
from base import Agent
from easy21.easy21 import Easy21Action, Easy21State
from easy21.feature_extractor import CUSTOM_FEATURES_LEN, custom_easy21_q_extractor


class NaiveAgent(Agent[Easy21State, Easy21Action]):

    def act(self, s: Easy21State) -> Easy21Action:
        if s.player_sum >= 20:
            return Easy21Action.STICK
        else:
            return Easy21Action.HIT


class _Eas21ControlBaseAgent:
    def action_space(self) -> list[Easy21Action]:
        return [Easy21Action.HIT, Easy21Action.STICK]

    def state_to_xy(self, s: Easy21State) -> tuple[int, int]:
        return (s.player_sum, s.dealer_sum)

    def get_xy_labels(self) -> tuple[str, str]:
        return ("Player sum", "Dealer showing")


class MCEasy21Agent(_Eas21ControlBaseAgent, MonteCarloAgent[Easy21State, Easy21Action]):
    """An agent that uses Monte Carlo methods to learn the value function for Easy21."""

    pass


class SarsaEasy21Agent(
    _Eas21ControlBaseAgent,
    SarsaAgent[Easy21State, Easy21Action],
    MonteCarloAgent[Easy21State, Easy21Action],
):
    """SARSA agent for Easy21 environment with dp."""

    pass


class SarsaLambdaEasy21Agent(
    _Eas21ControlBaseAgent,
    SarsaLambdaAgent[Easy21State, Easy21Action],
    MonteCarloAgent[Easy21State, Easy21Action],
):
    """SARSA(λ) agent for Easy21 environment with dp."""

    pass


class Easy21LinearValueApprox(LinearValueApproximator[Easy21State, Easy21Action]):
    """Linear value function approximator for Easy21."""

    def __init__(self) -> None:
        super().__init__(
            feature_extractor=custom_easy21_q_extractor,
            feature_vector_size=CUSTOM_FEATURES_LEN,
            alpha=0.01,
        )


class SarsaLambdaEasy21LinearApproxAgent(
    _Eas21ControlBaseAgent,
    LinearApproxAgent[Easy21State, Easy21Action],
    SarsaLambdaAgent[Easy21State, Easy21Action],
):
    """SARSA(λ) agent for Easy21 environment with linear function approximation."""

    def __init__(self, lambbda: float, gamma: float) -> None:
        LinearApproxAgent.__init__(self, value_approximator=Easy21LinearValueApprox())
        SarsaLambdaAgent.__init__(self, lambbda=lambbda, gamma=gamma)
