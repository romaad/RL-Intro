from agents.monte_carlo import MonteCarloAgent
from agents.q_agent import QAgent
from base import Action, State, Step


class SarsaAgent(QAgent[State, Action]):
    """
    SARSA agent
    The idea is similar to Monte Carlo, but we update the Q-values
    after each step using the SARSA update rule which simulates an optical Q' update
    https://davidstarsilver.wordpress.com/wp-content/uploads/2025/04/lecture-5-model-free-control-.pdf (see slide 22)
    """

    _gamma: float  # discount factor

    def __init__(self, gamma: float = 1.0) -> None:
        """
        alpha: learning rate
        gamma: discount factor
        """
        super().__init__()
        self._gamma = gamma

    # the act function is inherited from MonteCarloAgent (epsilon-greedy)

    def update_step(
        self,
        s: State,
        a: Action,
        r: float,
        s_next: State,
        a_next: Action,
    ) -> None:
        """Performs the SARSA Q-value update for a single step."""
        q_sa = self.q_value(s, a)
        q_snext_anext = self.q_value(s_next, a_next)
        # use alpha = 1/N(s,a) as in MC agent
        alpha = self.get_variable_learning_rate(s, a)
        # SARSA update rule
        q_sa_new = q_sa + alpha * (r + self._gamma * q_snext_anext - q_sa)
        self.update_q_value(s, a, q_sa_new)

    def update(self, steps: list[Step[State, Action]]) -> None:
        """No-op: all updates are done in update_step."""
        pass


class SarsaLambdaAgent(SarsaAgent[State, Action]):
    """
    SARSA(λ) agent
    The idea is similar to SARSA, but we use eligibility traces to update the Q-values
    For more state-action pairs than just the current one
    https://davidstarsilver.wordpress.com/wp-content/uploads/2025/04/lecture-5-model-free-control-.pdf (see slide 29)
    """

    _eligibility: dict[tuple[State, Action], float]
    _lambda: float  # eligibility trace decay factor

    def __init__(self, lambbda: float, gamma: float = 1.0) -> None:
        super().__init__(gamma)
        self._eligibility = {}
        self._lambda = lambbda

    @property
    def name(self) -> str:
        return f"SARSA(λ={self._lambda})"

    def update_step(
        self,
        s: State,
        a: Action,
        r: float,
        s_next: State,
        a_next: Action,
    ) -> None:
        """Performs the SARSA(λ) Q-value update for a single step."""
        q_sa = self.q_value(s, a)
        q_snext_anext = self.q_value(s_next, a_next)
        # use alpha = 1/N(s,a) as in MC agent
        alpha = self.get_variable_learning_rate(s, a)

        # update eligibility trace
        self._eligibility[(s, a)] = self._eligibility.get((s, a), 0.0) + 1.0

        # compute TD error
        delta = r + self._gamma * q_snext_anext - q_sa

        # update Q-values for all state-action pairs
        for (s_e, a_e), e_trace in self._eligibility.items():
            q_e = self.q_value(s_e, a_e)
            q_e_new = q_e + alpha * delta * e_trace
            self.update_q_value(s_e, a_e, q_e_new)
            # decay eligibility trace
            self._eligibility[(s_e, a_e)] = self._gamma * self._lambda * e_trace
