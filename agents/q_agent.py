import random
from base import Action, Agent, State
from plot import plot_value_function


class QAgent(Agent[State, Action]):
    """An agent that uses Q-based learning."""

    def action_space(self) -> list[Action]:
        raise NotImplementedError

    def get_epsilon(self, s: State) -> float:
        raise NotImplementedError

    def q_value(self, s: State, a: Action) -> float:
        raise NotImplementedError

    def visit(self, s: State) -> None:
        """Optional method to record the visit of a state-action pair."""
        pass

    def update_q_value(self, s: State, a: Action, value: float) -> None:
        raise NotImplementedError

    def act(self, s: State) -> Action:
        # Epsilon-greedy policy
        self.visit(s)  # record state visit
        epsilon = self.get_epsilon(s)
        if random.uniform(0, 1) < epsilon:
            return random.choice(self.action_space())
        else:
            # MC ARGMAX action selection
            return max(
                self.action_space(),
                key=lambda a: self.q_value(s, a),
            )

    def state_to_xy(self, s: State) -> tuple[int, int]:
        raise NotImplementedError

    def get_xy_labels(self) -> tuple[str, str]:
        raise NotImplementedError

    def get_states(self) -> list[State]:
        raise NotImplementedError

    def on_train_end(self) -> None:
        v_star = [
            (
                *self.state_to_xy(s),
                max(self.q_value(s, a) for a in self.action_space()),
            )
            for s in self.get_states()
        ]
        labelx, labely = self.get_xy_labels()
        plot_value_function(
            v_star,
            title="State-Value Function V* after Training",
            xlabel=labelx,
            ylabel=labely,
        )
