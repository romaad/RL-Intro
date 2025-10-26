import matplotlib.pyplot as plt
from matplotlib import cm
import numpy as np


def plot_value_function(
    v_star: list[tuple[int, int, float]],
    title: str = "State-Value Function V*",
) -> None:
    """Plots the state-value function V* as a 3D surface plot.

    Args:
        v_star: A list of tuples (x, y, value) representing the state-value function.
        title: The title of the plot.
    """
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")

    x_vals = [x for x, _, _ in v_star]
    y_vals = [y for _, y, _ in v_star]
    z_vals = [v for _, _, v in v_star]

    xi = np.linspace(min(x_vals), max(x_vals), len(set(x_vals)))
    yi = np.linspace(min(y_vals), max(y_vals), len(set(y_vals)))
    xi, yi = np.meshgrid(xi, yi)

    zi = np.array(z_vals).reshape(len(set(y_vals)), len(set(x_vals)))

    surf = ax.plot_surface(
        xi,
        yi,
        zi,
        cmap=cm.viridis,
        linewidth=0,
        antialiased=False,
    )
    ax.set_xlabel("Player Sum")
    ax.set_ylabel("Dealer Sum")
    ax.set_zlabel("V* Value")
    ax.set_title(title)
    fig.colorbar(surf, shrink=0.5, aspect=5)

    plt.show(block=True)
