import matplotlib.pyplot as plt
from matplotlib import cm
import numpy as np
from matplotlib.ticker import MaxNLocator

__show_plot = True


def turn_plot_off() -> None:
    global __show_plot
    __show_plot = False


def plot_value_function(
    v_star: list[tuple[int, int, float]], title: str, xlabel: str, ylabel: str
) -> None:
    """Plots the state-value function V* as a 3D surface plot.

    Args:
        v_star: A list of tuples (x, y, value) representing the state-value function.
        title: The title of the plot.
    """
    global __show_plot
    if not __show_plot:
        print("plot disabled")
        return

    x_vals = [x for x, _, _ in v_star]
    y_vals = [y for _, y, _ in v_star]
    V = {(x, y): v for x, y, v in v_star}

    x_range = np.arange(min(x_vals), max(x_vals) + 1)
    y_range = np.arange(min(y_vals), max(y_vals) + 1)
    X, Y = np.meshgrid(x_range, y_range)
    Z = np.apply_along_axis(
        lambda coord: V[(coord[0], coord[1])] if (coord[0], coord[1]) in V else -1,
        2,
        np.dstack([X, Y]),
    )

    def plot_surface(
        X: np.ndarray,
        Y: np.ndarray,
        Z: np.ndarray,
        title: str,
    ) -> None:
        fig = plt.figure(figsize=(20, 10))
        ax = fig.add_subplot(111, projection="3d")
        surf = ax.plot_surface(
            X,
            Y,
            Z,
            rstride=1,
            cstride=1,
            cmap=cm.coolwarm,
            vmin=-1.0,
            vmax=1.0,
        )
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_zlabel("Value")
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        ax.yaxis.set_major_locator(MaxNLocator(integer=True))
        ax.set_title(title)
        ax.view_init(ax.elev, 120)
        fig.colorbar(surf)
        plt.show()

    plot_surface(X, Y, Z, title)
