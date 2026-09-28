import matplotlib.pyplot as plt
import pandas as pd

from cv.evaluation.metrics import calculate_metrics
from cv.evaluation.plotting import plot_3d_trail, plot_error_graphs


def analize(true_data_path: str, processed_data_path: str, save_path: str) -> None:
    true_df = pd.read_csv(true_data_path)
    proc_df = pd.read_csv(processed_data_path)

    true_arr = true_df[["X", "Y", "Z"]].to_numpy()
    proc_arr = proc_df[["X", "Y", "Z"]].to_numpy()

    data, errors = calculate_metrics(true_arr, proc_arr)

    figure = plt.figure(figsize=(20, 10))
    grid = plt.GridSpec(4, 6, wspace=1.5, hspace=0.2)

    axis_3d = figure.add_subplot(grid[:, :3], projection="3d")

    axes_2d = [
        figure.add_subplot(grid[0, 3:]),
        figure.add_subplot(grid[1, 3:]),
        figure.add_subplot(grid[2, 3:]),
        figure.add_subplot(grid[3, 3:]),
    ]

    plot_3d_trail(axis_3d, data)
    plot_error_graphs(axes_2d, data, errors)

    plt.savefig(save_path, bbox_inches="tight")
