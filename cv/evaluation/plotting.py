from matplotlib.axes import Axes
from mpl_toolkits.mplot3d.axes3d import Axes3D

from cv.evaluation.data import TrajectoryData, TrajectoryErrors


def plot_3d_trail(axes_3d: Axes3D, data: TrajectoryData) -> None:
    axes_3d.set_title("Ball Trail")
    axes_3d.plot(
        data.true_arr[:, 0],
        data.true_arr[:, 1],
        data.true_arr[:, 2],
        color="blue",
        linewidth=2,
        label="Original (Blender)",
    )
    axes_3d.plot(
        data.proc_arr[:, 0],
        data.proc_arr[:, 1],
        data.proc_arr[:, 2],
        color="red",
        linewidth=1,
        linestyle="--",
        label="Processed (Algo)",
    )
    axes_3d.set_xlabel("X")
    axes_3d.set_ylabel("Y")
    axes_3d.set_zlabel("Z")
    axes_3d.set_zlim(-0.3, 0.3)
    axes_3d.set_ylim(-0.3, 0.3)
    axes_3d.legend()


def plot_error_graphs(
    axes: list[Axes], data: TrajectoryData, errors: TrajectoryErrors
) -> None:
    ax_dist, ax_x, ax_y, ax_z = axes

    ax_dist.plot(data.true_distance, errors.distance)
    ax_dist.set_title("Errors")
    ax_dist.set_ylabel("Distance error")
    ax_dist.set_xlabel("Distance")
    ax_dist.set_ylim(-4, 4)

    ax_x.plot(data.true_distance, errors.x)
    ax_x.set_ylabel("X error")
    ax_x.set_ylim(-4, 4)

    ax_y.plot(data.true_distance, errors.y)
    ax_y.set_ylabel("Y error")
    ax_y.set_ylim(-4, 4)

    ax_z.plot(data.true_distance, errors.z)
    ax_z.set_ylabel("Z error")
    ax_z.set_xlabel("Distance")
    ax_z.set_ylim(-4, 4)
