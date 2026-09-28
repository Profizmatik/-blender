from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class TrajectoryData:
    true_arr: np.ndarray
    proc_arr: np.ndarray
    true_distance: np.ndarray


@dataclass(frozen=True)
class TrajectoryErrors:
    distance: np.ndarray
    x: np.ndarray
    y: np.ndarray
    z: np.ndarray
