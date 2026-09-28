from dataclasses import dataclass, field

import numpy as np


@dataclass
class WorldContext:
    positions: np.ndarray = field(
        default_factory=lambda: np.empty(shape=(0, 3), dtype=np.int32)
    )
    object_ids: np.ndarray = field(
        default_factory=lambda: np.empty(shape=(0,), dtype=np.int32)
    )
