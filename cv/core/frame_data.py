from dataclasses import dataclass, field

import numpy as np


@dataclass
class FrameMetaData:
    bboxes: np.ndarray = field(
        default_factory=lambda: np.empty(shape=(0, 4), dtype=np.int32)
    )
    object_ids: np.ndarray = field(
        default_factory=lambda: np.empty(shape=(0,), dtype=np.int32)
    )


class FrameContext:
    def __init__(self, image: np.ndarray):
        self.original: np.ndarray = image
        self.original.flags.writeable = False
        self.processed: np.ndarray = image.copy()

        self.metadata: FrameMetaData = FrameMetaData()
