from cv.core.frame_data import FrameMetaData
from dataclasses import dataclass, field
from typing import Callable
import numpy as np
from cv.processing.time_tracker import convert_to_corner_coords, create_iou_matrix

type FilterFn = Callable[[FrameMetaData], FrameMetaData]
type GroupFilterFn = Callable[
    [FrameMetaData, FrameMetaData], tuple[FrameMetaData, FrameMetaData]
]


@dataclass
class FilterNode:
    left_filters: list[FilterFn] = field(default_factory=list)
    right_filters: list[FilterFn] = field(default_factory=list)
    group_filters: list[GroupFilterFn] = field(default_factory=list)

    def process(
        self, metadata1: FrameMetaData, metadata2: FrameMetaData
    ) -> tuple[FrameMetaData, FrameMetaData]:
        for left_filter, right_filter in zip(self.left_filters, self.right_filters):
            metadata1, metadata2 = left_filter(metadata1), right_filter(metadata2)

        for group_filter in self.group_filters:
            metadata1, metadata2 = group_filter(metadata1, metadata2)

        return metadata1, metadata2


def get_edge_distance_filter(x_margin: float, y_margin: float) -> FilterFn:
    def edge_distance_filter(metadata: FrameMetaData) -> FrameMetaData:
        bboxes = convert_to_corner_coords(metadata.bboxes)
        bboxes += np.array([-x_margin, -y_margin, x_margin, y_margin])

        mask = np.any(((bboxes > 1) | (bboxes < 0)), axis=1)

        metadata.bboxes = metadata.bboxes[~mask]
        metadata.object_ids = metadata.object_ids[~mask]

        return metadata

    return edge_distance_filter


def get_object_distance_filter(x_margin: float, y_margin: float) -> FilterFn:
    def get_intersect_mask(segments: np.ndarray) -> np.ndarray:
        starts = segments[:, 0]
        ends = segments[:, 1]

        start_le_end = starts[:, np.newaxis] <= ends[np.newaxis, :]

        return start_le_end & start_le_end.T

    def object_distance_filter(metadata: FrameMetaData) -> FrameMetaData:
        bboxes = convert_to_corner_coords(metadata.bboxes)
        bboxes += np.array([-x_margin, -y_margin, x_margin, y_margin])

        x = bboxes[:, [0, 2]]
        y = bboxes[:, [1, 3]]

        mask = get_intersect_mask(x) & get_intersect_mask(y)
        mask[np.eye(len(x), dtype=np.bool)] = False
        filter = np.any(mask, axis=1)

        metadata.bboxes = metadata.bboxes[~filter]
        metadata.object_ids = metadata.object_ids[~filter]

        return metadata

    return object_distance_filter
