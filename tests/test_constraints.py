import numpy as np

from cv.core.frame_data import FrameMetaData
from cv.processing.constraints import (
    get_edge_distance_filter,
    get_object_distance_filter,
)

resolution = (0, 0)  # does not metter


# ----- edge distance filter -----


def test_edge_distance_filter_one_object():
    edge_distance_filter = get_edge_distance_filter(0.05, 0.05)
    bboxes = np.array([[0.7, 0.8, 0.3, 0.1]])
    ids = np.array([0])

    metadata = FrameMetaData(0, resolution, bboxes, ids)

    metadata = edge_distance_filter(metadata)
    assert len(metadata.bboxes) != 0


def test_edge_distance_filter_object_perfectly_in_center():
    edge_distance_filter = get_edge_distance_filter(x_margin=0.1, y_margin=0.1)
    bboxes = np.array([[0.5, 0.5, 0.2, 0.2]])
    ids = np.array([10])

    metadata = FrameMetaData(0, resolution, bboxes, ids)
    filtered_metadata = edge_distance_filter(metadata)

    assert len(filtered_metadata.bboxes) == 1
    assert filtered_metadata.object_ids[0] == 10


def test_edge_distance_filter_object_too_close_to_left_edge():
    edge_distance_filter = get_edge_distance_filter(x_margin=0.1, y_margin=0.1)
    bboxes = np.array([[0.12, 0.5, 0.1, 0.2]])
    ids = np.array([20])

    metadata = FrameMetaData(0, resolution, bboxes, ids)
    filtered_metadata = edge_distance_filter(metadata)

    assert len(filtered_metadata.bboxes) == 0


def test_edge_distance_filter_object_too_close_to_right_edge():
    edge_distance_filter = get_edge_distance_filter(x_margin=0.1, y_margin=0.1)
    bboxes = np.array([[0.86, 0.5, 0.1, 0.2]])
    ids = np.array([30])

    metadata = FrameMetaData(0, resolution, bboxes, ids)
    filtered_metadata = edge_distance_filter(metadata)

    assert len(filtered_metadata.bboxes) == 0


def test_edge_distance_filter_object_too_close_to_top_edge():
    edge_distance_filter = get_edge_distance_filter(x_margin=0.1, y_margin=0.1)
    bboxes = np.array([[0.5, 0.14, 0.2, 0.1]])
    ids = np.array([40])

    metadata = FrameMetaData(0, resolution, bboxes, ids)
    filtered_metadata = edge_distance_filter(metadata)

    assert len(filtered_metadata.bboxes) == 0


def test_edge_distance_filter_object_too_close_to_bottom_edge():
    edge_distance_filter = get_edge_distance_filter(x_margin=0.1, y_margin=0.1)
    bboxes = np.array([[0.5, 0.86, 0.2, 0.1]])
    ids = np.array([50])

    metadata = FrameMetaData(0, resolution, bboxes, ids)
    filtered_metadata = edge_distance_filter(metadata)

    assert len(filtered_metadata.bboxes) == 0


def test_edge_distance_filter_multiple_objects_mixed_filtering():
    edge_distance_filter = get_edge_distance_filter(x_margin=0.1, y_margin=0.1)
    bboxes = np.array(
        [
            [0.5, 0.5, 0.2, 0.2],
            [0.05, 0.5, 0.05, 0.1],
            [0.4, 0.4, 0.1, 0.1],
            [0.5, 0.95, 0.1, 0.2],
        ]
    )
    ids = np.array([100, 101, 102, 103])

    metadata = FrameMetaData(0, resolution, bboxes, ids)
    filtered_metadata = edge_distance_filter(metadata)

    assert len(filtered_metadata.bboxes) == 2
    np.testing.assert_array_equal(filtered_metadata.object_ids, np.array([100, 102]))


def test_edge_distance_filter_empty_frame():
    edge_distance_filter = get_edge_distance_filter(0.05, 0.05)
    bboxes = np.empty((0, 4))
    ids = np.array([])

    metadata = FrameMetaData(0, resolution, bboxes, ids)
    filtered_metadata = edge_distance_filter(metadata)

    assert len(filtered_metadata.bboxes) == 0
    assert len(filtered_metadata.object_ids) == 0


def test_edge_distance_filter_zero_margins():
    edge_distance_filter = get_edge_distance_filter(0.0, 0.0)
    bboxes = np.array(
        [
            [0.05, 0.5, 0.1, 0.2],
            [0.95, 0.5, 0.1, 0.2],
        ]
    )
    ids = np.array([1, 2])

    metadata = FrameMetaData(0, resolution, bboxes, ids)
    filtered_metadata = edge_distance_filter(metadata)

    assert len(filtered_metadata.bboxes) == 2


# ----- object distance filter -----


def test_object_distance_filter_objects_too_close():
    distance_filter = get_object_distance_filter(x_margin=0.05, y_margin=0.3)
    bboxes = np.array(
        [
            [0.5, 0.5, 0.1, 0.1],
            [0.5, 0.7, 0.1, 0.1],
        ]
    )
    ids = np.array([10, 20])

    metadata = FrameMetaData(0, resolution, bboxes, ids)
    filtered_metadata = distance_filter(metadata)

    assert len(filtered_metadata.bboxes) == 0
    assert len(filtered_metadata.object_ids) == 0


def test_object_distance_filter_objects_far_away():
    distance_filter = get_object_distance_filter(x_margin=0.1, y_margin=0.1)
    bboxes = np.array([[0.2, 0.2, 0.1, 0.1], [0.8, 0.8, 0.1, 0.1]])
    ids = np.array([10, 20])

    metadata = FrameMetaData(0, resolution, bboxes, ids)
    filtered_metadata = distance_filter(metadata)

    assert len(filtered_metadata.bboxes) == 2
    assert len(filtered_metadata.object_ids) == 2
    np.testing.assert_array_equal(filtered_metadata.object_ids, np.array([10, 20]))


def test_object_distance_filtert_objects_close_only_by_one_axis():
    distance_filter = get_object_distance_filter(x_margin=0.1, y_margin=0.1)
    bboxes = np.array([[0.2, 0.5, 0.1, 0.1], [0.8, 0.5, 0.1, 0.1]])
    ids = np.array([10, 20])

    metadata = FrameMetaData(0, resolution, bboxes, ids)
    filtered_metadata = distance_filter(metadata)

    assert len(filtered_metadata.bboxes) == 2
    np.testing.assert_array_equal(filtered_metadata.object_ids, np.array([10, 20]))


def test_object_distance_filter_empty_frame():
    distance_filter = get_object_distance_filter(x_margin=0.05, y_margin=0.05)
    bboxes = np.empty((0, 4))
    ids = np.array([])

    metadata = FrameMetaData(0, resolution, bboxes, ids)
    filtered_metadata = distance_filter(metadata)

    assert len(filtered_metadata.bboxes) == 0
    assert len(filtered_metadata.object_ids) == 0


def test_object_distance_filter_single_object():
    distance_filter = get_object_distance_filter(x_margin=0.9, y_margin=0.9)
    bboxes = np.array([[0.5, 0.5, 0.2, 0.2]])
    ids = np.array([42])

    metadata = FrameMetaData(0, resolution, bboxes, ids)
    filtered_metadata = distance_filter(metadata)

    assert len(filtered_metadata.bboxes) == 1
    assert filtered_metadata.object_ids[0] == 42
