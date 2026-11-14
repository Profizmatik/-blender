import numpy as np
import pytest

from cv.core.world_data import WorldContext
from cv.processing.time_tracker import (
    WorldTrackingNode,
    global_matching,
    linear_predict,
)

resolution = (0, 0)


@pytest.fixture
def tracker():
    return WorldTrackingNode(
        global_matching, max_age=3, predict_fn=linear_predict, max_distance=5
    )


def test_simple_3d_tracking(tracker):
    velocities = np.array([[1.0, 1.0, 1.0], [0.0, -1.0, 0.0]])
    positions = np.array([[100.0, 100.0, 100.0], [50.0, 50.0, 70.0]])
    local_ids = np.array([0, 1])

    num_frames = 30
    for frame in range(num_frames):
        curr_positions = positions + velocities * frame
        expected_ids = np.array([1, 2])

        if frame % 2 == 1:
            curr_positions = curr_positions[::-1]
            expected_ids = expected_ids[::-1]

        world_context = WorldContext(
            frame, resolution, curr_positions, local_ids.copy()
        )
        world_context = tracker.process(world_context)

        np.testing.assert_array_equal(expected_ids, world_context.object_ids)
        local_ids += np.array([1, 2])


def test_occlusion_within_max_age(tracker):
    vel = np.array([[1.0, 0.0, 0.0]])
    pos = np.array([[10.0, 10.0, 10.0]])

    for frame in range(2):
        ctx = WorldContext(frame, resolution, pos + vel * frame, np.array([0]))
        ctx = tracker.process(ctx)
        np.testing.assert_array_equal([1], ctx.object_ids)

    for frame in range(2, 4):
        ctx = WorldContext(frame, resolution, np.empty((0, 3)), np.array([], dtype=int))
        ctx = tracker.process(ctx)
        assert len(ctx.object_ids) == 0

    reappeared_pos = pos + vel * 4
    ctx = WorldContext(4, resolution, reappeared_pos, np.array([5]))
    ctx = tracker.process(ctx)
    np.testing.assert_array_equal([1], ctx.object_ids)


def test_object_lost_after_max_age(tracker):
    vel = np.array([[1.0, 0.0, 0.0]])
    pos = np.array([[10.0, 10.0, 10.0]])

    for frame in range(2):
        ctx = WorldContext(frame, resolution, pos + vel * frame, np.array([0]))
        tracker.process(ctx)

    for frame in range(2, 6):
        ctx = WorldContext(frame, resolution, np.empty((0, 3)), np.array([], dtype=int))
        tracker.process(ctx)

    ctx = WorldContext(6, resolution, pos + vel * 6, np.array([0]))
    ctx = tracker.process(ctx)
    np.testing.assert_array_equal([2], ctx.object_ids)


def test_jump_exceeds_max_distance(tracker):
    pos = np.array([[10.0, 10.0, 10.0]])
    ctx = WorldContext(0, resolution, pos, np.array([0]))
    ctx = tracker.process(ctx)
    np.testing.assert_array_equal([1], ctx.object_ids)

    jumped_pos = np.array([[16.0, 10.0, 10.0]])
    ctx = WorldContext(1, resolution, jumped_pos, np.array([0]))
    ctx = tracker.process(ctx)
    np.testing.assert_array_equal([2], ctx.object_ids)


def test_new_object_spawns_mid_sequence(tracker):
    pos1 = np.array([0.0, 0.0, 0.0])
    vel1 = np.array([0.5, 0.0, 0.0])

    pos2 = np.array([50.0, 50.0, 50.0])
    vel2 = np.array([0.0, 0.5, 0.0])

    for frame in range(3):
        curr_p = np.array([pos1 + vel1 * frame])
        ctx = WorldContext(frame, resolution, curr_p, np.array([0]))
        ctx = tracker.process(ctx)
        np.testing.assert_array_equal([1], ctx.object_ids)

    for frame in range(3, 6):
        curr_p = np.array([pos1 + vel1 * frame, pos2 + vel2 * (frame - 3)])
        ctx = WorldContext(frame, resolution, curr_p, np.array([0, 1]))
        ctx = tracker.process(ctx)
        np.testing.assert_array_equal([1, 2], ctx.object_ids)


def test_crossing_trajectories(tracker):
    pos = np.array([[0.0, 5.0, 0.0], [5.0, 0.0, 0.0]])
    vel = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])

    for frame in range(10):
        curr_pos = pos + vel * frame
        ctx = WorldContext(frame, resolution, curr_pos, np.array([0, 1]))
        ctx = tracker.process(ctx)
        np.testing.assert_array_equal([1, 2], ctx.object_ids)


def test_empty_frames_sequence(tracker):
    for frame in range(5):
        ctx = WorldContext(frame, resolution, np.empty((0, 3)), np.array([], dtype=int))
        ctx = tracker.process(ctx)
        assert len(ctx.object_ids) == 0
