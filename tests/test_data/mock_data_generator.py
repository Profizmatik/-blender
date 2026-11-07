import cv2
import numpy as np


def crate_test_data(bboxes: np.ndarray, resolution: tuple[int, int]) -> np.ndarray:
    background = np.array(cv2.imread("tests/test_data/backgrounds/background-1.png"))
    background = cv2.imread("tests/test_data/backgrounds/background-1.png")
    if background is None:
        raise RuntimeError("create test data: can not read background image")

    plane = cv2.imread("tests/test_data/planes/plane-2.png", cv2.IMREAD_UNCHANGED)
    if plane is None:
        raise RuntimeError("create test data: can not read plane image")

    res_w, res_h = resolution
    background = cv2.resize(background, (res_w, res_h), interpolation=cv2.INTER_LINEAR)

    for bbox in bboxes:
        cx, cy, w, h = bbox
        pixel_w = round(w * res_w)
        pixel_h = round(h * res_h)

        plane = cv2.resize(plane, (pixel_w, pixel_h), interpolation=cv2.INTER_LINEAR)

        start_x = round((cx - w / 2) * res_w)
        start_y = round((cy - h / 2) * res_h)

        start_x = max(0, min(start_x, res_w - pixel_w))
        start_y = max(0, min(start_y, res_h - pixel_h))

        plane_h, plane_w, _ = plane.shape
        roi = background[start_y : start_y + plane_h, start_x : start_x + plane_w]

        b, g, r, alpha = cv2.split(plane)
        overlay_color = cv2.merge((b, g, r))

        mask = (alpha / 255.0)[:, :, np.newaxis]
        mask_inv = 1.0 - mask

        blended_roi = roi * mask_inv + mask * overlay_color

        background[start_y : start_y + plane_h, start_x : start_x + plane_w] = (
            blended_roi.astype(np.uint8)
        )

    return background


def create_cost_matrix(bboxes1: np.ndarray, bboxes2: np.ndarray) -> np.ndarray:
    cost_matrix = np.linalg.norm(
        bboxes1[:, np.newaxis, :] - bboxes2[np.newaxis, :, :], axis=2
    )
    return cost_matrix
