import cv2
import numpy as np

from cv.core.frame_data import FrameContext


def process_frame(context: FrameContext) -> FrameContext:
    hsv = cv2.cvtColor(context.processed, cv2.COLOR_BGR2HSV)

    range1 = (np.array([0, 120, 70]), np.array([10, 255, 255]))
    range2 = (np.array([170, 120, 70]), np.array([180, 255, 255]))
    mask1 = cv2.inRange(hsv, *range1)
    mask2 = cv2.inRange(hsv, *range2)
    mask = cv2.bitwise_or(mask1, mask2)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        biggest_contour = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(biggest_contour)

        yolo_x = (x + w / 2) / context.original.shape[1]
        yolo_y = (y + h / 2) / context.original.shape[0]
        yolo_w = w / context.original.shape[1]
        yolo_h = h / context.original.shape[0]

        new_bbox = np.array([[yolo_x, yolo_y, yolo_w, yolo_h]], dtype=np.float32)
        context.metadata.bboxes = np.vstack((context.metadata.bboxes, new_bbox))

        new_id = np.array([0], dtype=np.int32)
        context.metadata.object_ids = np.hstack((context.metadata.object_ids, new_id))

    return context
