import cv2

from cv.core.frame_data import FrameContext


class VideoSource:
    def __init__(self, left_video_path: str, right_video_path: str):
        self.cap_left = cv2.VideoCapture(left_video_path)
        self.cap_right = cv2.VideoCapture(right_video_path)

    def read_stereo_pair(self) -> tuple[FrameContext, FrameContext] | None:
        ret_l, frame_l = self.cap_left.read()
        ret_r, frame_r = self.cap_right.read()

        if not ret_l or not ret_r:
            self.release()
            return None

        return FrameContext(frame_l), FrameContext(frame_r)

    def release(self) -> None:
        if self.cap_left.isOpened():
            self.cap_left.release()
        if self.cap_right.isOpened():
            self.cap_right.release()
