import cv2

from cv.core.frame_data import FrameContext
from cv.core.logging import log, ArgsType, KwargsType


def log_stereo_pair(
    stereo_pair: tuple[FrameContext, FrameContext], _: ArgsType, __: KwargsType
) -> str:
    if stereo_pair is None:
        return "end"
    return (
        "left: "
        + repr(stereo_pair[0].metadata)
        + " right: "
        + repr(stereo_pair[1].metadata)
    )


class VideoSourceNode:
    def __init__(self, left_video_path: str, right_video_path: str):
        self.cap_left = cv2.VideoCapture(left_video_path)
        self.cap_right = cv2.VideoCapture(right_video_path)
        self._frame_id = 0

    @log(title="video source", log_fn=log_stereo_pair)
    def process(self) -> tuple[FrameContext, FrameContext] | None:
        ret_l, frame_l = self.cap_left.read()
        ret_r, frame_r = self.cap_right.read()

        if not ret_l or not ret_r:
            self.release()
            return None
        sterio_pair = (
            FrameContext(frame_l, self._frame_id),
            FrameContext(frame_r, self._frame_id),
        )
        self._frame_id += 1

        return sterio_pair

    def release(self) -> None:
        if self.cap_left.isOpened():
            self.cap_left.release()
        if self.cap_right.isOpened():
            self.cap_right.release()
