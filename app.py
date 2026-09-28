from typing import Final

from cv.core.engine import Engine
from cv.core.pipeline import Pipeline
from cv.evaluation.flows import analize
from cv.geometry.triangulator import create_triangulator
from cv.io.csv_result_handler import CSVResultHandler
from cv.io.video_source import VideoSource
from cv.processing.frame_processor import process_frame
from cv.processing.metadata_merger import dont_merge_metadata

LEFT_CAMERA_SOURCE_PAHT: Final[str] = "./left_camera_25fps.mp4"
RIGHT_CAMERA_SOURCE_PAHT: Final[str] = "./right_camera_25fps.mp4"
TRUE_DATA_PATH: Final[str] = "coords.csv"
PROCESSED_DATA_PATH: Final[str] = "fake_coords.csv"
OUTPUT_METRICS_PATH: Final[str] = "analize.png"

BASELINE: Final[float] = 30.0
F_MM: Final[float] = 50.0
W_MM: Final[float] = 36.0


if __name__ == "__main__":
    triangulate = create_triangulator(f_mm=F_MM, w_mm=W_MM, baseline=BASELINE)
    frame_reader = VideoSource(LEFT_CAMERA_SOURCE_PAHT, RIGHT_CAMERA_SOURCE_PAHT)
    engine = Engine(process_frame, dont_merge_metadata, triangulate)

    res_handler = CSVResultHandler(PROCESSED_DATA_PATH)
    with res_handler:
        pipeline = Pipeline(engine, frame_reader, res_handler)
        pipeline.start()

    analize(TRUE_DATA_PATH, PROCESSED_DATA_PATH, OUTPUT_METRICS_PATH)
