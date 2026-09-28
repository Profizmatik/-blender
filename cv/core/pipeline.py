from dataclasses import dataclass
from typing import Protocol

from cv.core.engine import Engine
from cv.core.frame_data import FrameContext
from cv.core.world_data import WorldContext


class StereoSource(Protocol):
    def read_stereo_pair(self) -> tuple[FrameContext, FrameContext] | None:
        pass


class ResultHandler(Protocol):
    def handle(self, world_context: WorldContext) -> None:
        pass


@dataclass
class Pipeline:
    engine: Engine
    source: StereoSource
    result_handler: ResultHandler
    is_running: bool = False

    def start(self):
        self.is_running = True
        while self.is_running:
            stereo_pair = self.source.read_stereo_pair()

            if stereo_pair is None:
                self.is_running = False
                break

            ctx_left, ctx_right = stereo_pair
            world_ctx: WorldContext = self.engine.process(ctx_left, ctx_right)

            self.result_handler.handle(world_ctx)
