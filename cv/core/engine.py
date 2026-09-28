from collections.abc import Callable
from dataclasses import dataclass

from cv.core.frame_data import FrameContext
from cv.core.world_data import WorldContext

type MergeMetaDataFn = Callable[
    [FrameContext, FrameContext], tuple[FrameContext, FrameContext]
]
type TriangulateFn = Callable[[FrameContext, FrameContext], WorldContext]
type ProcessFrameFn = Callable[[FrameContext], FrameContext]


@dataclass
class Engine:
    process_frame: ProcessFrameFn
    merge_meta_data: MergeMetaDataFn
    triangulate: TriangulateFn

    def process(self, context1: FrameContext, context2: FrameContext) -> WorldContext:
        context1 = self.process_frame(context1)
        context2 = self.process_frame(context2)
        context1, context2 = self.merge_meta_data(context1, context2)
        return self.triangulate(context1, context2)
