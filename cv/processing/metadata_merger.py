from cv.core.frame_data import FrameContext


def dont_merge_metadata(
    context1: FrameContext, context2: FrameContext
) -> tuple[FrameContext, FrameContext]:
    return context1, context2
