import functools
import logging
from collections.abc import Callable
from typing import Any, ParamSpec, TypeVar

P = ParamSpec("P")
R = TypeVar("R")
type ResultType = Any
type ArgsType = tuple[Any, ...]
type KwargsType = dict[str, Any]
type LogFn = Callable[[ResultType, ArgsType, KwargsType], str]

logging.basicConfig(
    filename="cv.log",
    filemode="a",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    encoding="utf-8",
)
logger = logging.getLogger(__name__)


def log_result(result: ResultType, _: ArgsType, __: KwargsType) -> str:
    return repr(result)


def log_params(_: ResultType, args: ArgsType, kwargs: KwargsType) -> str:
    parts = []

    if args:
        args_str = ", ".join(repr(arg) for arg in args)
        parts.append(f"args: ({args_str})")

    if kwargs:
        kwargs_str = ", ".join(f"{key}={val!r}" for key, val in kwargs.items())
        parts.append(f"kwargs: {{{kwargs_str}}}")

    return " | ".join(parts)


def log(
    title: str | None = None, log_fn: LogFn = log_result
) -> Callable[[Callable[P, R]], Callable[P, R]]:
    def log_it(func: Callable[P, R]) -> Callable[P, R]:
        log_title = title or func.__name__

        @functools.wraps(func)
        def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
            try:
                result = func(*args, **kwargs)
                logger.info(f"'{log_title}' returned: {log_fn(result, args, kwargs)}")
            except Exception as e:
                logger.error(f"'{log_title}' failed with error {e!r}", exc_info=True)
                raise

            return result

        return wrapper

    return log_it
