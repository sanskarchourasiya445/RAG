"""
app/utils/timer.py

Responsibility: a single, reusable way to measure elapsed time, used for
the console's "response generation time" / "total query time" reporting.

Implemented as a context manager rather than repeating
`start = time.time(); ...; elapsed = time.time() - start` at every call
site, which is easy to get subtly wrong (e.g. forgetting to capture
elapsed time when an exception is raised mid-block).
"""

from __future__ import annotations

import time
from types import TracebackType
from typing import Optional, Type


class Timer:
    """Usage:

        with Timer() as t:
            do_something_slow()
        print(t.elapsed_seconds)

    elapsed_seconds is populated even if the block raises, since __exit__
    always runs and we record elapsed time before re-raising (we return
    False, i.e. we never suppress exceptions).
    """

    def __init__(self) -> None:
        self.start_time: Optional[float] = None
        self.elapsed_seconds: float = 0.0

    def __enter__(self) -> "Timer":
        self.start_time = time.perf_counter()
        return self

    def __exit__(
        self,
        exc_type: Optional[Type[BaseException]],
        exc_val: Optional[BaseException],
        exc_tb: Optional[TracebackType],
    ) -> bool:
        assert self.start_time is not None
        self.elapsed_seconds = time.perf_counter() - self.start_time
        return False  # never suppress exceptions

    @property
    def elapsed_display(self) -> str:
        return f"{self.elapsed_seconds:.2f}s"