"""Task execution instrumentation and orchestration primitives.

Provides wrappers and utilities for managing the lifecycle of Hazen analysis
tasks. Currently focused on execution timing and performance metadata collection
via :func:`timed_execution`.

This module serves as the foundation for future multi-task orchestration
capabilities, including aggregate metrics, parallel execution, and pipeline
coordination.
"""

from __future__ import annotations

# Type checking
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Callable

# Python imports
import time
from typing import ParamSpec

# Local imports
from hazenlib.types import Measurement, Result

P = ParamSpec("P")


def timed_execution(
    task_method: Callable[P, Result],
    *args: P.args,
    **kwargs: P.kwargs,
) -> Result:
    """Execute task method and append timing metadata to result.

    Example:
        >>> result = timed_execution(task.run, calc='T1', plate_number=4)
        >>> # result.measurements now contains ExecutionMetadata entry

    Args:
        task_method: Callable that returns a Result object
        *args: Positional arguments passed to task_method
        **kwargs: Keyword arguments passed to task_method

    Returns:
        Result: The original result with added execution timing measurement

    """
    start: float = time.perf_counter()

    # Execute the actual task
    result: Result = task_method(*args, **kwargs)
    # Calculate and inject timing
    elapsed: float = time.perf_counter() - start
    result.add_measurement(
        Measurement(
            name="ExecutionMetadata",
            type="measured",
            description="analysis_duration",
            value=round(elapsed, 4),
            unit="s",
            visibility="intermediate",
        ),
    )
    return result
