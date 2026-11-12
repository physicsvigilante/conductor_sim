"""Resource cleanup that never hides an already-active physics/user exception."""
from __future__ import annotations

from collections.abc import Callable, Iterable


def cleanup_resources(actions: Iterable[tuple[str, Callable[[], object]]],
                      primary: BaseException | None = None) -> None:
    """Attempt every cleanup, preserving a primary error and its diagnostics.

    With no active failure the first cleanup error is raised normally. With an
    active failure, secondary errors are attached as strings in cleanup_errors
    (also exception notes on Python 3.11+), never silently substituted for it.
    This covers cancellation/KeyboardInterrupt as well as ordinary failures.
    """
    failures: list[tuple[str, BaseException]] = []
    for name, action in actions:
        try:
            action()
        except BaseException as error:
            failures.append((name, error))
    if not failures:
        return
    target = primary if primary is not None else failures[0][1]
    details = tuple(f"{name}: {type(error).__name__}: {error}"
                    for name, error in failures)
    target.cleanup_errors = (*getattr(target, 'cleanup_errors', ()), *details)
    add_note = getattr(target, 'add_note', None)
    if callable(add_note):
        for detail in details:
            add_note('Secondary cleanup failure: ' + detail)
    if primary is None:
        raise target
