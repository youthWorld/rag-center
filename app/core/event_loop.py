from __future__ import annotations

import asyncio
import sys


def selector_event_loop_factory(*, use_subprocess: bool = False) -> asyncio.AbstractEventLoop:
    """Create an event loop compatible with psycopg's async connections."""

    del use_subprocess
    if sys.platform == "win32":
        return asyncio.SelectorEventLoop()
    return asyncio.new_event_loop()
