from __future__ import annotations

import abc
import asyncio
import logging
import time
from collections.abc import Callable

logger = logging.getLogger(__name__)

_WAIT_DEAD_PROBE_INTERVAL_SECONDS = 1.0


class WorkerUnreachableError(Exception):
    pass


class WorkerStillBusyError(Exception):
    pass


class BaseWorkerHandle(abc.ABC):
    @abc.abstractmethod
    async def wait_ready(self, *, timeout: float, allow_server_uuid_change: bool = False) -> None: ...

    async def wait_idle(self, *, timeout: float) -> None:
        raise NotImplementedError(f"{type(self).__name__} cannot tell whether the worker is running a call")

    async def wait_dead(self, *, timeout: float) -> None:
        deadline = time.monotonic() + timeout
        while True:
            if await self.probe_is_dead():
                return
            if time.monotonic() >= deadline:
                logger.error("Timed out after %.0fs waiting for %r to die; proceeding anyway", timeout, self)
                return
            await asyncio.sleep(_WAIT_DEAD_PROBE_INTERVAL_SECONDS)

    @abc.abstractmethod
    async def probe_is_dead(self) -> bool: ...


class LazyWorkerHandle(BaseWorkerHandle):
    """Resolve the real handle on first use rather than at construction."""

    def __init__(self, resolve: Callable[[], BaseWorkerHandle]) -> None:
        self._resolve = resolve
        self._resolved: BaseWorkerHandle | None = None

    async def wait_ready(self, *, timeout: float) -> None:
        await self._handle.wait_ready(timeout=timeout)

    async def probe_is_dead(self) -> bool:
        return await self._handle.probe_is_dead()

    def __getattr__(self, name: str):
        # a worker resolves its peers from its own constructor, which runs before any cell has been
        # given its ports, so an address read there is the empty one a worker holds before it is served
        return getattr(self._handle, name)

    @property
    def _handle(self) -> BaseWorkerHandle:
        if self._resolved is None:
            self._resolved = self._resolve()
        return self._resolved
