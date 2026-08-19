from __future__ import annotations

import asyncio


from miles.utils.ft_utils.api_server.handles import _CellHandler


class _RecordingOperations:
    def __init__(self) -> None:
        self.suspended: list[str] = []

    async def suspend(self, *, cell_id: str) -> None:
        self.suspended.append(cell_id)


class _BusyGate:
    """Stands in for the controller: it answers only once whoever holds its lock is done."""

    def __init__(self) -> None:
        self.released = asyncio.Event()
        self.stopped: list[str] = []

    async def stop_cell_between_weight_updates(self, cell_id: str) -> None:
        await self.released.wait()
        self.stopped.append(cell_id)


def _handler(*, operations, suspend_gate) -> _CellHandler:
    return _CellHandler(
        cell_type="rollout",
        operations=operations,
        controllers=[],
        pool_ids=["inference-engine-0"],
        suspend_gate=suspend_gate,
    )


async def test_a_gated_suspend_waits_for_the_gate_instead_of_killing_the_cell():
    """A suspend arriving mid weight update must not reach the worker manager until the update ends."""
    operations = _RecordingOperations()
    gate = _BusyGate()
    handler = _handler(operations=operations, suspend_gate=gate)

    suspending = asyncio.create_task(handler.suspend("inference-engine-0-0-2"))
    await asyncio.sleep(0)
    assert not gate.stopped
    assert operations.suspended == []

    gate.released.set()
    await suspending
    assert gate.stopped == ["inference-engine-0-0-2"]
    assert operations.suspended == []


async def test_an_ungated_suspend_still_goes_straight_to_the_operations():
    """Backends without a gate keep the direct path, so nothing waits on a lock they do not hold."""
    operations = _RecordingOperations()
    handler = _handler(operations=operations, suspend_gate=None)

    await handler.suspend("inference-engine-0-0-2")

    assert operations.suspended == ["inference-engine-0-0-2"]
