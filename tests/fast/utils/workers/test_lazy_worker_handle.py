from tests.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=30, suite="stage-a-cpu", labels=[])

import pytest

from miles.utils.workers.worker_handle import BaseWorkerHandle, LazyWorkerHandle


class _Recording(BaseWorkerHandle):
    def __init__(self) -> None:
        self.calls: list[str] = []

    async def wait_ready(self, *, timeout: float) -> None:
        self.calls.append(f"wait_ready:{timeout}")

    async def probe_is_dead(self) -> bool:
        self.calls.append("probe_is_dead")
        return False

    async def start_update_weights(self, *, model_id: str) -> str:
        self.calls.append(f"start_update_weights:{model_id}")
        return "info"


class TestAHandleIsResolvedWhenItIsUsed:
    def test_building_it_asks_the_provider_for_nothing(self):
        """A worker resolves its peers from its own constructor, which runs before any cell has been
        given its ports, so resolving there reads the empty address a worker holds before it is served."""
        resolutions: list[int] = []

        LazyWorkerHandle(lambda: resolutions.append(1) or _Recording())

        assert resolutions == []

    @pytest.mark.asyncio
    async def test_the_first_call_resolves_it(self):
        """Every later attempt is answered by the same handle the provider names once ports exist."""
        target = _Recording()
        handle = LazyWorkerHandle(lambda: target)

        assert await handle.start_update_weights(model_id="actor") == "info"

        assert target.calls == ["start_update_weights:actor"]

    @pytest.mark.asyncio
    async def test_it_is_resolved_once_however_often_it_is_called(self):
        """Re-resolving would hand out a second handle to a worker that is meant to be addressed once."""
        resolutions: list[_Recording] = []

        def _resolve() -> _Recording:
            resolutions.append(_Recording())
            return resolutions[-1]

        handle = LazyWorkerHandle(_resolve)
        await handle.probe_is_dead()
        await handle.wait_ready(timeout=1.0)
        await handle.start_update_weights(model_id="actor")

        assert len(resolutions) == 1

    @pytest.mark.asyncio
    async def test_the_lifecycle_calls_reach_the_real_handle_too(self):
        """These are declared on the base class, so forwarding them is not the fallback path's job."""
        target = _Recording()
        handle = LazyWorkerHandle(lambda: target)

        await handle.wait_ready(timeout=2.0)
        assert await handle.probe_is_dead() is False

        assert target.calls == ["wait_ready:2.0", "probe_is_dead"]
