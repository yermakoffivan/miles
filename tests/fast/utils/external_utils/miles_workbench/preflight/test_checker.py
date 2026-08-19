from __future__ import annotations

import subprocess
import threading
from typing import Any

import pytest

from miles.utils.external_utils.miles_workbench.preflight import checker as checker_module

NAMESPACE = "rl"

_RULES = {
    "pods": ("create", "delete", "get", "list", "patch", "update", "watch"),
    "services": ("create", "delete", "get"),
    "pods/exec": ("create",),
}


class _RecordingKubectl:
    def __init__(self, *, meeting: int) -> None:
        self.calls: list[list[str]] = []
        self._lock = threading.Lock()
        self._meeting = threading.Barrier(meeting, timeout=30.0)

    def __call__(self, *args: str) -> subprocess.CompletedProcess[str]:
        with self._lock:
            self.calls.append(list(args))
        self._meeting.wait()
        return subprocess.CompletedProcess(args=list(args), returncode=0, stdout="yes\n", stderr="")


def _checker(monkeypatch: pytest.MonkeyPatch, kubectl: Any) -> checker_module.Checker:
    checker = checker_module.Checker(NAMESPACE)
    monkeypatch.setattr(checker, "kubectl", kubectl)
    return checker


class TestRulesAreAskedConcurrently:
    def test_a_rule_set_is_asked_in_one_parallel_pass(self, monkeypatch):
        """A plan runs to several hundred rules, and asking them in turn makes the wait a round trip
        each; the barrier only opens once that many callers are in flight together."""
        kubectl = _RecordingKubectl(meeting=sum(len(verbs) for verbs in _RULES.values()))

        assert _checker(monkeypatch, kubectl).denied_rules(_RULES) == []

    def test_every_rule_is_still_asked_of_the_cluster(self, monkeypatch):
        """Answering in parallel must not drop or merge a rule: each verb needs its own answer."""
        kubectl = _RecordingKubectl(meeting=sum(len(verbs) for verbs in _RULES.values()))

        _checker(monkeypatch, kubectl).denied_rules(_RULES)

        asked = {(call[2], call[3]) for call in kubectl.calls}
        assert asked == {(verb, resource.partition("/")[0]) for resource, verbs in _RULES.items() for verb in verbs}

    def test_a_subresource_is_asked_for_by_name(self, monkeypatch):
        """kubectl reads pods/exec as the pods resource unless the subresource is named separately."""
        kubectl = _RecordingKubectl(meeting=1)

        _checker(monkeypatch, kubectl).denied_rules({"pods/exec": ("create",)})

        assert "--subresource=exec" in kubectl.calls[0]

    def test_an_answer_already_held_is_not_asked_again(self, monkeypatch):
        """The plan repeats rules across its checks, and re-asking them would undo the parallel pass."""
        kubectl = _RecordingKubectl(meeting=1)
        checker = _checker(monkeypatch, kubectl)

        checker.denied_rules({"pods": ("get",)})
        checker.denied_rules({"pods": ("get",)})

        assert len(kubectl.calls) == 1
