from miles.utils.external_utils.command_utils.base_backend import BaseCommandBackend
from miles.utils.external_utils.command_utils.ray_backend.backend import RayCommandBackend


def patch_helper(monkeypatch, name: str, replacement, *, backend_class: type = BaseCommandBackend) -> None:
    assert hasattr(backend_class, name), f"no method of {backend_class.__name__} is named {name}"
    monkeypatch.setattr(backend_class, name, replacement, raising=False)


def record_commands(monkeypatch) -> list[str]:
    """Replace every command-executing backend method with a recorder and return the list it appends to."""
    commands: list[str] = []

    def fake_exec_command(self, cmd: str, capture_output: bool = False, **kwargs) -> str | None:
        commands.append(cmd)
        return "0" if capture_output else None

    def fake_exec_command_multi_node(
        self,
        cmd: str,
        capture_output: bool = False,
        num_nodes: int | None = None,
        num_gpus_per_node: int | None = None,
    ) -> list[str | None]:
        commands.append(f"[multi_node num_nodes={num_nodes}] {cmd}")
        return ["0"]

    # the ray backend overrides the gpu and multi-node forms, and a patch on the base
    # class never reaches an override
    patch_helper(monkeypatch, "exec_command_cpu", fake_exec_command)
    patch_helper(monkeypatch, "exec_command_gpu", fake_exec_command, backend_class=RayCommandBackend)
    patch_helper(monkeypatch, "exec_command_multi_node", fake_exec_command_multi_node, backend_class=RayCommandBackend)

    return commands
