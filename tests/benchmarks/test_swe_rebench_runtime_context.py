import threading
import unittest

from src.benchmarks.swe_rebench.runtime import (
    CommandResult,
    RepositoryRuntime,
    RepositoryRuntimeClosedError,
    RepositoryRuntimeContextError,
    bind_repository_runtime,
    get_repository_runtime,
)


class ContextRuntime(RepositoryRuntime):
    def list_files(self, path=".", **kwargs):
        self.ensure_open()
        return path

    def read_file(self, path, **kwargs):
        self.ensure_open()
        return path

    def search_code(self, query, **kwargs):
        self.ensure_open()
        return query

    def apply_patch(self, patch, **kwargs):
        self.ensure_open()
        return patch

    def run_command(self, command, **kwargs):
        self.ensure_open()
        return CommandResult(command, 0, "", "", 0.0)

    def get_diff(self, path=".", **kwargs):
        self.ensure_open()
        return path

    def get_patch(self, **kwargs):
        self.ensure_open()
        return "patch"


class RepositoryRuntimeContextTests(unittest.TestCase):
    def make_runtime(self, instance_id: str) -> ContextRuntime:
        return ContextRuntime(instance_id, "deadbeef", "/testbed")

    def test_runtime_is_available_only_inside_binding(self) -> None:
        runtime = self.make_runtime("owner__repo-1")
        with self.assertRaises(RepositoryRuntimeContextError):
            get_repository_runtime()

        with bind_repository_runtime(runtime):
            self.assertIs(get_repository_runtime(), runtime)

        with self.assertRaises(RepositoryRuntimeContextError):
            get_repository_runtime()

    def test_nested_binding_restores_outer_runtime(self) -> None:
        outer = self.make_runtime("owner__outer-1")
        inner = self.make_runtime("owner__inner-1")

        with bind_repository_runtime(outer):
            with bind_repository_runtime(inner):
                self.assertIs(get_repository_runtime(), inner)
            self.assertIs(get_repository_runtime(), outer)

    def test_bindings_are_isolated_between_threads(self) -> None:
        runtimes = [
            self.make_runtime("owner__first-1"),
            self.make_runtime("owner__second-1"),
        ]
        barrier = threading.Barrier(2)
        observed: list[str] = []

        def observe(runtime: ContextRuntime) -> None:
            with bind_repository_runtime(runtime):
                barrier.wait()
                observed.append(get_repository_runtime().instance_id)

        threads = [threading.Thread(target=observe, args=(item,)) for item in runtimes]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()

        self.assertCountEqual(observed, [item.instance_id for item in runtimes])

    def test_binding_does_not_own_runtime_lifecycle(self) -> None:
        runtime = self.make_runtime("owner__repo-1")
        with bind_repository_runtime(runtime):
            pass
        self.assertFalse(runtime.is_closed)

    def test_closed_runtime_cannot_be_bound_or_retrieved(self) -> None:
        runtime = self.make_runtime("owner__repo-1")
        runtime.close()
        with self.assertRaises(RepositoryRuntimeClosedError):
            with bind_repository_runtime(runtime):
                pass

        open_runtime = self.make_runtime("owner__repo-2")
        with bind_repository_runtime(open_runtime):
            open_runtime.close()
            with self.assertRaises(RepositoryRuntimeClosedError):
                get_repository_runtime()

    def test_binding_rejects_non_runtime_values(self) -> None:
        with self.assertRaises(TypeError):
            with bind_repository_runtime(object()):
                pass


if __name__ == "__main__":
    unittest.main()
