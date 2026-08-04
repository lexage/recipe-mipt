import unittest

from src.benchmarks.swe_rebench.runtime import (
    CommandResult,
    RepositoryRuntime,
    RepositoryRuntimeClosedError,
)


class DummyRuntime(RepositoryRuntime):
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
        return CommandResult(command, 0, "ok", "", 0.1)

    def get_diff(self, path=".", **kwargs):
        self.ensure_open()
        return path


class RepositoryRuntimeTests(unittest.TestCase):
    def test_runtime_metadata_and_operations_are_backend_neutral(self) -> None:
        runtime = DummyRuntime("owner__repo-1", "deadbeef", "/testbed")

        self.assertEqual(runtime.instance_id, "owner__repo-1")
        self.assertEqual(runtime.base_commit, "deadbeef")
        self.assertEqual(runtime.workdir, "/testbed")
        self.assertEqual(runtime.list_files("src"), "src")
        self.assertEqual(runtime.read_file("src/module.py"), "src/module.py")
        self.assertEqual(runtime.search_code("symbol"), "symbol")
        self.assertEqual(runtime.apply_patch("diff"), "diff")
        self.assertEqual(runtime.run_command("pytest").exit_code, 0)
        self.assertEqual(runtime.get_diff(), ".")

    def test_context_manager_closes_runtime_even_after_error(self) -> None:
        runtime = DummyRuntime("owner__repo-1", "deadbeef", "/testbed")

        with self.assertRaisesRegex(RuntimeError, "failure"):
            with runtime:
                raise RuntimeError("failure")

        self.assertTrue(runtime.is_closed)
        with self.assertRaises(RepositoryRuntimeClosedError):
            runtime.list_files()

    def test_close_is_idempotent(self) -> None:
        runtime = DummyRuntime("owner__repo-1", "deadbeef", "/testbed")
        runtime.close()
        runtime.close()
        self.assertTrue(runtime.is_closed)

    def test_runtime_metadata_must_be_non_empty_strings(self) -> None:
        for values in (
            ("", "commit", "/testbed"),
            ("instance", "", "/testbed"),
            ("instance", "commit", ""),
        ):
            with self.subTest(values=values):
                with self.assertRaises(ValueError):
                    DummyRuntime(*values)

    def test_abstract_runtime_cannot_be_instantiated(self) -> None:
        with self.assertRaises(TypeError):
            RepositoryRuntime("instance", "commit", "/testbed")


class CommandResultTests(unittest.TestCase):
    def test_success_and_timeout_results(self) -> None:
        success = CommandResult("pytest", 0, "passed", "", 1.5)
        timeout = CommandResult("pytest", None, "partial", "", 10.0, timed_out=True)

        self.assertEqual(success.exit_code, 0)
        self.assertFalse(success.timed_out)
        self.assertIsNone(timeout.exit_code)
        self.assertTrue(timeout.timed_out)

    def test_rejects_inconsistent_results(self) -> None:
        invalid_arguments = (
            ("", 0, "", "", 0.0, False),
            ("command", "0", "", "", 0.0, False),
            ("command", 0, b"bytes", "", 0.0, False),
            ("command", 0, "", "", -1.0, False),
            ("command", 1, "", "", 1.0, True),
        )
        for arguments in invalid_arguments:
            with self.subTest(arguments=arguments):
                with self.assertRaises(ValueError):
                    CommandResult(*arguments)


if __name__ == "__main__":
    unittest.main()
