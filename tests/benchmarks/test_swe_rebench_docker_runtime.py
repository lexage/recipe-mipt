import unittest
from types import SimpleNamespace
from unittest.mock import Mock

from src.benchmarks.swe_rebench import InstanceImage
from src.benchmarks.swe_rebench.runtime import (
    DockerRepositoryRuntime,
    RepositoryRuntimeError,
)


class FakeContainer:
    def __init__(self, base_commit="deadbeef", status=b""):
        self.responses = [
            SimpleNamespace(exit_code=0, output=(b"", b"")),
            SimpleNamespace(exit_code=0, output=(base_commit.encode() + b"\n", b"")),
            SimpleNamespace(exit_code=0, output=(status, b"")),
        ]
        self.exec_calls = []
        self.stop = Mock()
        self.remove = Mock()

    def exec_run(self, command, **kwargs):
        self.exec_calls.append((command, kwargs))
        return self.responses.pop(0)


class FakeClient:
    def __init__(self, container, image_exists=True):
        self.container = container
        self.images = SimpleNamespace(get=Mock(), pull=Mock())
        if not image_exists:
            self.images.get.side_effect = RuntimeError("missing")
        self.containers = SimpleNamespace(run=Mock(return_value=container))


class DockerRepositoryRuntimeTests(unittest.TestCase):
    def setUp(self):
        self.image = InstanceImage(
            "swebench/sweb.eval.x86_64.owner_1776_repo-1:latest",
            "linux/x86_64",
        )

    def create(self, client, **kwargs):
        return DockerRepositoryRuntime.create(
            client=client,
            image=self.image,
            instance_id="owner__repo-1",
            base_commit="deadbeef",
            run_id="test-run",
            **kwargs,
        )

    def test_starts_and_validates_dedicated_inference_container(self):
        container = FakeContainer()
        client = FakeClient(container)
        runtime = self.create(client, memory_limit="4g", nano_cpus=2_000_000_000)

        options = client.containers.run.call_args.kwargs
        self.assertEqual(options["image"], self.image.name)
        self.assertEqual(options["working_dir"], "/testbed")
        self.assertEqual(options["platform"], "linux/x86_64")
        self.assertEqual(options["command"], ["sleep", "infinity"])
        self.assertTrue(options["detach"])
        self.assertTrue(options["network_disabled"])
        self.assertEqual(options["mem_limit"], "4g")
        self.assertEqual(options["nano_cpus"], 2_000_000_000)
        self.assertEqual(options["labels"]["swe-rebench.role"], "inference")
        self.assertEqual(len(container.exec_calls), 3)
        self.assertFalse(runtime.is_closed)

    def test_missing_image_is_pulled_by_default(self):
        client = FakeClient(FakeContainer(), image_exists=False)
        self.create(client)
        client.images.pull.assert_called_once_with(self.image.name)

    def test_pull_policies_are_enforced(self):
        always = FakeClient(FakeContainer())
        self.create(always, pull_policy="always")
        always.images.pull.assert_called_once_with(self.image.name)
        always.images.get.assert_not_called()

        never = FakeClient(FakeContainer(), image_exists=False)
        with self.assertRaisesRegex(RepositoryRuntimeError, "not available"):
            self.create(never, pull_policy="never")
        never.containers.run.assert_not_called()

    def test_checkout_commit_mismatch_removes_container(self):
        container = FakeContainer(base_commit="cafebabe")
        client = FakeClient(container)
        with self.assertRaisesRegex(RepositoryRuntimeError, "expected deadbeef"):
            self.create(client)
        container.stop.assert_called_once()
        container.remove.assert_called_once_with(force=True)

    def test_dirty_checkout_removes_container(self):
        container = FakeContainer(status=b" M changed.py\n")
        client = FakeClient(container)
        with self.assertRaisesRegex(RepositoryRuntimeError, "not clean"):
            self.create(client)
        container.remove.assert_called_once_with(force=True)

    def test_close_stops_and_removes_container_idempotently(self):
        container = FakeContainer()
        runtime = self.create(FakeClient(container))
        runtime.close()
        runtime.close()
        container.stop.assert_called_once_with(timeout=10)
        container.remove.assert_called_once_with(force=True)
        self.assertTrue(runtime.is_closed)

    def test_keep_container_skips_cleanup_for_debugging(self):
        container = FakeContainer()
        runtime = self.create(FakeClient(container), keep_container=True)
        runtime.close()
        container.stop.assert_not_called()
        container.remove.assert_not_called()
        self.assertTrue(runtime.is_closed)

    def test_executes_command_in_testbed_and_normalizes_output(self):
        container = FakeContainer()
        runtime = self.create(FakeClient(container))
        container.responses.append(
            SimpleNamespace(exit_code=3, output=(b"stdout\n", b"stderr\n"))
        )
        result = runtime.run_command("pytest -q")

        command, options = container.exec_calls[-1]
        self.assertEqual(command, ["/bin/sh", "-lc", "pytest -q"])
        self.assertEqual(options["workdir"], "/testbed")
        self.assertEqual(result.exit_code, 3)
        self.assertEqual(result.stdout, "stdout\n")
        self.assertEqual(result.stderr, "stderr\n")

    def test_invalid_configuration_is_rejected(self):
        client = FakeClient(FakeContainer())
        with self.assertRaisesRegex(RepositoryRuntimeError, "pull policy"):
            self.create(client, pull_policy="sometimes")
        with self.assertRaisesRegex(RepositoryRuntimeError, "run_id"):
            DockerRepositoryRuntime.create(
                client=client,
                image=self.image,
                instance_id="owner__repo-1",
                base_commit="deadbeef",
                run_id="",
            )


if __name__ == "__main__":
    unittest.main()
