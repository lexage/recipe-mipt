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
            SimpleNamespace(exit_code=0, output=(b"", b"")),
            SimpleNamespace(exit_code=0, output=(base_commit.encode() + b"\n", b"")),
            SimpleNamespace(exit_code=0, output=(status, b"")),
            SimpleNamespace(
                exit_code=0,
                output=(b"/opt/miniconda3/envs/testbed/bin/python\n", b""),
            ),
            SimpleNamespace(exit_code=0, output=(b"Python 3.12\n", b"")),
            SimpleNamespace(exit_code=0, output=(b"", b"")),
            SimpleNamespace(
                exit_code=0,
                output=(b"/opt/miniconda3/envs/testbed\n", b""),
            ),
        ]
        self.exec_calls = []
        self.put_archive = Mock(return_value=True)
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

    def direct_runtime(self, container):
        return DockerRepositoryRuntime(
            client=FakeClient(container),
            container=container,
            image=self.image,
            instance_id="owner__repo-1",
            base_commit="deadbeef",
        )

    def test_starts_and_validates_dedicated_inference_container(self):
        container = FakeContainer()
        client = FakeClient(container)
        runtime = self.create(client, memory_limit="4g", nano_cpus=2_000_000_000)

        options = client.containers.run.call_args.kwargs
        self.assertEqual(options["image"], self.image.name)
        self.assertEqual(options["working_dir"], "/testbed")
        self.assertEqual(options["platform"], "linux/x86_64")
        self.assertEqual(options["command"], ["tail", "-f", "/dev/null"])
        self.assertEqual(options["user"], "root")
        self.assertNotIn("BASH_ENV", options["environment"])
        self.assertTrue(options["detach"])
        self.assertTrue(options["network_disabled"])
        self.assertEqual(options["mem_limit"], "4g")
        self.assertEqual(options["nano_cpus"], 2_000_000_000)
        self.assertEqual(options["labels"]["swe-rebench.role"], "inference")
        self.assertEqual(len(container.exec_calls), 8)
        self.assertFalse(runtime.is_closed)

    def test_rejects_inactive_testbed_environment(self):
        container = FakeContainer()
        container.responses[6] = SimpleNamespace(
            exit_code=1, output=(b"", b"wrong environment\n")
        )
        client = FakeClient(container)

        with self.assertRaisesRegex(RepositoryRuntimeError, "not active"):
            self.create(client)

        container.stop.assert_called_once()
        container.remove.assert_called_once_with(force=True)

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
        self.assertEqual(
            command[:6],
            [
                "timeout",
                "--signal=KILL",
                "120s",
                "/bin/bash",
                "-c",
            ],
        )
        wrapped_command = command[6]
        self.assertIn(
            "source /opt/miniconda3/etc/profile.d/conda.sh || exit 127",
            wrapped_command,
        )
        self.assertIn("conda activate testbed || exit 127", wrapped_command)
        self.assertTrue(wrapped_command.endswith("\npytest -q"))
        self.assertEqual(options["workdir"], "/testbed")
        self.assertEqual(options["environment"]["PAGER"], "cat")
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

    def test_docker_file_listing_reading_and_search(self):
        listing_container = FakeContainer()
        listing_container.responses = [
            SimpleNamespace(exit_code=0, output=(b"", b"")),
            SimpleNamespace(
                exit_code=0,
                output=(b"calculator.py\tf\nnested\td\nextra.py\tf\n", b""),
            ),
        ]
        listing = self.direct_runtime(listing_container).list_files(
            "src", max_entries=2
        )
        self.assertIn("calculator.py", listing)
        self.assertIn("nested/", listing)
        self.assertIn("output truncated", listing)

        read_container = FakeContainer()
        read_container.responses = [
            SimpleNamespace(exit_code=0, output=(b"", b"")),
            SimpleNamespace(exit_code=0, output=(b"42\n2\n", b"")),
            SimpleNamespace(exit_code=0, output=(b"", b"")),
            SimpleNamespace(exit_code=0, output=(b"first\nsecond\n", b"")),
        ]
        content = self.direct_runtime(read_container).read_file("src/calculator.py")
        self.assertIn("1 | first", content)
        self.assertIn("2 | second", content)

        search_container = FakeContainer()
        search_container.responses = [
            SimpleNamespace(exit_code=0, output=(b"", b"")),
            SimpleNamespace(
                exit_code=0,
                output=(b"src/calculator.py:2:5:left - right\n", b""),
            ),
        ]
        matches = self.direct_runtime(search_container).search_code("left - right")
        self.assertIn("src/calculator.py:2", matches)

    def test_docker_patch_and_diff_operations(self):
        patch = """diff --git a/src/calculator.py b/src/calculator.py
--- a/src/calculator.py
+++ b/src/calculator.py
@@ -1 +1 @@
-old
+new
"""
        patch_container = FakeContainer()
        patch_container.responses = [
            SimpleNamespace(exit_code=0, output=(b"", b"")),
            SimpleNamespace(exit_code=0, output=(b"", b"")),
            SimpleNamespace(exit_code=0, output=(b"", b"")),
            SimpleNamespace(exit_code=0, output=(b"", b"")),
            SimpleNamespace(exit_code=0, output=(b"", b"")),
        ]
        applied = self.direct_runtime(patch_container).apply_patch(patch)
        self.assertIn("Patch applied successfully", applied)
        patch_container.put_archive.assert_called_once()
        self.assertEqual(patch_container.put_archive.call_args.args[0], "/tmp")

        diff_container = FakeContainer()
        diff_container.responses = [
            SimpleNamespace(exit_code=0, output=(b"", b"")),
            SimpleNamespace(exit_code=0, output=(b"", b"")),
            SimpleNamespace(exit_code=0, output=(b" M src/calculator.py\n", b"")),
            SimpleNamespace(
                exit_code=0, output=(b"diff --git a/src/calculator.py\n", b"")
            ),
        ]
        diff = self.direct_runtime(diff_container).get_diff()
        self.assertIn("M src/calculator.py", diff)
        self.assertIn("diff --git", diff)

        patch_container = FakeContainer()
        patch_container.responses = [
            SimpleNamespace(exit_code=0, output=(b"", b"")),
            SimpleNamespace(
                exit_code=0, output=(b"diff --git a/src/calculator.py\n", b"")
            ),
            SimpleNamespace(exit_code=0, output=(b"", b"")),
        ]
        model_patch = self.direct_runtime(patch_container).get_patch()
        self.assertEqual(model_patch, "diff --git a/src/calculator.py\n")

    def test_timeout_is_enforced_inside_container(self):
        container = FakeContainer()
        container.responses = [SimpleNamespace(exit_code=124, output=(b"partial", b""))]
        result = self.direct_runtime(container).run_command("sleep 10", timeout=1)
        self.assertTrue(result.timed_out)
        self.assertIsNone(result.exit_code)
        self.assertEqual(result.stdout, "partial")

    def test_symlink_escape_is_rejected_by_container_realpath(self):
        container = FakeContainer()
        container.responses = [SimpleNamespace(exit_code=3, output=(b"", b""))]
        runtime = self.direct_runtime(container)
        with self.assertRaisesRegex(RepositoryRuntimeError, "outside"):
            runtime.read_file("external/file.py")


if __name__ == "__main__":
    unittest.main()
