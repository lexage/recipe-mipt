import unittest
from src.benchmarks.swe_rebench import (
    InstanceImage,
    InstanceImageError,
    InstanceImageResolver,
    SWERebenchTask,
)


def make_task(instance_id: str = "Owner__Repo-123") -> SWERebenchTask:
    return SWERebenchTask(
        instance_id=instance_id,
        repo="Owner/Repo",
        base_commit="deadbeef",
        problem_statement="Fix the issue.",
    )


class InstanceImageTests(unittest.TestCase):
    def test_requires_valid_coordinates(self) -> None:
        for kwargs in (
            {"name": "", "platform": "linux/amd64"},
            {"name": "image", "platform": ""},
            {"name": "image", "platform": "linux/amd64", "workdir": "testbed"},
        ):
            with self.subTest(kwargs=kwargs):
                with self.assertRaises(InstanceImageError):
                    InstanceImage(**kwargs)


class InstanceImageResolverTests(unittest.TestCase):
    def test_dataset_image_precedes_manifest(self) -> None:
        task = SWERebenchTask(
            instance_id="Owner__Repo-123",
            repo="Owner/Repo",
            base_commit="deadbeef",
            problem_statement="Fix the issue.",
            docker_image="registry/official@sha256:123",
            docker_run_args={"cap_add": ["SYS_PTRACE"]},
        )
        resolver = InstanceImageResolver(
            manifest={task.instance_id: InstanceImage("registry/old", "linux/x86_64")}
        )

        image = resolver.resolve(task)

        self.assertEqual(image.name, "registry/official@sha256:123")
        self.assertEqual(image.source, "dataset")
        self.assertEqual(image.cap_add, ("SYS_PTRACE",))

    def test_remote_convention_matches_harness_image_key(self) -> None:
        image = InstanceImageResolver(allow_convention=True).resolve(make_task())

        self.assertEqual(
            image.name,
            "swebench/sweb.eval.x86_64.owner_1776_repo-123:latest",
        )
        self.assertEqual(image.platform, "linux/x86_64")
        self.assertEqual(image.workdir, "/testbed")
        self.assertEqual(image.source, "convention")

    def test_local_image_key_keeps_double_underscore(self) -> None:
        image = InstanceImageResolver(
            namespace=None,
            architecture="arm64",
            image_tag="pinned",
            allow_convention=True,
        ).resolve(make_task())

        self.assertEqual(
            image.name,
            "sweb.eval.arm64.owner__repo-123:pinned",
        )
        self.assertEqual(image.platform, "linux/arm64/v8")

    def test_production_resolution_requires_pinned_coordinates(self) -> None:
        with self.assertRaisesRegex(InstanceImageError, "No pinned image"):
            InstanceImageResolver().resolve(make_task())

    def test_loads_exact_coordinates_from_manifest(self) -> None:
        import json
        import tempfile
        from pathlib import Path

        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "images.json"
            path.write_text(
                json.dumps(
                    {
                        "Owner__Repo-123": {
                            "instance_image_key": "registry/exact@sha256:abc",
                            "platform": "linux/amd64",
                        }
                    }
                ),
                encoding="utf-8",
            )
            image = InstanceImageResolver.from_manifest(str(path)).resolve(make_task())

        self.assertEqual(image.name, "registry/exact@sha256:abc")
        self.assertEqual(image.platform, "linux/amd64")
        self.assertEqual(image.source, "manifest")

    def test_manifest_must_cover_selected_instance(self) -> None:
        resolver = InstanceImageResolver(
            manifest={"another": InstanceImage("registry/another", "linux/amd64")}
        )
        with self.assertRaisesRegex(InstanceImageError, "No pinned image"):
            resolver.resolve(make_task())

    def test_invalid_configuration_and_task_are_rejected(self) -> None:
        for kwargs in (
            {"namespace": ""},
            {"architecture": "ppc64"},
            {"image_tag": ""},
            {"workdir": "testbed"},
        ):
            with self.subTest(kwargs=kwargs):
                with self.assertRaises(InstanceImageError):
                    InstanceImageResolver(**kwargs)

        with self.assertRaises(TypeError):
            InstanceImageResolver().resolve(object())


if __name__ == "__main__":
    unittest.main()
