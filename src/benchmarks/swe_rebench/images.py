"""Resolve official per-instance container images for SWE-rebench inference."""

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any, Literal, Optional

from .data_types import SWERebenchTask


class InstanceImageError(ValueError):
    """Raised when an instance image cannot be resolved safely."""


@dataclass(frozen=True)
class InstanceImage:
    """Pinned container image coordinates for one inference instance."""

    name: str
    platform: str
    workdir: str = "/testbed"
    source: Literal["test_spec", "override", "convention"] = "convention"

    def __post_init__(self) -> None:
        for field in ("name", "platform", "workdir"):
            value = getattr(self, field)
            if not isinstance(value, str) or not value.strip():
                raise InstanceImageError(f"Image {field} must be a non-empty string")
        if not self.workdir.startswith("/"):
            raise InstanceImageError("Image workdir must be an absolute POSIX path")


class InstanceImageResolver:
    """Resolve images using a pinned harness TestSpec or its naming convention.

    A production runner should provide ``test_spec_factory`` from its pinned
    SWE-rebench fork. The convention fallback mirrors the SWE-bench harness and
    is kept configurable for local smoke tests and prebuilt image registries.
    """

    ARCH_TO_PLATFORM = {
        "x86_64": "linux/x86_64",
        "arm64": "linux/arm64/v8",
    }

    def __init__(
        self,
        *,
        namespace: Optional[str] = "swebench",
        architecture: str = "x86_64",
        image_tag: str = "latest",
        workdir: str = "/testbed",
        overrides: Optional[Mapping[str, str]] = None,
        test_spec_factory: Optional[Callable[[SWERebenchTask], Any]] = None,
    ) -> None:
        if namespace is not None and not isinstance(namespace, str):
            raise InstanceImageError("namespace must be a string or None")
        if namespace is not None and not namespace.strip():
            raise InstanceImageError("namespace must not be empty")
        if architecture not in self.ARCH_TO_PLATFORM:
            raise InstanceImageError("Unsupported image architecture: " + architecture)
        if not isinstance(image_tag, str) or not image_tag.strip():
            raise InstanceImageError("image_tag must be a non-empty string")
        if not isinstance(workdir, str) or not workdir.startswith("/"):
            raise InstanceImageError("workdir must be an absolute POSIX path")

        self.namespace = namespace
        self.architecture = architecture
        self.image_tag = image_tag
        self.workdir = workdir
        self.overrides = dict(overrides or {})
        self.test_spec_factory = test_spec_factory

        for instance_id, image_name in self.overrides.items():
            if not isinstance(instance_id, str) or not instance_id.strip():
                raise InstanceImageError("Override instance IDs must be non-empty")
            if not isinstance(image_name, str) or not image_name.strip():
                raise InstanceImageError(
                    f"Image override for {instance_id!r} must be non-empty"
                )

    def resolve(self, task: SWERebenchTask) -> InstanceImage:
        """Return the image selected for a validated SWE-rebench task."""

        if not isinstance(task, SWERebenchTask):
            raise TypeError("task must be a SWERebenchTask")

        if task.instance_id in self.overrides:
            return InstanceImage(
                name=self.overrides[task.instance_id],
                platform=self.ARCH_TO_PLATFORM[self.architecture],
                workdir=self.workdir,
                source="override",
            )

        if self.test_spec_factory is not None:
            return self._resolve_from_test_spec(self.test_spec_factory(task))

        key = (
            f"sweb.eval.{self.architecture}."
            f"{task.instance_id.lower()}:{self.image_tag}"
        )
        if self.namespace is not None:
            key = f"{self.namespace}/{key}".replace("__", "_1776_")
        return InstanceImage(
            name=key,
            platform=self.ARCH_TO_PLATFORM[self.architecture],
            workdir=self.workdir,
            source="convention",
        )

    def _resolve_from_test_spec(self, test_spec: Any) -> InstanceImage:
        """Read only the public image coordinates exposed by a harness TestSpec."""

        try:
            name = test_spec.instance_image_key
            platform = test_spec.platform
        except (AttributeError, TypeError) as error:
            raise InstanceImageError(
                "Pinned test_spec_factory must return an object exposing "
                "instance_image_key and platform"
            ) from error
        return InstanceImage(
            name=name,
            platform=platform,
            workdir=self.workdir,
            source="test_spec",
        )
