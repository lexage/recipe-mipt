"""Resolve official per-instance container images for SWE-rebench inference."""

import json
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
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
    source: Literal["dataset", "manifest", "convention"] = "manifest"
    user: str = "root"
    cap_add: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        for field in ("name", "platform", "workdir", "user"):
            value = getattr(self, field)
            if not isinstance(value, str) or not value.strip():
                raise InstanceImageError(f"Image {field} must be a non-empty string")
        if not self.workdir.startswith("/"):
            raise InstanceImageError("Image workdir must be an absolute POSIX path")
        if not all(
            isinstance(capability, str) and capability.strip()
            for capability in self.cap_add
        ):
            raise InstanceImageError("Image cap_add must contain non-empty strings")


class InstanceImageResolver:
    """Resolve exact images; naming-convention fallback is explicitly opt-in."""

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
        manifest: Optional[Mapping[str, InstanceImage]] = None,
        allow_convention: bool = False,
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
        self.manifest = dict(manifest or {})
        self.allow_convention = allow_convention

        for instance_id, image in self.manifest.items():
            if not isinstance(instance_id, str) or not instance_id.strip():
                raise InstanceImageError("Manifest instance IDs must be non-empty")
            if not isinstance(image, InstanceImage):
                raise InstanceImageError(
                    f"Manifest image for {instance_id!r} must be an InstanceImage"
                )

    @classmethod
    def from_manifest(
        cls, path: str | Path, *, workdir: str = "/testbed"
    ) -> "InstanceImageResolver":
        """Load pinned image names and platforms exported from the evaluator fork."""

        try:
            raw = json.loads(Path(path).read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise InstanceImageError(
                f"Could not load image manifest {path}: {error}"
            ) from error
        if not isinstance(raw, dict) or not raw:
            raise InstanceImageError("Image manifest must be a non-empty JSON object")
        manifest: dict[str, InstanceImage] = {}
        for instance_id, coordinates in raw.items():
            if not isinstance(coordinates, dict):
                raise InstanceImageError(
                    f"Manifest entry for {instance_id!r} must be an object"
                )
            try:
                image_name = (
                    coordinates.get("name") or coordinates["instance_image_key"]
                )
                manifest[instance_id] = InstanceImage(
                    name=image_name,
                    platform=coordinates["platform"],
                    workdir=coordinates.get("workdir", workdir),
                    source="manifest",
                    user=coordinates.get("user", "root"),
                    cap_add=tuple(coordinates.get("cap_add", ())),
                )
            except (KeyError, TypeError, InstanceImageError) as error:
                raise InstanceImageError(
                    f"Invalid manifest entry for {instance_id!r}: {error}"
                ) from error
        return cls(manifest=manifest, workdir=workdir)

    def resolve(self, task: SWERebenchTask) -> InstanceImage:
        """Return the image selected for a validated SWE-rebench task."""

        if not isinstance(task, SWERebenchTask):
            raise TypeError("task must be a SWERebenchTask")

        if task.image_name or task.docker_image:
            return InstanceImage(
                name=task.instance_image,
                platform=self.ARCH_TO_PLATFORM[self.architecture],
                workdir=self.workdir,
                source="dataset",
                cap_add=tuple((task.docker_run_args or {}).get("cap_add", ())),
            )

        if task.instance_id in self.manifest:
            return self.manifest[task.instance_id]

        if not self.allow_convention:
            raise InstanceImageError(
                f"No pinned image for {task.instance_id}; provide an image manifest "
                "or explicitly enable convention fallback for local smoke tests"
            )

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
