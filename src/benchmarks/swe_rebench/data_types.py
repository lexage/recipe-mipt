"""Validated data types used by SWE-rebench prediction generation."""

import json
from dataclasses import dataclass
from typing import Any, Mapping, Optional


class SWERebenchDataError(ValueError):
    """Raised when an input record does not satisfy the generator contract."""


@dataclass(frozen=True)
class SWERebenchTask:
    """The non-secret subset of a SWE-rebench task exposed to the generator."""

    instance_id: str
    repo: str
    base_commit: str
    problem_statement: str
    hints_text: Optional[str] = None
    version: Optional[str] = None
    image_name: Optional[str] = None
    docker_image: Optional[str] = None
    docker_run_args: Optional[dict[str, Any]] = None

    @property
    def instance_image(self) -> str:
        image = self.image_name or self.docker_image
        if not image:
            raise SWERebenchDataError(
                f"Instance {self.instance_id} has no image_name or docker_image"
            )
        return image

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "SWERebenchTask":
        if not isinstance(data, Mapping):
            raise SWERebenchDataError("SWE-rebench record must be a mapping")

        required = ("instance_id", "repo", "base_commit", "problem_statement")
        values: dict[str, str] = {}
        for field in required:
            value = data.get(field)
            if not isinstance(value, str) or not value.strip():
                raise SWERebenchDataError(
                    f"SWE-rebench field '{field}' must be a non-empty string"
                )
            values[field] = value

        optional: dict[str, Any] = {}
        for field in ("hints_text", "version", "image_name", "docker_image"):
            value = data.get(field)
            if value is not None and not isinstance(value, str):
                raise SWERebenchDataError(
                    f"SWE-rebench field '{field}' must be a string or null"
                )
            optional[field] = value

        # Only retain container-start options used by the official harness.  Never
        # forward arbitrary HostConfig values supplied by a dataset record.
        safe_run_args: dict[str, Any] = {}
        install_config = data.get("install_config")
        if isinstance(install_config, str):
            try:
                install_config = json.loads(install_config)
            except json.JSONDecodeError as error:
                raise SWERebenchDataError(
                    "install_config must contain valid JSON"
                ) from error
        if isinstance(install_config, Mapping):
            docker_specs = install_config.get("docker_specs")
            if isinstance(docker_specs, Mapping):
                run_args = docker_specs.get("run_args")
                if isinstance(run_args, Mapping):
                    cap_add = run_args.get("cap_add")
                    if cap_add is not None:
                        if not isinstance(cap_add, list) or not all(
                            isinstance(item, str) and item.strip() for item in cap_add
                        ):
                            raise SWERebenchDataError(
                                "install_config.docker_specs.run_args.cap_add "
                                "must be a list of strings"
                            )
                        safe_run_args["cap_add"] = list(cap_add)
        optional["docker_run_args"] = safe_run_args or None

        # Evaluator-only fields (patch, test_patch and test expectations) are
        # deliberately not retained, preventing accidental reference leakage.
        return cls(**values, **optional)
