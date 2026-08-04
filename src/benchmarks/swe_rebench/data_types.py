"""Validated data types used by SWE-rebench prediction generation."""

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

        optional: dict[str, Optional[str]] = {}
        for field in ("hints_text", "version"):
            value = data.get(field)
            if value is not None and not isinstance(value, str):
                raise SWERebenchDataError(
                    f"SWE-rebench field '{field}' must be a string or null"
                )
            optional[field] = value

        # Evaluator-only fields (patch, test_patch and test expectations) are
        # deliberately not retained, preventing accidental reference leakage.
        return cls(**values, **optional)
