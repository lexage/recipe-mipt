"""Dataset loading and deterministic selection for SWE-rebench tasks."""

import json
from collections.abc import Iterable, Iterator, Sequence
from pathlib import Path
from typing import Any, Optional, overload

from .data_types import SWERebenchDataError, SWERebenchTask


class DatasetSWERebench(Sequence[SWERebenchTask]):
    """An ordered, validated collection of unique SWE-rebench tasks."""

    def __init__(self, records: Iterable[dict[str, Any] | SWERebenchTask]):
        items: list[SWERebenchTask] = []
        by_id: dict[str, SWERebenchTask] = {}
        for position, record in enumerate(records):
            try:
                item = (
                    record
                    if isinstance(record, SWERebenchTask)
                    else SWERebenchTask.from_dict(record)
                )
            except SWERebenchDataError as error:
                raise SWERebenchDataError(
                    f"Invalid SWE-rebench record at position {position}: {error}"
                ) from error
            if item.instance_id in by_id:
                raise SWERebenchDataError(
                    f"Duplicate SWE-rebench instance_id: {item.instance_id}"
                )
            items.append(item)
            by_id[item.instance_id] = item

        self._items = tuple(items)
        self._by_id = by_id

    @classmethod
    def from_json(cls, path: str | Path, split: str = "test") -> "DatasetSWERebench":
        """Load records from a JSON array, split mapping, or JSONL file."""

        dataset_path = Path(path).expanduser()
        if not dataset_path.is_file():
            raise SWERebenchDataError(f"Dataset file does not exist: {dataset_path}")

        try:
            if dataset_path.suffix.lower() == ".jsonl":
                records = cls._read_jsonl(dataset_path)
            elif dataset_path.suffix.lower() == ".json":
                records = cls._read_json(dataset_path, split)
            else:
                raise SWERebenchDataError(
                    "Local SWE-rebench dataset must use .json or .jsonl"
                )
        except OSError as error:
            raise SWERebenchDataError(
                f"Could not read dataset file {dataset_path}: {error}"
            ) from error
        return cls(records)

    @classmethod
    def from_huggingface(
        cls,
        name: str = "SWE-rebench/SWE-rebench",
        split: str = "test",
        revision: Optional[str] = None,
        cache_dir: Optional[str | Path] = None,
    ) -> "DatasetSWERebench":
        """Load a pinned split with the optional Hugging Face datasets package."""

        if not name.strip() or not split.strip():
            raise SWERebenchDataError("Dataset name and split must be non-empty")
        try:
            from datasets import load_dataset
        except ImportError as error:
            raise SWERebenchDataError(
                "Loading from Hugging Face requires the 'datasets' package"
            ) from error

        kwargs: dict[str, Any] = {"split": split}
        if revision is not None:
            kwargs["revision"] = revision
        if cache_dir is not None:
            kwargs["cache_dir"] = str(cache_dir)
        try:
            records = load_dataset(name, **kwargs)
        except Exception as error:
            raise SWERebenchDataError(
                f"Could not load Hugging Face dataset '{name}' split '{split}': {error}"
            ) from error
        return cls(records)

    @classmethod
    def load(
        cls,
        name_or_path: str | Path = "SWE-rebench/SWE-rebench",
        split: str = "test",
        revision: Optional[str] = None,
        cache_dir: Optional[str | Path] = None,
    ) -> "DatasetSWERebench":
        """Load a local file when it exists, otherwise use a Hub dataset name."""

        candidate = Path(name_or_path).expanduser()
        if candidate.is_file():
            return cls.from_json(candidate, split=split)
        if candidate.suffix.lower() in {".json", ".jsonl"}:
            raise SWERebenchDataError(f"Dataset file does not exist: {candidate}")
        return cls.from_huggingface(
            str(name_or_path), split=split, revision=revision, cache_dir=cache_dir
        )

    @staticmethod
    def read_instance_ids(path: str | Path) -> list[str]:
        """Read non-empty, non-comment instance IDs from a text file."""

        ids_path = Path(path).expanduser()
        if not ids_path.is_file():
            raise SWERebenchDataError(f"Instance ID file does not exist: {ids_path}")
        instance_ids = [
            line.strip()
            for line in ids_path.read_text(encoding="utf-8").splitlines()
            if line.strip() and not line.lstrip().startswith("#")
        ]
        if len(instance_ids) != len(set(instance_ids)):
            raise SWERebenchDataError("Instance ID file contains duplicate values")
        return instance_ids

    def select(
        self,
        instance_ids: Optional[Iterable[str]] = None,
        start: int = 0,
        stop: Optional[int] = None,
        limit: Optional[int] = None,
    ) -> "DatasetSWERebench":
        """Return a deterministic subset while preserving dataset order."""

        if start < 0 or (stop is not None and stop < start):
            raise SWERebenchDataError("Require 0 <= start <= stop")
        if limit is not None and limit < 0:
            raise SWERebenchDataError("limit must be non-negative")

        items: Iterable[SWERebenchTask] = self._items
        if instance_ids is not None:
            requested = list(instance_ids)
            if len(requested) != len(set(requested)):
                raise SWERebenchDataError("Requested instance IDs contain duplicates")
            missing = sorted(set(requested) - self._by_id.keys())
            if missing:
                raise SWERebenchDataError(
                    "Instance IDs not found in dataset: " + ", ".join(missing)
                )
            requested_set = set(requested)
            items = (item for item in items if item.instance_id in requested_set)

        selected = list(items)[start:stop]
        if limit is not None:
            selected = selected[:limit]
        return type(self)(selected)

    @staticmethod
    def _read_jsonl(path: Path) -> Iterator[dict[str, Any]]:
        with path.open(encoding="utf-8") as dataset_file:
            for line_number, line in enumerate(dataset_file, start=1):
                if not line.strip():
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError as error:
                    raise SWERebenchDataError(
                        f"Invalid JSON on line {line_number} of {path}: {error.msg}"
                    ) from error
                if not isinstance(record, dict):
                    raise SWERebenchDataError(
                        f"JSONL record on line {line_number} must be an object"
                    )
                yield record

    @staticmethod
    def _read_json(path: Path, split: str) -> list[dict[str, Any]]:
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as error:
            raise SWERebenchDataError(f"Invalid JSON in {path}: {error.msg}") from error
        if isinstance(payload, dict):
            if split not in payload:
                raise SWERebenchDataError(
                    f"Dataset JSON has no requested split '{split}'"
                )
            payload = payload[split]
        if not isinstance(payload, list) or not all(
            isinstance(record, dict) for record in payload
        ):
            raise SWERebenchDataError(
                "Dataset JSON must be an array of objects or a mapping of splits"
            )
        return payload

    def __len__(self) -> int:
        return len(self._items)

    @overload
    def __getitem__(self, index: int) -> SWERebenchTask: ...

    @overload
    def __getitem__(self, index: slice) -> Sequence[SWERebenchTask]: ...

    @overload
    def __getitem__(self, index: str) -> SWERebenchTask: ...

    def __getitem__(
        self, index: int | slice | str
    ) -> SWERebenchTask | Sequence[SWERebenchTask]:
        if isinstance(index, str):
            try:
                return self._by_id[index]
            except KeyError as error:
                raise KeyError(f"Unknown SWE-rebench instance_id: {index}") from error
        return self._items[index]

    def __iter__(self) -> Iterator[SWERebenchTask]:
        return iter(self._items)
