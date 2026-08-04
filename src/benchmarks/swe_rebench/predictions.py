"""Strict, resumable JSONL output for SWE-rebench predictions."""

import fcntl
import json
import os
import threading
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Optional

from .data_types import SWERebenchDataError


@dataclass(frozen=True)
class SWERebenchPrediction:
    instance_id: str
    model_name_or_path: str
    model_patch: str

    def __post_init__(self) -> None:
        for field in ("instance_id", "model_name_or_path"):
            value = getattr(self, field)
            if not isinstance(value, str) or not value.strip():
                raise SWERebenchDataError(f"Prediction {field} must be non-empty")
        if not isinstance(self.model_patch, str):
            raise SWERebenchDataError("Prediction model_patch must be a string")


class PredictionsWriter:
    """Append predictions atomically and support restart-safe completed-ID lookup."""

    def __init__(
        self,
        path: str | Path,
        *,
        model_name_or_path: str,
        resume: bool = False,
        fsync: bool = False,
    ) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.model_name_or_path = model_name_or_path
        self.fsync = fsync
        self._lock = threading.Lock()
        if self.path.exists() and not resume and self.path.stat().st_size:
            raise SWERebenchDataError(
                f"Predictions file already exists; use resume: {self.path}"
            )
        self._completed = self._read_existing()

    @property
    def completed_ids(self) -> frozenset[str]:
        with self._lock:
            return frozenset(self._completed)

    def write(self, instance_id: str, model_patch: str) -> SWERebenchPrediction:
        prediction = SWERebenchPrediction(
            instance_id=instance_id,
            model_name_or_path=self.model_name_or_path,
            model_patch=model_patch,
        )
        encoded = (json.dumps(asdict(prediction), ensure_ascii=False) + "\n").encode()
        with self._lock:
            if instance_id in self._completed:
                raise SWERebenchDataError(f"Prediction already exists: {instance_id}")
            descriptor = os.open(
                self.path, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o644
            )
            try:
                fcntl.flock(descriptor, fcntl.LOCK_EX)
                # Recheck on disk while holding the process-safe lock.
                existing = self._read_existing_unlocked()
                if instance_id in existing:
                    raise SWERebenchDataError(
                        f"Prediction already exists: {instance_id}"
                    )
                remaining = memoryview(encoded)
                while remaining:
                    written = os.write(descriptor, remaining)
                    remaining = remaining[written:]
                if self.fsync:
                    os.fsync(descriptor)
            finally:
                fcntl.flock(descriptor, fcntl.LOCK_UN)
                os.close(descriptor)
            self._completed.add(instance_id)
        return prediction

    def _read_existing(self) -> set[str]:
        if not self.path.exists():
            return set()
        return self._read_existing_unlocked()

    def _read_existing_unlocked(self) -> set[str]:
        completed: set[str] = set()
        if not self.path.exists():
            return completed
        with self.path.open(encoding="utf-8") as predictions_file:
            for line_number, line in enumerate(predictions_file, start=1):
                if not line.strip():
                    continue
                try:
                    record: Any = json.loads(line)
                    prediction = SWERebenchPrediction(**record)
                except (json.JSONDecodeError, TypeError, SWERebenchDataError) as error:
                    raise SWERebenchDataError(
                        f"Invalid prediction on line {line_number}: {error}"
                    ) from error
                if prediction.instance_id in completed:
                    raise SWERebenchDataError(
                        f"Duplicate prediction: {prediction.instance_id}"
                    )
                completed.add(prediction.instance_id)
        return completed


class RunArtifactsWriter:
    """Write operational metadata and errors outside evaluator input JSONL."""

    def __init__(self, directory: str | Path) -> None:
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()

    def write_metadata(self, metadata: dict[str, Any]) -> None:
        path = self.directory / "run_metadata.json"
        temporary = path.with_suffix(".json.tmp")
        temporary.write_text(
            json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        temporary.replace(path)

    def write_error(
        self,
        instance_id: str,
        stage: str,
        error: BaseException,
        *,
        duration_seconds: Optional[float] = None,
    ) -> None:
        record = {
            "instance_id": instance_id,
            "stage": stage,
            "error_type": type(error).__name__,
            "message": str(error),
            "duration_seconds": duration_seconds,
        }
        encoded = json.dumps(record, ensure_ascii=False) + "\n"
        with self._lock:
            with (self.directory / "errors.jsonl").open("a", encoding="utf-8") as file:
                file.write(encoded)
                file.flush()
