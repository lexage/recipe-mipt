"""Validated hand-off to the external SWE-rebench evaluation harness."""

import json
import subprocess
from datetime import datetime, timezone
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .data_types import SWERebenchDataError
from .predictions import SWERebenchPrediction

REQUIRED_HARNESS_OPTIONS = (
    "--dataset_name",
    "--split",
    "--predictions_path",
    "--max_workers",
    "--run_id",
)


def validate_predictions(
    path: str | Path, *, allowed_instance_ids: set[str] | None = None
) -> tuple[SWERebenchPrediction, ...]:
    """Validate the complete evaluator input before starting expensive containers."""

    predictions_path = Path(path).expanduser().resolve()
    if not predictions_path.is_file():
        raise SWERebenchDataError(
            f"Predictions file does not exist: {predictions_path}"
        )

    predictions: list[SWERebenchPrediction] = []
    seen: set[str] = set()
    with predictions_path.open(encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            if not line.strip():
                continue
            try:
                record: Any = json.loads(line)
                if not isinstance(record, dict) or set(record) != {
                    "instance_id",
                    "model_name_or_path",
                    "model_patch",
                }:
                    raise SWERebenchDataError(
                        "record must contain exactly the prediction fields"
                    )
                prediction = SWERebenchPrediction(**record)
            except (json.JSONDecodeError, TypeError, SWERebenchDataError) as error:
                raise SWERebenchDataError(
                    f"Invalid prediction on line {line_number}: {error}"
                ) from error
            if prediction.instance_id in seen:
                raise SWERebenchDataError(
                    f"Duplicate prediction on line {line_number}: {prediction.instance_id}"
                )
            if (
                allowed_instance_ids is not None
                and prediction.instance_id not in allowed_instance_ids
            ):
                raise SWERebenchDataError(
                    f"Prediction is not in the selected dataset: {prediction.instance_id}"
                )
            seen.add(prediction.instance_id)
            predictions.append(prediction)

    if not predictions:
        raise SWERebenchDataError("Predictions file is empty")
    return tuple(predictions)


@dataclass(frozen=True)
class EvaluationConfig:
    fork_path: Path
    predictions_path: Path
    dataset_name: str
    split: str
    run_id: str
    max_workers: int = 4
    timeout: int = 1_800
    namespace: str = "swebench"
    instance_image_tag: str = "latest"
    instance_ids: tuple[str, ...] = ()
    report_dir: Path = Path(".")

    def __post_init__(self) -> None:
        if self.max_workers < 1 or self.timeout < 1:
            raise ValueError("Evaluation limits must be positive")
        for name in ("dataset_name", "split", "run_id"):
            if not getattr(self, name).strip():
                raise ValueError(f"{name} must be non-empty")


class SWERebenchEvaluator:
    """Run the official harness as a subprocess without importing its internals."""

    def __init__(self, config: EvaluationConfig, *, python: str = "python") -> None:
        self.config = config
        self.python = python

    def check_compatibility(self) -> str:
        fork = self.config.fork_path.expanduser().resolve()
        if not fork.is_dir():
            raise RuntimeError(f"Evaluator checkout does not exist: {fork}")
        result = subprocess.run(
            [self.python, "-m", "swebench.harness.run_evaluation", "--help"],
            cwd=fork,
            text=True,
            capture_output=True,
            check=False,
            timeout=30,
        )
        if result.returncode:
            detail = result.stderr.strip() or result.stdout.strip()
            raise RuntimeError(f"Could not start SWE-rebench evaluator: {detail}")
        missing = [
            option for option in REQUIRED_HARNESS_OPTIONS if option not in result.stdout
        ]
        if missing:
            raise RuntimeError(
                "Evaluator CLI is incompatible; missing options: " + ", ".join(missing)
            )
        return result.stdout

    def command(self) -> list[str]:
        config = self.config
        command = [
            self.python,
            "-m",
            "swebench.harness.run_evaluation",
            "--dataset_name",
            config.dataset_name,
            "--split",
            config.split,
            "--predictions_path",
            str(config.predictions_path.expanduser().resolve()),
            "--max_workers",
            str(config.max_workers),
            "--timeout",
            str(config.timeout),
            "--run_id",
            config.run_id,
            "--namespace",
            config.namespace,
            "--instance_image_tag",
            config.instance_image_tag,
            "--report_dir",
            str(config.report_dir.expanduser().resolve()),
        ]
        if config.instance_ids:
            command.extend(("--instance_ids", *config.instance_ids))
        return command

    def write_metadata(self) -> Path:
        """Record the exact hand-off before the external process starts."""

        config = self.config
        report_dir = config.report_dir.expanduser().resolve()
        report_dir.mkdir(parents=True, exist_ok=True)
        metadata_path = report_dir / "evaluation_metadata.json"
        temporary = metadata_path.with_suffix(".json.tmp")
        metadata = {
            "created_at": datetime.now(timezone.utc).isoformat(),
            "dataset_name": config.dataset_name,
            "split": config.split,
            "run_id": config.run_id,
            "predictions_path": str(config.predictions_path.expanduser().resolve()),
            "fork_path": str(config.fork_path.expanduser().resolve()),
            "fork_commit": self._git_revision(config.fork_path),
            "generator_commit": self._git_revision(Path(__file__).parents[3]),
            "command": self.command(),
            "instance_ids": list(config.instance_ids),
        }
        temporary.write_text(
            json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        temporary.replace(metadata_path)
        return metadata_path

    def run(
        self, *, check_only: bool = False
    ) -> subprocess.CompletedProcess[str] | None:
        validate_predictions(self.config.predictions_path)
        self.check_compatibility()
        self.write_metadata()
        if check_only:
            return None
        return subprocess.run(
            self.command(),
            cwd=self.config.fork_path.expanduser().resolve(),
            text=True,
            check=False,
        )

    @staticmethod
    def _git_revision(path: Path) -> str | None:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=path.expanduser().resolve(),
            text=True,
            capture_output=True,
            check=False,
            timeout=10,
        )
        return result.stdout.strip() if result.returncode == 0 else None
