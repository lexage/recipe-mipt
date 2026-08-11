"""Materialize a pinned SWE-rebench dataset snapshot for inference and evaluation."""

import argparse
import hashlib
import json
from collections.abc import Iterable
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, help="Hub dataset name or JSON/JSONL")
    parser.add_argument("--split", required=True)
    parser.add_argument(
        "--dataset-revision",
        help="Required for Hub datasets; commit SHA or immutable dataset revision",
    )
    parser.add_argument(
        "--created-at-from",
        help="Include tasks created at or after this UTC date/time",
    )
    parser.add_argument(
        "--created-at-before",
        help="Include tasks created before this UTC date/time",
    )
    parser.add_argument("--instance-ids-file")
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--stop", type=int)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--output", required=True, help="Pinned JSONL output path")
    return parser.parse_args()


def _read_local(path: Path, split: str) -> list[dict[str, Any]]:
    if path.suffix.lower() == ".jsonl":
        records = [
            json.loads(line)
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
    elif path.suffix.lower() == ".json":
        payload = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(payload, dict):
            if split not in payload:
                raise ValueError(f"Dataset JSON has no split {split!r}")
            payload = payload[split]
        records = payload
    else:
        raise ValueError("Local dataset must use .json or .jsonl")
    if not isinstance(records, list) or not all(
        isinstance(item, dict) for item in records
    ):
        raise ValueError("Dataset must contain JSON objects")
    return records


def _load_records(
    dataset: str, split: str, dataset_revision: str | None
) -> list[dict[str, Any]]:
    candidate = Path(dataset).expanduser()
    if candidate.is_file():
        return _read_local(candidate, split)
    if candidate.suffix.lower() in {".json", ".jsonl"}:
        raise FileNotFoundError(f"Dataset file does not exist: {candidate}")
    if not dataset_revision:
        raise ValueError(
            "Hub datasets require --dataset-revision; use an immutable commit SHA"
        )
    from datasets import load_dataset

    loaded = load_dataset(dataset, split=split, revision=dataset_revision)
    return [dict(record) for record in loaded]


def _read_instance_ids(path: str | Path | None) -> list[str] | None:
    if path is None:
        return None
    values = [
        line.strip()
        for line in Path(path).expanduser().read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]
    if len(values) != len(set(values)):
        raise ValueError("Instance ID file contains duplicate values")
    return values


def _parse_utc_datetime(value: str | datetime, *, field_name: str) -> datetime:
    """Parse an ISO-8601 value and normalize it to UTC."""

    if isinstance(value, datetime):
        parsed = value
    elif isinstance(value, str) and value.strip():
        normalized = value.strip()
        if normalized.endswith("Z"):
            normalized = normalized[:-1] + "+00:00"
        try:
            parsed = datetime.fromisoformat(normalized)
        except ValueError as error:
            raise ValueError(
                f"{field_name} must be an ISO-8601 date/time"
            ) from error
    else:
        raise ValueError(f"{field_name} must be a non-empty ISO-8601 date/time")

    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    else:
        parsed = parsed.astimezone(timezone.utc)
    return parsed


def _created_at_bounds(
    created_at_from: str | datetime | None,
    created_at_before: str | datetime | None,
) -> tuple[datetime | None, datetime | None]:
    lower = (
        _parse_utc_datetime(created_at_from, field_name="created_at_from")
        if created_at_from is not None
        else None
    )
    upper = (
        _parse_utc_datetime(created_at_before, field_name="created_at_before")
        if created_at_before is not None
        else None
    )
    if lower is not None and upper is not None and lower >= upper:
        raise ValueError("Require created_at_from < created_at_before")
    return lower, upper


def materialize_snapshot(
    *,
    dataset: str,
    split: str,
    output: str | Path,
    dataset_revision: str | None = None,
    instance_ids: Iterable[str] | None = None,
    created_at_from: str | datetime | None = None,
    created_at_before: str | datetime | None = None,
    start: int = 0,
    stop: int | None = None,
    limit: int | None = None,
) -> tuple[Path, Path]:
    """Write raw records, including evaluator fields, to an immutable local JSONL."""

    if start < 0 or (stop is not None and stop < start):
        raise ValueError("Require 0 <= start <= stop")
    if limit is not None and limit < 0:
        raise ValueError("limit must be non-negative")
    created_at_lower, created_at_upper = _created_at_bounds(
        created_at_from, created_at_before
    )
    records = _load_records(dataset, split, dataset_revision)
    source_instance_count = len(records)
    by_id: dict[str, dict[str, Any]] = {}
    for position, record in enumerate(records):
        instance_id = record.get("instance_id")
        if not isinstance(instance_id, str) or not instance_id.strip():
            raise ValueError(f"Record {position} has no valid instance_id")
        if instance_id in by_id:
            raise ValueError(f"Duplicate instance_id: {instance_id}")
        by_id[instance_id] = record

    if created_at_lower is not None or created_at_upper is not None:
        filtered_records: list[dict[str, Any]] = []
        for position, record in enumerate(records):
            instance_id = record["instance_id"]
            try:
                created_at = _parse_utc_datetime(
                    record.get("created_at"),
                    field_name="created_at",
                )
            except ValueError as error:
                raise ValueError(
                    f"Record {position} ({instance_id}) has invalid created_at: {error}"
                ) from error
            if created_at_lower is not None and created_at < created_at_lower:
                continue
            if created_at_upper is not None and created_at >= created_at_upper:
                continue
            filtered_records.append(record)
        records = filtered_records
    created_at_filtered_count = len(records)

    if instance_ids is not None:
        requested = list(instance_ids)
        if len(requested) != len(set(requested)):
            raise ValueError("Requested instance IDs contain duplicates")
        missing = sorted(set(requested) - by_id.keys())
        if missing:
            raise ValueError("Instance IDs not found: " + ", ".join(missing))
        requested_set = set(requested)
        records = [record for record in records if record["instance_id"] in requested_set]

    records = records[start:stop]
    if limit is not None:
        records = records[:limit]
    if not records:
        raise ValueError("Snapshot selection is empty")

    output_path = Path(output).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    encoded = "".join(
        json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n"
        for record in records
    ).encode("utf-8")
    temporary = output_path.with_suffix(output_path.suffix + ".tmp")
    temporary.write_bytes(encoded)
    temporary.replace(output_path)

    metadata = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "source_dataset": dataset,
        "source_split": split,
        "source_revision": dataset_revision,
        "source_instance_count": source_instance_count,
        "created_at_filtered_count": created_at_filtered_count,
        "filters": {
            "created_at": {
                "from": (
                    created_at_lower.isoformat()
                    if created_at_lower is not None
                    else None
                ),
                "before": (
                    created_at_upper.isoformat()
                    if created_at_upper is not None
                    else None
                ),
            }
        },
        "snapshot_path": str(output_path),
        "snapshot_sha256": hashlib.sha256(encoded).hexdigest(),
        "instance_count": len(records),
        "instance_ids": [record["instance_id"] for record in records],
    }
    metadata_path = output_path.with_suffix(output_path.suffix + ".metadata.json")
    metadata_temporary = metadata_path.with_suffix(metadata_path.suffix + ".tmp")
    metadata_temporary.write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    metadata_temporary.replace(metadata_path)
    return output_path, metadata_path


def main() -> int:
    args = parse_args()
    output, metadata = materialize_snapshot(
        dataset=args.dataset,
        split=args.split,
        dataset_revision=args.dataset_revision,
        instance_ids=_read_instance_ids(args.instance_ids_file),
        created_at_from=args.created_at_from,
        created_at_before=args.created_at_before,
        start=args.start,
        stop=args.stop,
        limit=args.limit,
        output=args.output,
    )
    print(output)
    print(metadata)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
