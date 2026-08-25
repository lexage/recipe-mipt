#!/usr/bin/env python3
"""Filter a JSONL dataset by instance_id and write rows in a fixed order."""

from __future__ import annotations

import argparse
import json
import os
import tempfile
from pathlib import Path
from typing import Any


INSTANCE_IDS = [
    "ASPP__pelita-875",
    "DKISTDC__dkist-475",
    "Delgan__loguru-1239",
    "Delgan__loguru-1299",
    "NeurodataWithoutBorders__pynwb-2052",
    "NeurodataWithoutBorders__pynwb-2056",
    "NeurodataWithoutBorders__pynwb-2060",
    "PrefectHQ__prefect-17123",
    "PyPSA__linopy-412",
    "PyPSA__linopy-438",
    "SpikeInterface__spikeinterface-3829",
    "TeamGraphix__graphix-225",
    "Toloka__crowd-kit-128",
    "UXARRAY__uxarray-1003",
    "All-Hands-AI__openhands-aci-55",
    "AzureAD__microsoft-authentication-library-for-python-795",
    "Ch00k__ffmpy-84",
    "CrossGL__crosstl-257",
    "Delgan__loguru-1297",
    "Delgan__loguru-1303",
    "Delgan__loguru-1306",
    "GoogleCloudPlatform__cloud-sql-python-connector-1221",
    "IAMconsortium__nomenclature-418",
    "IAMconsortium__nomenclature-431",
    "JoaquinAmatRodrigo__skforecast-819",
    "Kinto__kinto-http.py-384",
    "MatterMiners__tardis-354",
    "MatterMiners__tardis-361",
    "NeurodataWithoutBorders__pynwb-1975",
    "OCHA-DAP__hdx-python-country-63",
    "OSeMOSYS__otoole-243",
    "OpenFreeEnergy__openfe-1186",
    "ASPP__pelita-863",
    "Blaizzy__mlx-vlm-179",
    "Colin-b__httpx_auth-105",
    "DKISTDC__dkist-453",
    "DS4SD__docling-824",
    "DavidVujic__python-polylith-289",
    "Delgan__loguru-1236",
    "IAMconsortium__nomenclature-419",
    "IAMconsortium__nomenclature-442",
    "IAMconsortium__nomenclature-456",
    "IAMconsortium__nomenclature-460",
    "IAMconsortium__nomenclature-477",
    "ImperialCollegeLondon__pycsvy-94",
    "JeffLIrion__python-androidtv-351",
    "Lightning-AI__litdata-461",
    "Lightning-AI__litdata-499",
    "MAIF__arta-37",
    "MAIF__meteole-33",
]


def read_jsonl(path: Path) -> dict[str, dict[str, Any]]:
    """Read rows and index them by unique instance_id."""
    rows_by_id: dict[str, dict[str, Any]] = {}

    with path.open("r", encoding="utf-8") as source:
        for line_number, line in enumerate(source, start=1):
            if not line.strip():
                continue

            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Invalid JSON at {path}:{line_number}: {exc.msg}"
                ) from exc

            if not isinstance(row, dict):
                raise ValueError(f"Expected a JSON object at {path}:{line_number}")

            instance_id = row.get("instance_id")
            if not isinstance(instance_id, str) or not instance_id:
                raise ValueError(
                    f"Missing or invalid instance_id at {path}:{line_number}"
                )
            if instance_id in rows_by_id:
                raise ValueError(
                    f"Duplicate instance_id {instance_id!r} at {path}:{line_number}"
                )

            rows_by_id[instance_id] = row

    return rows_by_id


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    """Write JSONL atomically so an interrupted run does not corrupt the file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    file_descriptor, temporary_name = tempfile.mkstemp(
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
        text=True,
    )

    try:
        with os.fdopen(file_descriptor, "w", encoding="utf-8") as target:
            for row in rows:
                target.write(json.dumps(row, ensure_ascii=False) + "\n")
        os.replace(temporary_name, path)
    except BaseException:
        Path(temporary_name).unlink(missing_ok=True)
        raise


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Keep only the configured instance_id values in a JSONL file and "
            "write them in the configured order."
        )
    )
    parser.add_argument("input", type=Path, help="Source JSONL file")
    output_group = parser.add_mutually_exclusive_group(required=True)
    output_group.add_argument("-o", "--output", type=Path, help="Output JSONL file")
    output_group.add_argument(
        "--in-place",
        action="store_true",
        help="Replace the input file after successful validation",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_path = args.input.resolve()
    output_path = input_path if args.in_place else args.output.resolve()

    rows_by_id = read_jsonl(input_path)
    missing_ids = [instance_id for instance_id in INSTANCE_IDS if instance_id not in rows_by_id]
    if missing_ids:
        missing = "\n".join(f"  - {instance_id}" for instance_id in missing_ids)
        raise SystemExit(
            f"Nothing was written: {len(missing_ids)} requested IDs are missing:\n{missing}"
        )

    selected_rows = [rows_by_id[instance_id] for instance_id in INSTANCE_IDS]
    write_jsonl(output_path, selected_rows)
    print(f"Written {len(selected_rows)} rows to {output_path}")


if __name__ == "__main__":
    main()
