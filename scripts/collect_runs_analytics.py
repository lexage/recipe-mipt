#!/usr/bin/env python3
"""Collect SWE-rebench experiment analytics from directories under ``runs``."""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


COLUMNS = (
    "run_name",
    "model_name",
    "total_instances",
    "resolved_instances",
    "unresolved_instances",
    "empty_patch_instances",
    "resolved_ids",
    "critic_tool_name",
    "trajectory_summary_enabled",
    "critic_tool_call_count",
    "critic_tool_task_ids",
    "inference_duration_seconds",
    "evaluation_duration_seconds",
)

MAX_COLUMN_WIDTHS = {
    "run_name": 55,
    "model_name": 36,
    "total_instances": 18,
    "resolved_instances": 21,
    "unresolved_instances": 23,
    "empty_patch_instances": 25,
    "resolved_ids": 50,
    "critic_tool_name": 22,
    "trajectory_summary_enabled": 30,
    "critic_tool_call_count": 25,
    "critic_tool_task_ids": 50,
    "inference_duration_seconds": 28,
    "evaluation_duration_seconds": 29,
}

CRITIC_TOOL_TYPES = {"CRITIC_TOOL", "SWE_REBENCH_CRITIC_TOOL"}
REPORT_COUNT_FIELDS = {
    "total_instances",
    "resolved_instances",
    "unresolved_instances",
    "empty_patch_instances",
}
INSTANCE_ID_PATTERN = re.compile(r"\binstance_id=([^\s\t]+)")


@dataclass(frozen=True)
class PipelineAnalytics:
    model_name: str | None
    critic_tool_names: tuple[str, ...]
    trajectory_summary_enabled: bool


def warn(message: str) -> None:
    print(f"Предупреждение: {message}", file=sys.stderr)


def load_mapping(path: Path, *, yaml_file: bool = False) -> dict[str, Any] | None:
    try:
        with path.open(encoding="utf-8") as source:
            value = yaml.safe_load(source) if yaml_file else json.load(source)
    except (OSError, json.JSONDecodeError, yaml.YAMLError) as error:
        warn(f"не удалось прочитать {path}: {error}")
        return None

    if not isinstance(value, dict):
        warn(
            f"ожидался объект в {path}, "
            f"получено {type(value).__name__}"
        )
        return None
    return value


def as_mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def primary_agent_params(config: Mapping[str, Any]) -> Mapping[str, Any]:
    components = as_mapping(config.get("components"))
    for component_name in ("agent", "react_agent", "solver_agent"):
        component = as_mapping(components.get(component_name))
        params = as_mapping(component.get("params"))
        if params:
            return params
    return {}


def extract_pipeline_analytics(config: Mapping[str, Any]) -> PipelineAnalytics:
    components = as_mapping(config.get("components"))
    agent_params = primary_agent_params(config)

    model_name_value = agent_params.get("model_name")
    model_name = (
        str(model_name_value).strip()
        if model_name_value is not None and str(model_name_value).strip()
        else None
    )

    summary_value = agent_params.get("trajectory_summary_enabled", False)
    if isinstance(summary_value, str):
        trajectory_summary_enabled = summary_value.strip().lower() == "true"
    else:
        trajectory_summary_enabled = bool(summary_value)

    critic_tool_names: list[str] = []
    tools = components.get("tools")
    if isinstance(tools, list):
        for tool in tools:
            tool_config = as_mapping(tool)
            tool_type = str(tool_config.get("type", "")).strip().upper()
            if tool_type not in CRITIC_TOOL_TYPES:
                continue
            params = as_mapping(tool_config.get("params"))
            name_value = params.get("name", params.get("method", "critic"))
            name = str(name_value).strip()
            if name and name not in critic_tool_names:
                critic_tool_names.append(name)

    return PipelineAnalytics(
        model_name=model_name,
        critic_tool_names=tuple(critic_tool_names),
        trajectory_summary_enabled=trajectory_summary_enabled,
    )


def find_evaluation_report(run_dir: Path) -> dict[str, Any] | None:
    evaluation_dir = run_dir / "evaluation"
    if not evaluation_dir.is_dir():
        warn(
            f"в запуске {run_dir.name} отсутствует папка evaluation"
        )
        return None

    reports: list[tuple[Path, dict[str, Any]]] = []
    for path in sorted(evaluation_dir.glob("*.json")):
        if path.name == "evaluation_metadata.json":
            continue
        data = load_mapping(path)
        if data is not None and REPORT_COUNT_FIELDS.issubset(data):
            reports.append((path, data))

    if not reports:
        warn(
            f"для запуска {run_dir.name} не найден итоговый JSON оценки"
        )
        return None

    exact = [
        report
        for report in reports
        if report[0].name.endswith(f".{run_dir.name}.json")
    ]
    matching = exact or [report for report in reports if run_dir.name in report[0].name]
    selected = matching or reports
    if len(selected) > 1:
        warn(
            f"для запуска {run_dir.name} найдено несколько отчётов "
            "оценки; "
            f"используется {selected[0][0].name}"
        )
    return selected[0][1]


def infer_instance_id_from_filename(path: Path) -> str | None:
    marker = "swe-rebench-"
    if marker not in path.stem:
        return None
    instance_id = path.stem.rsplit(marker, maxsplit=1)[1].strip()
    return instance_id or None


def build_action_pattern(critic_tool_names: Iterable[str]) -> re.Pattern[str] | None:
    names = sorted(
        {name.strip() for name in critic_tool_names if name.strip()},
        key=len,
        reverse=True,
    )
    if not names:
        return None
    alternatives = "|".join(re.escape(name) for name in names)
    return re.compile(
        rf"(?<![A-Z0-9_])ACTION:\s*(?:{alternatives})(?![A-Z0-9_.-])",
        flags=re.IGNORECASE,
    )


def collect_critic_calls(
    logs_dir: Path, critic_tool_names: Iterable[str]
) -> tuple[int, list[str]]:
    action_pattern = build_action_pattern(critic_tool_names)
    if action_pattern is None:
        return 0, []
    if not logs_dir.is_dir():
        warn(f"не найдена папка логов {logs_dir}")
        return 0, []

    call_count = 0
    task_ids: set[str] = set()
    unattributed_calls = 0

    for path in sorted(logs_dir.rglob("*.log")):
        current_instance_id = infer_instance_id_from_filename(path)
        try:
            source = path.open(encoding="utf-8", errors="replace")
        except OSError as error:
            warn(f"не удалось прочитать лог {path}: {error}")
            continue

        with source:
            for line in source:
                instance_match = INSTANCE_ID_PATTERN.search(line)
                if instance_match:
                    current_instance_id = instance_match.group(1)

                matches = list(action_pattern.finditer(line))
                if not matches:
                    continue
                call_count += len(matches)
                if current_instance_id:
                    task_ids.add(current_instance_id)
                else:
                    unattributed_calls += len(matches)

    if unattributed_calls:
        warn(
            f"для {unattributed_calls} вызовов критики в {logs_dir} "
            "не удалось определить instance_id"
        )
    return call_count, sorted(task_ids)


def serialize_ids(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, (list, tuple, set)):
        ids = sorted(value) if isinstance(value, set) else list(value)
        return json.dumps(ids, ensure_ascii=False)
    warn(
        "ожидался список идентификаторов, "
        f"получено {type(value).__name__}"
    )
    return json.dumps([str(value)], ensure_ascii=False)


def nested_value(data: Mapping[str, Any], section: str, field: str) -> Any:
    return as_mapping(data.get(section)).get(field)


def collect_run(run_dir: Path) -> dict[str, Any]:
    pipeline_path = run_dir / "pipeline.yaml"
    config = (
        load_mapping(pipeline_path, yaml_file=True)
        if pipeline_path.is_file()
        else None
    )
    if config is None:
        warn(
            f"для запуска {run_dir.name} не удалось прочитать pipeline.yaml"
        )
        pipeline = PipelineAnalytics(None, (), False)
    else:
        pipeline = extract_pipeline_analytics(config)

    report = find_evaluation_report(run_dir) or {}

    timings_path = run_dir / "timings.json"
    timings = load_mapping(timings_path) if timings_path.is_file() else None
    if timings is None:
        warn(
            f"для запуска {run_dir.name} не удалось прочитать timings.json"
        )
        timings = {}

    critic_call_count, critic_task_ids = collect_critic_calls(
        run_dir / "logs", pipeline.critic_tool_names
    )

    return {
        "run_name": run_dir.name,
        "model_name": pipeline.model_name,
        "total_instances": report.get("total_instances"),
        "resolved_instances": report.get("resolved_instances"),
        "unresolved_instances": report.get("unresolved_instances"),
        "empty_patch_instances": report.get("empty_patch_instances"),
        "resolved_ids": serialize_ids(report.get("resolved_ids")),
        "critic_tool_name": ", ".join(pipeline.critic_tool_names),
        "trajectory_summary_enabled": pipeline.trajectory_summary_enabled,
        "critic_tool_call_count": critic_call_count,
        "critic_tool_task_ids": serialize_ids(critic_task_ids),
        "inference_duration_seconds": nested_value(
            timings, "inference", "duration_seconds"
        ),
        "evaluation_duration_seconds": nested_value(
            timings, "evaluation", "duration_seconds"
        ),
    }


def collect_runs(runs_dir: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for run_dir in sorted(path for path in runs_dir.iterdir() if path.is_dir()):
        if not (run_dir / "pipeline.yaml").is_file():
            warn(f"папка {run_dir.name} пропущена: нет pipeline.yaml")
            continue
        rows.append(collect_run(run_dir))
    return rows


def csv_value(value: Any) -> Any:
    if isinstance(value, bool):
        return str(value).lower()
    return "" if value is None else value


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=COLUMNS)
        writer.writeheader()
        for row in rows:
            writer.writerow({column: csv_value(row.get(column)) for column in COLUMNS})


def write_excel(path: Path, rows: list[dict[str, Any]]) -> None:
    try:
        from openpyxl import Workbook
        from openpyxl.styles import Alignment, Font, PatternFill
        from openpyxl.utils import get_column_letter
    except ImportError as error:
        raise RuntimeError(
            "Для записи Excel установите зависимости проекта, "
            "включая openpyxl"
        ) from error

    workbook = Workbook()
    sheet = workbook.active
    sheet.title = "runs"
    sheet.append(list(COLUMNS))
    for row in rows:
        sheet.append([row.get(column) for column in COLUMNS])

    header_fill = PatternFill(fill_type="solid", fgColor="1F4E78")
    for cell in sheet[1]:
        cell.fill = header_fill
        cell.font = Font(color="FFFFFF", bold=True)
        cell.alignment = Alignment(horizontal="center", vertical="center")

    sheet.freeze_panes = "A2"
    sheet.auto_filter.ref = sheet.dimensions
    sheet.sheet_view.showGridLines = False
    sheet.sheet_view.zoomScale = 85
    sheet.row_dimensions[1].height = 32
    sheet.page_setup.orientation = "landscape"
    sheet.page_setup.fitToWidth = 1
    sheet.page_setup.fitToHeight = 0
    sheet.sheet_properties.pageSetUpPr.fitToPage = True

    numeric_columns = {
        "total_instances",
        "resolved_instances",
        "unresolved_instances",
        "empty_patch_instances",
        "critic_tool_call_count",
        "inference_duration_seconds",
        "evaluation_duration_seconds",
    }
    for column_index, column_name in enumerate(COLUMNS, start=1):
        column_letter = get_column_letter(column_index)
        values = [column_name, *(str(row.get(column_name) or "") for row in rows)]
        width = min(
            max(len(value) for value in values) + 2,
            MAX_COLUMN_WIDTHS[column_name],
        )
        sheet.column_dimensions[column_letter].width = max(width, 12)
        if column_name in numeric_columns:
            for cell in sheet[column_letter][1:]:
                cell.number_format = "#,##0"

    wrapped_columns = {1, 2, 7, 9, 11}
    for row in sheet.iter_rows(min_row=2):
        for cell in row:
            cell.alignment = Alignment(
                vertical="top",
                wrap_text=cell.column in wrapped_columns,
            )

    path.parent.mkdir(parents=True, exist_ok=True)
    workbook.save(path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Собрать сводную аналитику запусков SWE-rebench "
            "из папки runs "
            "в CSV и Excel."
        )
    )
    parser.add_argument(
        "--runs-dir",
        type=Path,
        default=Path("runs"),
        help=(
            "Папка с экспериментами (по умолчанию: runs)"
        ),
    )
    parser.add_argument(
        "--output-prefix",
        type=Path,
        default=None,
        help=(
            "Путь без расширения для выходных файлов; "
            "по умолчанию "
            "<runs-dir>/runs_analytics"
        ),
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    runs_dir = args.runs_dir.expanduser().resolve()
    if not runs_dir.is_dir():
        print(
            f"Папка с запусками не найдена: {runs_dir}",
            file=sys.stderr,
        )
        return 2

    output_prefix = args.output_prefix or runs_dir / "runs_analytics"
    output_prefix = output_prefix.expanduser().resolve()
    if output_prefix.suffix.lower() in {".csv", ".xlsx"}:
        output_prefix = output_prefix.with_suffix("")

    rows = collect_runs(runs_dir)
    csv_path = output_prefix.with_suffix(".csv")
    excel_path = output_prefix.with_suffix(".xlsx")
    write_csv(csv_path, rows)
    try:
        write_excel(excel_path, rows)
    except RuntimeError as error:
        print(str(error), file=sys.stderr)
        return 1

    print(f"Обработано запусков: {len(rows)}")
    print(f"CSV: {csv_path}")
    print(f"Excel: {excel_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
