"""Run one SWE-rebench inference instance and optionally evaluate its patch."""

import argparse
import subprocess
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--instance-id", required=True)
    parser.add_argument("--fork-path", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--model-name-or-path", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--split", required=True)
    parser.add_argument("--dataset-revision")
    parser.add_argument("--namespace", required=True)
    parser.add_argument("--architecture", default="x86_64")
    parser.add_argument("--image-tag", default="latest")
    parser.add_argument("--image-manifest")
    parser.add_argument("--allow-image-convention", action="store_true")
    parser.add_argument("--evaluation-check-only", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    root = Path(__file__).resolve().parent
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    ids_path = output_dir / "instance_ids.txt"
    ids_path.write_text(args.instance_id + "\n", encoding="utf-8")
    predictions_path = output_dir / "predictions.jsonl"

    inference = [
        sys.executable,
        str(root / "run_swe_rebench.py"),
        "--config",
        args.config,
        "--dataset",
        args.dataset,
        "--split",
        args.split,
        "--instance-ids-file",
        str(ids_path),
        "--limit",
        "1",
        "--output",
        str(predictions_path),
        "--model-name-or-path",
        args.model_name_or_path,
        "--run-id",
        args.run_id,
        "--num-workers",
        "1",
        "--evaluator-fork-path",
        args.fork_path,
        "--namespace",
        args.namespace,
        "--architecture",
        args.architecture,
        "--image-tag",
        args.image_tag,
    ]
    if args.dataset_revision:
        inference.extend(("--dataset-revision", args.dataset_revision))
    if args.image_manifest:
        inference.extend(("--image-manifest", args.image_manifest))
    elif args.allow_image_convention:
        inference.append("--allow-image-convention")
    inference_result = subprocess.run(inference, check=False)
    if inference_result.returncode:
        return inference_result.returncode

    evaluation = [
        sys.executable,
        str(root / "evaluate_swe_rebench.py"),
        "--fork-path",
        args.fork_path,
        "--predictions-path",
        str(predictions_path),
        "--dataset-name",
        args.dataset,
        "--split",
        args.split,
        "--run-id",
        args.run_id,
        "--max-workers",
        "1",
        "--namespace",
        args.namespace,
        "--instance-image-tag",
        args.image_tag,
        "--instance-ids",
        args.instance_id,
        "--report-dir",
        str(output_dir / "evaluation"),
        "--inference-metadata",
        str(output_dir / "run_metadata.json"),
    ]
    if args.dataset_revision:
        evaluation.extend(("--dataset-revision", args.dataset_revision))
    if args.evaluation_check_only:
        evaluation.append("--check-only")
    evaluation_result = subprocess.run(evaluation, check=False)
    if evaluation_result.returncode or args.evaluation_check_only:
        return evaluation_result.returncode
    report_dir = output_dir / "evaluation"
    reports = [
        path
        for path in report_dir.rglob("*")
        if path.is_file() and path.name != "evaluation_metadata.json"
    ]
    if not reports:
        raise RuntimeError(
            "Evaluator exited successfully but produced no instance report in "
            f"{report_dir}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
