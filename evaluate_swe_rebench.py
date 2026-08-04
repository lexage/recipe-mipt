"""Validate predictions and invoke the external SWE-rebench evaluator."""

import argparse
from pathlib import Path

from src.benchmarks.swe_rebench import EvaluationConfig, SWERebenchEvaluator


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fork-path", required=True)
    parser.add_argument("--predictions-path", required=True)
    parser.add_argument("--dataset-name", default="SWE-rebench/SWE-rebench")
    parser.add_argument("--split", default="test")
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--max-workers", type=int, default=4)
    parser.add_argument("--timeout", type=int, default=1_800)
    parser.add_argument("--namespace", default="swebench")
    parser.add_argument("--instance-image-tag", default="latest")
    parser.add_argument("--report-dir", default="swe-rebench-evaluation")
    parser.add_argument("--instance-ids", nargs="*", default=())
    parser.add_argument(
        "--check-only",
        action="store_true",
        help="Validate JSONL and evaluator CLI without starting evaluation containers",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    evaluator = SWERebenchEvaluator(
        EvaluationConfig(
            fork_path=Path(args.fork_path),
            predictions_path=Path(args.predictions_path),
            dataset_name=args.dataset_name,
            split=args.split,
            run_id=args.run_id,
            max_workers=args.max_workers,
            timeout=args.timeout,
            namespace=args.namespace,
            instance_image_tag=args.instance_image_tag,
            instance_ids=tuple(args.instance_ids),
            report_dir=Path(args.report_dir),
        )
    )
    result = evaluator.run(check_only=args.check_only)
    if args.check_only:
        print("Predictions and evaluator CLI are compatible.")
        return 0
    return result.returncode if result is not None else 0


if __name__ == "__main__":
    raise SystemExit(main())
