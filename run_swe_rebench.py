"""Generate SWE-rebench prediction JSONL with an isolated container per task."""

import argparse
import json
from pathlib import Path

from src.benchmarks.swe_rebench import (
    DatasetSWERebench,
    DockerRepositoryRuntime,
    InstanceImageResolver,
    PredictionsWriter,
    RunArtifactsWriter,
    SWERebenchInferenceRunner,
    SWERebenchPromptBuilder,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate patches for SWE-rebench instances"
    )
    parser.add_argument("--config", required=True, help="Pipeline YAML file")
    parser.add_argument(
        "--dataset", default="SWE-rebench/SWE-rebench", help="Hub name or JSON/JSONL"
    )
    parser.add_argument("--split", default="test")
    parser.add_argument("--dataset-revision")
    parser.add_argument("--instance-ids-file")
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--stop", type=int)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--output", required=True, help="Prediction JSONL path")
    parser.add_argument("--model-name-or-path", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--num-workers", type=int, default=1)
    parser.add_argument("--namespace", default="swebench")
    parser.add_argument("--architecture", default="x86_64")
    parser.add_argument("--image-tag", default="latest")
    parser.add_argument(
        "--pull-policy", choices=("always", "missing", "never"), default="missing"
    )
    parser.add_argument("--memory-limit")
    parser.add_argument("--nano-cpus", type=int)
    parser.add_argument("--allow-network", action="store_true")
    parser.add_argument("--keep-containers", action="store_true")
    parser.add_argument("--include-hints", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--fail-fast", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    from src.pipelines.configs import ConfigLoader
    from src.pipelines.pipeline_builder import PipelineBuilder

    try:
        import docker
    except ImportError as error:
        raise RuntimeError(
            "Docker inference requires the Docker SDK for Python"
        ) from error

    dataset = DatasetSWERebench.load(
        args.dataset, split=args.split, revision=args.dataset_revision
    )
    instance_ids = (
        DatasetSWERebench.read_instance_ids(args.instance_ids_file)
        if args.instance_ids_file
        else None
    )
    tasks = dataset.select(
        instance_ids=instance_ids,
        start=args.start,
        stop=args.stop,
        limit=args.limit,
    )
    output = Path(args.output)
    predictions = PredictionsWriter(
        output,
        model_name_or_path=args.model_name_or_path,
        resume=args.resume,
        fsync=True,
    )
    artifacts = RunArtifactsWriter(output.parent)
    artifacts.write_metadata(
        {
            "dataset": args.dataset,
            "split": args.split,
            "dataset_revision": args.dataset_revision,
            "run_id": args.run_id,
            "model_name_or_path": args.model_name_or_path,
            "pipeline_config": str(Path(args.config).resolve()),
            "namespace": args.namespace,
            "architecture": args.architecture,
            "image_tag": args.image_tag,
        }
    )

    pipeline_config = ConfigLoader().load_from_yaml(args.config)

    def pipeline_factory():
        return PipelineBuilder().build(pipeline_config)

    image_resolver = InstanceImageResolver(
        namespace=args.namespace,
        architecture=args.architecture,
        image_tag=args.image_tag,
    )

    def runtime_factory(task, image):
        return DockerRepositoryRuntime.create(
            client=docker.from_env(),
            image=image,
            instance_id=task.instance_id,
            base_commit=task.base_commit,
            run_id=args.run_id,
            pull_policy=args.pull_policy,
            keep_container=args.keep_containers,
            network_disabled=not args.allow_network,
            memory_limit=args.memory_limit,
            nano_cpus=args.nano_cpus,
        )

    runner = SWERebenchInferenceRunner(
        pipeline_factory=pipeline_factory,
        runtime_factory=runtime_factory,
        image_resolver=image_resolver,
        predictions_writer=predictions,
        artifacts_writer=artifacts,
        prompt_builder=SWERebenchPromptBuilder(include_hints=args.include_hints),
        max_workers=args.num_workers,
        fail_fast=args.fail_fast,
    )
    summary = runner.run(tasks)
    print(json.dumps(summary.__dict__, indent=2))
    return 1 if summary.failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
