"""Generate SWE-rebench prediction JSONL with an isolated container per task."""

import argparse
import hashlib
import json
import logging
import subprocess
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
from src.utils.loggers import create_logging

logging.getLogger("openai").setLevel(logging.ERROR)
logging.getLogger("httpx").setLevel(logging.ERROR)
logging.getLogger("httpcore").setLevel(logging.ERROR)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate patches for SWE-rebench instances"
    )
    parser.add_argument("--config", required=True, help="Pipeline YAML file")
    parser.add_argument("--dataset", required=True, help="Hub name or JSON/JSONL")
    parser.add_argument("--split", required=True)
    parser.add_argument("--dataset-revision")
    parser.add_argument("--instance-ids-file")
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--stop", type=int)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--output", required=True, help="Prediction JSONL path")
    parser.add_argument("--model-name-or-path", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--num-workers", type=int, default=1)
    parser.add_argument(
        "--namespace", help="Image namespace; required only for convention fallback"
    )
    parser.add_argument("--architecture", default="x86_64")
    parser.add_argument("--image-tag", default="latest")
    parser.add_argument(
        "--image-manifest",
        help="Pinned JSON mapping instance IDs to exact image names and platforms",
    )
    parser.add_argument(
        "--allow-image-convention",
        action="store_true",
        help="Unsafe opt-in naming fallback for local smoke tests only",
    )
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
    parser.add_argument("--task-timeout", type=int, default=3_600)
    parser.add_argument(
        "--logs-path",
        help="Log directory or file; overrides logs_path from pipeline YAML",
    )
    parser.add_argument(
        "--evaluator-fork-path",
        help="Optional pinned SWE-bench-fork checkout recorded in run metadata",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    import docker
    import yaml

    from src.pipelines.configs import ConfigLoader
    from src.pipelines.pipeline_builder import PipelineBuilder

    pipeline_config = ConfigLoader().load_from_yaml(args.config)
    output = Path(args.output)
    logs_path = (
        args.logs_path or pipeline_config.logs_path or str(output.parent / "logs")
    )
    # Parent lifecycle messages go to the main log. Each task process inherits
    # the routing handler and writes existing agent logs plus stage events to a
    # dedicated process file, without changing agent implementations.
    create_logging(
        log_path=logs_path,
        tag=args.run_id,
        n_workers=args.num_workers,
        route=True,
    )
    logging.info(
        "STAGE\tRUN_START\trun_id=%s\tdataset=%s\tsplit=%s\tworkers=%s",
        args.run_id,
        args.dataset,
        args.split,
        args.num_workers,
    )
    logging.info("STAGE\tCONFIG_LOADED\tpath=%s\tlogs_path=%s", args.config, logs_path)

    dataset = DatasetSWERebench.load(
        args.dataset, split=args.split, revision=args.dataset_revision
    )
    logging.info("STAGE\tDATASET_LOADED\ttasks=%s", len(dataset))
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
    logging.info(
        "STAGE\tTASKS_SELECTED\tcount=%s\tinstance_ids=%s",
        len(tasks),
        ",".join(task.instance_id for task in tasks),
    )
    predictions = PredictionsWriter(
        output,
        model_name_or_path=args.model_name_or_path,
        resume=args.resume,
        fsync=True,
    )
    artifacts = RunArtifactsWriter(output.parent)
    config_path = Path(args.config).resolve()
    selected_ids = [task.instance_id for task in tasks]
    generator = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=Path(__file__).resolve().parent,
        text=True,
        capture_output=True,
        check=False,
    )
    evaluator_commit = None
    if args.evaluator_fork_path:
        evaluator_revision = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=Path(args.evaluator_fork_path).expanduser().resolve(),
            text=True,
            capture_output=True,
            check=False,
        )
        if evaluator_revision.returncode != 0:
            raise RuntimeError(
                "Could not determine evaluator fork revision: "
                + (
                    evaluator_revision.stderr.strip()
                    or evaluator_revision.stdout.strip()
                )
            )
        evaluator_commit = evaluator_revision.stdout.strip()
    raw_pipeline_config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    components = raw_pipeline_config.get("components", {})
    agent_params = components.get("agent", {}).get("params", {})
    planner_params = components.get("planner", {}).get("params", {})
    command_tool = next(
        (
            tool.get("params", {})
            for tool in components.get("tools", [])
            if tool.get("type") == "RUN_COMMAND_TOOL"
        ),
        {},
    )
    run_metadata = {
        "dataset": args.dataset,
        "split": args.split,
        "dataset_revision": args.dataset_revision,
        "run_id": args.run_id,
        "model_name_or_path": args.model_name_or_path,
        "pipeline": raw_pipeline_config.get("type"),
        "temperature": agent_params.get(
            "temperature", planner_params.get("temperature")
        ),
        "max_iterations": agent_params.get(
            "max_iterations", planner_params.get("maximum_steps")
        ),
        "history_context": agent_params.get("history_context"),
        "llm_timeout": agent_params.get(
            "llm_timeout", planner_params.get("llm_timeout")
        ),
        "command_timeout": command_tool.get("default_timeout"),
        "selected_instance_ids": selected_ids,
        "generator_commit": (
            generator.stdout.strip() if generator.returncode == 0 else None
        ),
        "evaluator_commit": evaluator_commit,
        "pipeline_config": str(config_path),
        "pipeline_config_sha256": hashlib.sha256(config_path.read_bytes()).hexdigest(),
        "namespace": args.namespace,
        "architecture": args.architecture,
        "image_tag": args.image_tag,
        "image_manifest": args.image_manifest,
        "allow_image_convention": args.allow_image_convention,
        "task_timeout": args.task_timeout,
        "logs_path": str(Path(logs_path).expanduser()),
        "network_enabled": args.allow_network,
        "image_by_instance": {},
    }

    def pipeline_factory():
        return PipelineBuilder().build(pipeline_config)

    if args.image_manifest:
        image_resolver = InstanceImageResolver.from_manifest(args.image_manifest)
    else:
        image_resolver = InstanceImageResolver(
            namespace=args.namespace,
            architecture=args.architecture,
            image_tag=args.image_tag,
            allow_convention=args.allow_image_convention,
        )
    image_by_instance = {}
    for task in tasks:
        resolved_image = image_resolver.resolve(task)
        logging.info(
            "STAGE\tIMAGE_RESOLVED\tinstance_id=%s\timage=%s\tsource=%s",
            task.instance_id,
            resolved_image.name,
            resolved_image.source,
        )
        image_by_instance[task.instance_id] = {
            "name": resolved_image.name,
            "platform": resolved_image.platform,
            "source": resolved_image.source,
            "workdir": resolved_image.workdir,
            "user": resolved_image.user,
            "cap_add": list(resolved_image.cap_add),
        }
    run_metadata["image_by_instance"] = image_by_instance
    artifacts.write_metadata(run_metadata)

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
        task_timeout=args.task_timeout,
    )
    summary = runner.run(tasks)
    logging.info(
        "STAGE\tRUN_FINISHED\tsubmitted=%s\tcompleted=%s\tfailed=%s\tskipped=%s",
        summary.submitted,
        summary.completed,
        summary.failed,
        summary.skipped,
    )
    print(json.dumps(summary.__dict__, indent=2))
    return 1 if summary.failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
