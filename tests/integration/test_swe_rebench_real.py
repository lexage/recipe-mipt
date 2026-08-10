"""Opt-in integration check against a real image and evaluator checkout."""

import json
import os
import tempfile
import unittest
from pathlib import Path

from src.benchmarks.swe_rebench import (
    DatasetSWERebench,
    DockerRepositoryRuntime,
    EvaluationConfig,
    InstanceImageResolver,
    SWERebenchEvaluator,
)


REAL_TASK = os.environ.get("SWE_REBENCH_REAL_TASK")
EVALUATOR_FORK = os.environ.get("SWE_REBENCH_EVALUATOR_FORK")


@unittest.skipUnless(
    REAL_TASK and EVALUATOR_FORK,
    "Set SWE_REBENCH_REAL_TASK and SWE_REBENCH_EVALUATOR_FORK",
)
class RealSWERebenchIntegrationTests(unittest.TestCase):
    def test_real_image_lifecycle_and_evaluator_contract(self):
        import docker

        client = docker.from_env(timeout=600)
        client.ping()
        task = DatasetSWERebench.load(REAL_TASK, split="test", revision=None)[0]
        image = InstanceImageResolver().resolve(task)
        container_ids = []

        for suffix in ("first", "second"):
            runtime = DockerRepositoryRuntime.create(
                client=client,
                image=image,
                instance_id=task.instance_id,
                base_commit=task.base_commit,
                run_id=f"real-integration-{suffix}",
                pull_policy="missing",
            )
            container_ids.append(runtime.container.id)
            head = runtime.run_command("git rev-parse HEAD")
            self.assertEqual(head.exit_code, 0)
            self.assertEqual(head.stdout.strip(), task.base_commit)
            runtime.close()

        self.assertNotEqual(container_ids[0], container_ids[1])
        for container_id in container_ids:
            with self.assertRaises(docker.errors.NotFound):
                client.containers.get(container_id)

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            predictions = root / "predictions.jsonl"
            predictions.write_text(
                json.dumps(
                    {
                        "instance_id": task.instance_id,
                        "model_name_or_path": "integration/check",
                        "model_patch": "",
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            metadata = root / "run_metadata.json"
            metadata.write_text(
                json.dumps(
                    {
                        "dataset": str(Path(REAL_TASK).expanduser()),
                        "dataset_revision": None,
                        "split": "test",
                        "selected_instance_ids": [task.instance_id],
                    }
                ),
                encoding="utf-8",
            )
            evaluator = SWERebenchEvaluator(
                EvaluationConfig(
                    fork_path=Path(EVALUATOR_FORK),
                    predictions_path=predictions,
                    dataset_name=str(Path(REAL_TASK).expanduser()),
                    split="test",
                    run_id="real-integration",
                    max_workers=1,
                    namespace=os.environ.get("SWE_REBENCH_NAMESPACE", "swerebench"),
                    instance_ids=(task.instance_id,),
                    report_dir=root / "evaluation",
                    inference_metadata=metadata,
                )
            )
            self.assertIsNone(evaluator.run(check_only=True))


if __name__ == "__main__":
    unittest.main()
