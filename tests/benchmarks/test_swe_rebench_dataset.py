import json
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from src.benchmarks.swe_rebench import (
    DatasetSWERebench,
    SWERebenchDataError,
    SWERebenchTask,
)


def make_record(instance_id: str = "owner__repo-1") -> dict:
    return {
        "instance_id": instance_id,
        "repo": "owner/repo",
        "base_commit": "deadbeef",
        "problem_statement": "Fix the broken behavior.",
        "hints_text": "Inspect the parser.",
        "version": "1.0",
        "patch": "reference patch must not leak",
        "test_patch": "hidden tests must not leak",
        "FAIL_TO_PASS": ["test_hidden"],
        "PASS_TO_PASS": ["test_existing"],
    }


class SWERebenchTaskTests(unittest.TestCase):
    def test_from_dict_keeps_only_generator_fields(self) -> None:
        task = SWERebenchTask.from_dict(make_record())

        self.assertEqual(task.instance_id, "owner__repo-1")
        self.assertEqual(task.repo, "owner/repo")
        self.assertFalse(hasattr(task, "patch"))
        self.assertFalse(hasattr(task, "test_patch"))
        self.assertFalse(hasattr(task, "FAIL_TO_PASS"))
        self.assertFalse(hasattr(task, "version"))

    def test_required_fields_must_be_non_empty_strings(self) -> None:
        for field in ("instance_id", "repo", "base_commit", "problem_statement"):
            for invalid in (None, "", "   ", 123):
                record = make_record()
                record[field] = invalid
                with self.subTest(field=field, invalid=invalid):
                    with self.assertRaisesRegex(SWERebenchDataError, field):
                        SWERebenchTask.from_dict(record)

    def test_optional_fields_must_be_strings_or_null(self) -> None:
        record = make_record()
        record["hints_text"] = 2
        with self.assertRaisesRegex(SWERebenchDataError, "hints_text"):
            SWERebenchTask.from_dict(record)

    def test_official_image_and_safe_docker_run_args_are_retained(self) -> None:
        record = make_record()
        record.update(
            {
                "image_name": "registry/task@sha256:abc",
                "install_config": {
                    "docker_specs": {
                        "run_args": {
                            "cap_add": ["SYS_PTRACE"],
                            "privileged": True,
                            "volumes": ["/:/host"],
                        }
                    }
                },
            }
        )
        task = SWERebenchTask.from_dict(record)

        self.assertEqual(task.instance_image, "registry/task@sha256:abc")
        self.assertEqual(task.docker_run_args, {"cap_add": ["SYS_PTRACE"]})
        self.assertNotIn("privileged", task.docker_run_args)


class DatasetSWERebenchTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root = Path(self.temp_dir.name)
        self.records = [make_record(f"owner__repo-{number}") for number in range(4)]

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def test_collection_rejects_duplicate_instance_ids(self) -> None:
        with self.assertRaisesRegex(SWERebenchDataError, "Duplicate"):
            DatasetSWERebench([self.records[0], self.records[0]])

    def test_collection_supports_iteration_index_slice_and_id(self) -> None:
        dataset = DatasetSWERebench(self.records)

        self.assertEqual(len(dataset), 4)
        self.assertEqual(dataset[0].instance_id, "owner__repo-0")
        self.assertEqual(dataset[1:3][1].instance_id, "owner__repo-2")
        self.assertEqual(dataset["owner__repo-3"].base_commit, "deadbeef")
        self.assertEqual(
            [task.instance_id for task in dataset],
            [
                "owner__repo-0",
                "owner__repo-1",
                "owner__repo-2",
                "owner__repo-3",
            ],
        )
        with self.assertRaises(KeyError):
            _ = dataset["missing"]

    def test_loads_json_array_and_split_mapping(self) -> None:
        array_path = self.root / "array.json"
        array_path.write_text(json.dumps(self.records), encoding="utf-8")
        split_path = self.root / "splits.json"
        split_path.write_text(json.dumps({"test": self.records[:2]}), encoding="utf-8")

        self.assertEqual(len(DatasetSWERebench.from_json(array_path)), 4)
        self.assertEqual(len(DatasetSWERebench.from_json(split_path, split="test")), 2)
        with self.assertRaisesRegex(SWERebenchDataError, "no requested split"):
            DatasetSWERebench.from_json(split_path, split="dev")

    def test_loads_jsonl_and_reports_line_errors(self) -> None:
        jsonl_path = self.root / "tasks.jsonl"
        jsonl_path.write_text(
            "\n".join(json.dumps(record) for record in self.records), encoding="utf-8"
        )
        self.assertEqual(len(DatasetSWERebench.from_json(jsonl_path)), 4)

        jsonl_path.write_text(
            json.dumps(self.records[0]) + "\n{broken\n", encoding="utf-8"
        )
        with self.assertRaisesRegex(SWERebenchDataError, "line 2"):
            DatasetSWERebench.from_json(jsonl_path)

    def test_local_loader_rejects_missing_and_unsupported_files(self) -> None:
        with self.assertRaisesRegex(SWERebenchDataError, "does not exist"):
            DatasetSWERebench.load(self.root / "missing.jsonl")
        unsupported = self.root / "tasks.txt"
        unsupported.write_text("[]", encoding="utf-8")
        with self.assertRaisesRegex(SWERebenchDataError, ".json or .jsonl"):
            DatasetSWERebench.from_json(unsupported)

    def test_huggingface_loader_forwards_split_revision_and_cache(self) -> None:
        load_dataset = Mock(return_value=self.records)
        datasets_stub = types.ModuleType("datasets")
        datasets_stub.load_dataset = load_dataset

        with patch.dict(sys.modules, {"datasets": datasets_stub}):
            dataset = DatasetSWERebench.from_huggingface(
                "org/benchmark",
                split="dev",
                revision="abc123",
                cache_dir=self.root / "cache",
            )

        self.assertEqual(len(dataset), 4)
        load_dataset.assert_called_once_with(
            "org/benchmark",
            split="dev",
            revision="abc123",
            cache_dir=str(self.root / "cache"),
        )

    def test_select_filters_range_limit_and_preserves_dataset_order(self) -> None:
        dataset = DatasetSWERebench(self.records)
        selected = dataset.select(
            instance_ids=["owner__repo-3", "owner__repo-1"], limit=1
        )
        ranged = dataset.select(start=1, stop=4, limit=2)

        self.assertEqual([task.instance_id for task in selected], ["owner__repo-1"])
        self.assertEqual(
            [task.instance_id for task in ranged],
            ["owner__repo-1", "owner__repo-2"],
        )
        with self.assertRaisesRegex(SWERebenchDataError, "not found"):
            dataset.select(instance_ids=["missing"])
        with self.assertRaisesRegex(SWERebenchDataError, "duplicates"):
            dataset.select(instance_ids=["owner__repo-1", "owner__repo-1"])
        with self.assertRaisesRegex(SWERebenchDataError, "0 <= start"):
            dataset.select(start=-1)

    def test_reads_instance_id_file_and_rejects_duplicates(self) -> None:
        ids_path = self.root / "instances.txt"
        ids_path.write_text(
            "# smoke subset\nowner__repo-1\n\nowner__repo-3\n", encoding="utf-8"
        )
        self.assertEqual(
            DatasetSWERebench.read_instance_ids(ids_path),
            ["owner__repo-1", "owner__repo-3"],
        )

        ids_path.write_text("owner__repo-1\nowner__repo-1\n", encoding="utf-8")
        with self.assertRaisesRegex(SWERebenchDataError, "duplicate"):
            DatasetSWERebench.read_instance_ids(ids_path)


if __name__ == "__main__":
    unittest.main()
