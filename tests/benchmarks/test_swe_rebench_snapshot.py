import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from snapshot_swe_rebench import materialize_snapshot


class SWERebenchSnapshotTests(unittest.TestCase):
    def test_snapshot_preserves_evaluator_fields_and_records_digest(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "source.json"
            source.write_text(
                json.dumps(
                    {
                        "test": [
                            {
                                "instance_id": "owner__repo-1",
                                "repo": "owner/repo",
                                "base_commit": "deadbeef",
                                "problem_statement": "Fix it",
                                "patch": "gold patch",
                                "test_patch": "hidden tests",
                            },
                            {
                                "instance_id": "owner__repo-2",
                                "repo": "owner/repo",
                                "base_commit": "cafebabe",
                                "problem_statement": "Fix another issue",
                                "patch": "another gold patch",
                                "test_patch": "more hidden tests",
                            },
                        ]
                    }
                ),
                encoding="utf-8",
            )
            output = root / "pinned.jsonl"

            snapshot, metadata_path = materialize_snapshot(
                dataset=str(source),
                split="test",
                output=output,
                instance_ids=["owner__repo-2"],
            )

            record = json.loads(snapshot.read_text(encoding="utf-8"))
            self.assertEqual(record["instance_id"], "owner__repo-2")
            self.assertEqual(record["patch"], "another gold patch")
            self.assertEqual(record["test_patch"], "more hidden tests")
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            self.assertEqual(metadata["instance_count"], 1)
            self.assertEqual(
                metadata["snapshot_sha256"],
                hashlib.sha256(snapshot.read_bytes()).hexdigest(),
            )

    def test_snapshot_filters_created_at_with_half_open_utc_range(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "source.json"
            source.write_text(
                json.dumps(
                    {
                        "filtered": [
                            {
                                "instance_id": "owner__repo-lower",
                                "created_at": "2024-01-01 00:00:00",
                            },
                            {
                                "instance_id": "owner__repo-inside",
                                "created_at": "2024-06-15 12:30:00",
                            },
                            {
                                "instance_id": "owner__repo-upper",
                                "created_at": "2025-01-01 00:00:00",
                            },
                        ]
                    }
                ),
                encoding="utf-8",
            )

            snapshot, metadata_path = materialize_snapshot(
                dataset=str(source),
                split="filtered",
                output=root / "filtered.jsonl",
                created_at_from="2024-01-01T03:00:00+03:00",
                created_at_before="2025-01-01T00:00:00Z",
            )

            records = [
                json.loads(line)
                for line in snapshot.read_text(encoding="utf-8").splitlines()
            ]
            self.assertEqual(
                [record["instance_id"] for record in records],
                ["owner__repo-lower", "owner__repo-inside"],
            )
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            self.assertEqual(metadata["source_instance_count"], 3)
            self.assertEqual(metadata["created_at_filtered_count"], 2)
            self.assertEqual(metadata["instance_count"], 2)
            self.assertEqual(
                metadata["filters"]["created_at"],
                {
                    "from": "2024-01-01T00:00:00+00:00",
                    "before": "2025-01-01T00:00:00+00:00",
                },
            )

    def test_created_at_filter_is_applied_before_limit(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "source.jsonl"
            source.write_text(
                "\n".join(
                    json.dumps(record)
                    for record in (
                        {
                            "instance_id": "owner__repo-old",
                            "created_at": "2023-12-31 23:59:59",
                        },
                        {
                            "instance_id": "owner__repo-first-new",
                            "created_at": "2024-01-01 00:00:00",
                        },
                        {
                            "instance_id": "owner__repo-second-new",
                            "created_at": "2024-01-02 00:00:00",
                        },
                    )
                ),
                encoding="utf-8",
            )

            snapshot, _ = materialize_snapshot(
                dataset=str(source),
                split="filtered",
                output=root / "filtered.jsonl",
                created_at_from="2024-01-01",
                limit=1,
            )

            record = json.loads(snapshot.read_text(encoding="utf-8"))
            self.assertEqual(record["instance_id"], "owner__repo-first-new")

    def test_created_at_filter_rejects_invalid_bounds_and_record_values(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "source.jsonl"
            source.write_text(
                json.dumps(
                    {
                        "instance_id": "owner__repo-invalid",
                        "created_at": "not-a-date",
                    }
                ),
                encoding="utf-8",
            )

            with self.assertRaisesRegex(
                ValueError, "created_at_from < created_at_before"
            ):
                materialize_snapshot(
                    dataset=str(source),
                    split="filtered",
                    output=root / "invalid-bounds.jsonl",
                    created_at_from="2024-01-01",
                    created_at_before="2024-01-01",
                )

            with self.assertRaisesRegex(
                ValueError, "owner__repo-invalid.*invalid created_at"
            ):
                materialize_snapshot(
                    dataset=str(source),
                    split="filtered",
                    output=root / "invalid-record.jsonl",
                    created_at_from="2024-01-01",
                )

            source.write_text(
                json.dumps({"instance_id": "owner__repo-missing"}),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(
                ValueError, "owner__repo-missing.*invalid created_at"
            ):
                materialize_snapshot(
                    dataset=str(source),
                    split="filtered",
                    output=root / "missing-record.jsonl",
                    created_at_before="2025-01-01",
                )

    def test_remote_dataset_requires_immutable_revision(self):
        with tempfile.TemporaryDirectory() as temporary:
            with self.assertRaisesRegex(ValueError, "dataset-revision"):
                materialize_snapshot(
                    dataset="nebius/SWE-rebench",
                    split="test",
                    output=Path(temporary) / "snapshot.jsonl",
                )


if __name__ == "__main__":
    unittest.main()
