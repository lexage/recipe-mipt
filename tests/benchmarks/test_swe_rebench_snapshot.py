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
