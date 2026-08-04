import importlib.util
import sys
import tempfile
import threading
import unittest
from pathlib import Path

# Load this stdlib-only module without importing optional dependencies exposed by
# src.tools.__init__, so its isolation tests also run in a minimal environment.
MODULE_PATH = (
    Path(__file__).parents[2] / "src" / "tools" / "repository_context.py"
)
SPEC = importlib.util.spec_from_file_location("repository_context_under_test", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
repository_context = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = repository_context
SPEC.loader.exec_module(repository_context)

RepositoryContext = repository_context.RepositoryContext
RepositoryContextError = repository_context.RepositoryContextError
RepositoryPathError = repository_context.RepositoryPathError
bind_repository_context = repository_context.bind_repository_context
get_repository_context = repository_context.get_repository_context
resolve_repository_path = repository_context.resolve_repository_path


class RepositoryContextTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root = Path(self.temp_dir.name) / "repository"
        self.root.mkdir()
        self.context = RepositoryContext(
            root=self.root,
            instance_id="owner__repo-123",
            base_commit="deadbeef",
        )

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def test_context_is_available_only_while_bound(self) -> None:
        with self.assertRaises(RepositoryContextError):
            get_repository_context()

        with bind_repository_context(self.context):
            self.assertIs(get_repository_context(), self.context)

        with self.assertRaises(RepositoryContextError):
            get_repository_context()

    def test_nested_binding_restores_previous_context(self) -> None:
        other_root = Path(self.temp_dir.name) / "other"
        other_root.mkdir()
        other = RepositoryContext(other_root, "owner__other-1", "cafebabe")

        with bind_repository_context(self.context):
            with bind_repository_context(other):
                self.assertIs(get_repository_context(), other)
            self.assertIs(get_repository_context(), self.context)

    def test_binding_is_isolated_between_threads(self) -> None:
        other_root = Path(self.temp_dir.name) / "other-thread"
        other_root.mkdir()
        other = RepositoryContext(other_root, "owner__thread-1", "cafebabe")
        barrier = threading.Barrier(2)
        observed = []

        def observe(context: RepositoryContext) -> None:
            with bind_repository_context(context):
                barrier.wait()
                observed.append(get_repository_context().instance_id)

        threads = [
            threading.Thread(target=observe, args=(self.context,)),
            threading.Thread(target=observe, args=(other,)),
        ]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()

        self.assertCountEqual(observed, [self.context.instance_id, other.instance_id])

    def test_resolves_existing_and_future_paths_inside_repository(self) -> None:
        source = self.root / "src" / "module.py"
        source.parent.mkdir()
        source.write_text("value = 1\n")

        with bind_repository_context(self.context):
            self.assertEqual(resolve_repository_path("src/module.py"), source)
            self.assertEqual(
                resolve_repository_path("src/future.py"), self.root / "src/future.py"
            )

    def test_rejects_absolute_parent_and_git_paths(self) -> None:
        with bind_repository_context(self.context):
            for path in (self.root / "file.py", "../outside.py", ".git/config"):
                with self.subTest(path=path):
                    with self.assertRaises(RepositoryPathError):
                        resolve_repository_path(path)

    def test_rejects_symlink_that_escapes_repository(self) -> None:
        outside = Path(self.temp_dir.name) / "outside"
        outside.mkdir()
        (self.root / "external").symlink_to(outside, target_is_directory=True)

        with bind_repository_context(self.context):
            with self.assertRaises(RepositoryPathError):
                resolve_repository_path("external/file.py")

    def test_context_requires_valid_values(self) -> None:
        with self.assertRaises(RepositoryContextError):
            RepositoryContext(self.root / "missing", "instance", "commit")
        with self.assertRaises(RepositoryContextError):
            RepositoryContext(self.root, "", "commit")
        with self.assertRaises(RepositoryContextError):
            RepositoryContext(self.root, "instance", "")


if __name__ == "__main__":
    unittest.main()
