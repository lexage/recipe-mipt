"""Local test of the prompt-optimization loop — no server, no LLM.

Exercises the real machinery (overrides applied at registration, candidate
plumbing over a pipeline, the rollout, the optimizer protocol) against a fake
pipeline whose behaviour depends on its prompt, so a candidate that improves
the prompt genuinely improves the score.

  python test_optimization.py
"""

import sys
import types

from typing import List


def _install_stubs():
    """The evaluator lazily imports the DS1000 package, which pulls pandas."""
    for name, build in (("pandas", _stub_pandas), ("tqdm", _stub_tqdm)):
        try:
            __import__(name)
        except ImportError:
            sys.modules[name] = build()


def _stub_pandas():
    module = types.ModuleType("pandas")
    module.DataFrame = object
    module.set_option = lambda *a, **k: None
    return module


def _stub_tqdm():
    module = types.ModuleType("tqdm")
    module.tqdm = lambda iterable=None, **kwargs: (iterable or [])
    return module


_install_stubs()

from src.agent_constructor.prompts import (                       # noqa: E402
    Prompt,
    active_overrides,
    clear_overrides,
    PromptDiscoverable,
)
from src.optimization import candidates as cand                   # noqa: E402
from src.optimization import split as split_mod                   # noqa: E402
from src.optimization.evaluator import PipelineEvaluator          # noqa: E402
from src.optimization.optimizers import OPTIMIZERS, get_optimizer  # noqa: E402


MAGIC = "ALWAYS ANSWER 42"
_failures: List[str] = []


def chk(name, ok, detail=""):
    print(f"[{'PASS' if ok else 'FAIL'}] {name}" + (f"  {detail}" if detail and not ok else ""))
    if not ok:
        _failures.append(name)


# ---------------------------------------------------------------------------
# A pipeline whose behaviour depends on a BUILD-TIME prompt (RagGuide-shaped)
# ---------------------------------------------------------------------------

class FakeGenerator(PromptDiscoverable):
    name = "fake_gen"

    def build(self) -> str:
        policy = self.prompt("policy", "Write documents about $lib.\n")
        self.prompt("tail", "### DOC\nTITLE:", optimizable=False)
        return policy


class FakePipeline:
    """Consumes its prompt at construction, exactly like SimplePipeline does."""

    builds = 0

    def __init__(self, vdb_path):
        FakePipeline.builds += 1
        self.vdb_path = vdb_path
        self.generator = FakeGenerator()
        self.policy = self.generator.build()
        self._components = {"generator": self.generator}
        self.closed = False

    def run(self, task):
        answer = "result = 42" if MAGIC in self.policy else "result = 0"
        return f"<code>{answer}</code>"

    def close(self):
        self.closed = True


CODE_CONTEXT = (
    "def test_execution(code):\n"
    "    ns = {}\n"
    "    exec(code, ns)\n"
    "    assert ns.get('result') == 42\n"
)


def local_scorer(task, answer):
    """In-process stand-in for ds1000_scorer.

    The real scorer spawns a subprocess; on Windows that re-imports the module
    chain, and the pandas/tqdm stubs installed above live only in this process.
    Same contract — (score, result_text, code).
    """
    code = (answer or "").split("</code>")[0].replace("<code>", "")
    namespace = {}
    try:
        exec(code, namespace)
    except Exception as exc:                              # noqa: BLE001
        return 0, f"failed: {exc}", code
    if namespace.get("result") == 42:
        return 1, "passed", code
    return 0, f"failed: result was {namespace.get('result')!r}", code


class FakeTask:
    def __init__(self, pid):
        self.prompt = f"task {pid}: produce result"
        self.code_context = CODE_CONTEXT
        self.metadata = {"problem_id": pid, "library": "Numpy"}


class FakeDataset:
    def __init__(self, n):
        self.items = [FakeTask(i) for i in range(n)]

    def __iter__(self):
        return iter(self.items)

    def __getitem__(self, pid):
        return self.items[pid]


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_split():
    split = split_mod.make_split(list(range(100)), n_train=10, seed=7)
    again = split_mod.make_split(list(reversed(range(100))), n_train=10, seed=7)
    chk("split is seeded and order-independent", split["train"] == again["train"])
    chk("split partitions the dataset",
        set(split["train"]).isdisjoint(split["eval"])
        and len(split["train"]) + len(split["eval"]) == 100)
    try:
        split_mod.make_split(list(range(5)), n_train=5)
        chk("split refuses to leave no eval tasks", False)
    except ValueError:
        chk("split refuses to leave no eval tasks", True)


def test_discovery():
    pipeline = FakePipeline("/tmp/none")
    prompts = cand.from_pipeline(pipeline)
    chk("prompts discovered through the pipeline",
        "fake_gen.policy" in prompts and "fake_gen.tail" in prompts,
        f"got={sorted(prompts)}")

    seed = cand.seed_candidate(prompts)
    chk("frozen prompt excluded from the candidate",
        "fake_gen.policy" in seed and "fake_gen.tail" not in seed)

    try:
        cand.apply(prompts, {"fake_gen.tail": "nope"})
        chk("applying to a frozen prompt raises", False)
    except KeyError:
        chk("applying to a frozen prompt raises", True)

    original = seed["fake_gen.policy"]
    with cand.applied(prompts, {"fake_gen.policy": "Other $lib text\n"}):
        chk("applied() mutates in place",
            prompts["fake_gen.policy"].text.startswith("Other"))
    chk("applied() restores", prompts["fake_gen.policy"].text == original)


def test_rollout_and_overrides(tmp):
    dataset = FakeDataset(6)
    evaluator = PipelineEvaluator(
        build_pipeline=FakePipeline,
        dataset=dataset,
        scratch_dir=tmp,
        num_workers=2,
        score_answer=local_scorer,
        progress=lambda message: None,
    )
    task_ids = [0, 1, 2, 3]

    seed = {"fake_gen.policy": "Write documents about $lib.\n"}
    base = evaluator.evaluate(seed, task_ids)
    chk("seed candidate scores 0", base.mean == 0.0, f"got {base.mean}")
    chk("feedback carries the execution error",
        any("FAILED" in text for text in base.feedback().values()))

    better = {"fake_gen.policy": f"Write documents about $lib. {MAGIC}\n"}
    improved = evaluator.evaluate(better, task_ids)
    chk("build-time override reaches the pipeline", improved.mean == 1.0,
        f"got {improved.mean}")
    chk("overrides cleared after the build", active_overrides() == {})
    chk("a pipeline is built per rollout", FakePipeline.builds >= 2)

    dropped = {"fake_gen.policy": "no placeholder here\n"}
    try:
        evaluator.evaluate(dropped, task_ids)
        chk("candidate dropping $lib is rejected", False)
    except ValueError as exc:
        chk("candidate dropping $lib is rejected", "lib" in str(exc))
    finally:
        clear_overrides()
    return evaluator, task_ids, seed


def test_optimizer(evaluator, task_ids, seed):
    calls = {"n": 0}

    def reflection_lm(prompt):
        calls["n"] += 1
        return f"Write excellent documents about $lib. {MAGIC}\n"

    optimizer = get_optimizer("random", reflection_lm=reflection_lm, seed=1)
    result = optimizer.optimize(seed=seed, score=evaluator.evaluate,
                                train_ids=task_ids, budget=3, log=lambda m: None)

    chk("optimizer improved the score", result.best_score > result.seed_score,
        f"{result.seed_score} -> {result.best_score}")
    chk("optimizer respected the budget", result.metric_calls <= 3,
        f"used {result.metric_calls}")
    chk("winner is the mutated candidate",
        MAGIC in result.best["fake_gen.policy"])
    chk("reflection model was consulted", calls["n"] >= 1)


def test_registry():
    chk("backends are registered", set(OPTIMIZERS) >= {"gepa", "random"},
        f"got={sorted(OPTIMIZERS)}")
    try:
        get_optimizer("nope", reflection_lm=lambda p: "")
        chk("unknown backend raises", False)
    except ValueError:
        chk("unknown backend raises", True)

    from src.optimization.optimizers.gepa_backend import check_installation
    report = check_installation()
    chk("gepa check reports cleanly when absent",
        report["installed"] or "pip install gepa" in report.get("error", ""),
        f"got={report}")


def main():
    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        print("=== split ===")
        test_split()
        print("\n=== discovery through the pipeline ===")
        test_discovery()
        print("\n=== rollout + build-time overrides ===")
        evaluator, task_ids, seed = test_rollout_and_overrides(tmp)
        print("\n=== optimizer protocol ===")
        test_optimizer(evaluator, task_ids, seed)
        print("\n=== backend registry ===")
        test_registry()

    print()
    if _failures:
        print(f"{len(_failures)} CHECK(S) FAILED: {_failures}")
        return 1
    print("all checks passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
