"""Optimize a pipeline's optimizable prompts against DS1000.

The pipeline is built from an existing YAML through `PipelineBuilder` — the
same path `run_ds1000.py` uses — and prompts are discovered from the built
pipeline. A rollout runs `pipeline.run(task)` and scores it with the real
DS1000 test, so a pipeline with several solvers and an aggregator is scored as
that pipeline, not as a reimplementation of it.

Only prompts marked `optimizable` enter the search; `### DOC` format contracts
stay frozen.

RUN ON THE SERVER (needs vLLM at :7215 and the embedder at :7216).

  python optimize_prompts.py --check
  python optimize_prompts.py -c test_configs_experimental_13/simple_example_gen_guides_v2.yaml \
      --optimizer random --budget 8 --n-train 20 --max-docs 150

Cost: prompts consumed at construction time (the generator's) require a
pipeline rebuild per rollout, so the config is patched down before every build
— `data_base.max_docs`, a per-rollout scratch `path_to_vector_db`, and
`generator.limit`. If every optimized prompt is consumed at query time, pass
`--reuse-pipeline` to build once and skip that.

Artifacts land in results/optimization/<timestamp>/:
  split.json         tasks the optimizer saw (feed to run_ds1000.py)
  seed_prompts.json  the starting candidate
  best_prompts.json  the winner (feed to run_ds1000.py --prompts)
  result.json        scores, history, budget spent
"""

import argparse
import json
import logging
import os
import shutil
import sys
import time

from datetime import datetime

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

logging.getLogger("openai").setLevel(logging.ERROR)
logging.getLogger("httpx").setLevel(logging.ERROR)

# Heavy imports (yaml, qdrant, the benchmark) live inside main() so that
# `--check` stays usable as a pre-flight even when the stack is broken —
# diagnosing a missing dependency should not require that dependency.


def component_params(config, name: str) -> dict:
    """Params dict(s) of a component — a YAML key may hold a list of configs."""
    entry = config.components.get(name)
    if entry is None:
        return {}
    if isinstance(entry, list):
        return entry[0].params if entry else {}
    return entry.params


def make_builder(config, args):
    """Returns build_pipeline(vdb_path) -> Pipeline.

    The config comes from `ConfigLoader.load_from_yaml` like everywhere else;
    each build gets a deep copy so the per-rollout patches (scratch vector-db
    path, corpus cap, generator call cap) never accumulate.
    """
    from src.pipelines.pipeline_builder import PipelineBuilder

    def patch(entry, updates: dict):
        for item in (entry if isinstance(entry, list) else [entry]):
            item.params.update(updates)

    def build(vdb_path: str):
        patched = config.model_copy(deep=True)

        data_base = patched.components.get("data_base")
        if data_base is not None:
            updates = {"path_to_vector_db": vdb_path}
            if args.max_docs:
                updates["max_docs"] = args.max_docs
            patch(data_base, updates)

        generator = patched.components.get("generator")
        if generator is not None and args.gen_limit:
            patch(generator, {"limit": args.gen_limit})

        return PipelineBuilder().build(patched)

    return build


def build_reflection_lm(config, args):
    """The LM that rewrites prompts — same vLLM endpoint as the solver."""
    from openai import OpenAI

    agent_params = component_params(config, "agent")
    url = args.reflection_url or agent_params.get("url")
    model = args.reflection_model or agent_params.get("model_name")
    client = OpenAI(base_url=url, api_key="vllm")

    def reflect(prompt: str) -> str:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content":
                 "You improve instructions given to another model. Reply with "
                 "the rewritten instruction only."},
                {"role": "user", "content": prompt},
            ],
            temperature=1.0,
            max_tokens=1200,
        )
        return response.choices[0].message.content or ""

    return reflect


def discover_prompts(cand, build, dataset, scratch: str, warm_task):
    """Build once to enumerate prompts, optionally warming query-time ones.

    Prompts register lazily, inside the method that uses them: a generator's
    register during construction, a solver's only when a task runs. One warm
    task makes the search space complete.
    """
    pipeline = build(os.path.join(scratch, "vdb_discovery"))
    try:
        if warm_task is not None:
            try:
                pipeline.run(warm_task.prompt)
            except Exception as exc:                      # noqa: BLE001
                print(f"  · warm-up task failed ({exc}); query-time prompts "
                      f"may be missing from the search space")
        prompts = cand.from_pipeline(pipeline)
        return cand.seed_candidate(prompts), sorted(
            name for name, p in prompts.items() if not p.optimizable
        )
    finally:
        pipeline.close()
        shutil.rmtree(os.path.join(scratch, "vdb_discovery"), ignore_errors=True)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Optimize a pipeline's optimizable prompts against DS1000.",
    )
    parser.add_argument("-c", "--config", help="Pipeline YAML to build from.")
    parser.add_argument("--check", action="store_true",
                        help="Report what the installed gepa exposes, then exit.")
    parser.add_argument("--optimizer", default="gepa",
                        help="Backend name (gepa, random).")
    parser.add_argument("--budget", type=int, default=20,
                        help="Max rollouts (metric calls).")
    parser.add_argument("--n-train", type=int, default=20,
                        help="DS1000 tasks the optimizer may see.")
    parser.add_argument("--max-docs", type=int, default=200,
                        help="Corpus cap per pipeline build (0 = full corpus).")
    parser.add_argument("--gen-limit", type=int, default=2,
                        help="Generator LLM calls per stage per build.")
    parser.add_argument("--reuse-pipeline", action="store_true",
                        help="Build once and mutate prompts in place. Only "
                             "correct if no optimized prompt is used at build "
                             "time (NOT the case for RagGuideGenerator).")
    parser.add_argument("--keep-scratch", action="store_true",
                        help="Keep per-rollout vector indices for inspection.")
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--dataset", default="data/ds1000/ds1000.jsonl.gz")
    parser.add_argument("--reflection-url")
    parser.add_argument("--reflection-model")
    parser.add_argument("-o", "--out", default="results/optimization")
    return parser.parse_args()


def main():
    args = parse_args()

    if args.check:
        from src.optimization.optimizers.gepa_backend import check_installation
        report = check_installation()
        print(json.dumps(report, indent=2))
        return 0 if report.get("installed") else 1

    if not args.config:
        raise SystemExit("-c/--config is required (or use --check)")

    from src.benchmarks.ds1000 import DatasetDS1000
    from src.optimization import candidates as cand
    from src.optimization import split as split_mod
    from src.optimization.evaluator import PipelineEvaluator
    from src.optimization.optimizers import get_optimizer
    from src.pipelines.configs import ConfigLoader

    out_dir = os.path.join(args.out, datetime.now().strftime("%Y%m%d_%H%M%S"))
    scratch = os.path.join(out_dir, "scratch")
    os.makedirs(scratch, exist_ok=True)
    print(f"- artifacts -> {out_dir}")

    config = ConfigLoader().load_from_yaml(path_to_cfg=args.config)
    build = make_builder(config, args)

    from src.benchmarks.ds1000 import DatasetDS1000

    dataset = DatasetDS1000(args.dataset)
    split = split_mod.make_split(
        [item.metadata.get("problem_id") for item in dataset],
        n_train=args.n_train, seed=args.seed,
    )
    split_mod.save_split(split, os.path.join(out_dir, "split.json"))
    print(f"- split: {len(split['train'])} train / {len(split['eval'])} held out")

    print("- discovering prompts (one pipeline build + one warm task)")
    seed, frozen = discover_prompts(
        cand, build, dataset, scratch, warm_task=dataset[split["train"][0]]
    )
    if not seed:
        raise SystemExit(
            "no optimizable prompts in this pipeline. Only components using "
            "PromptDiscoverable declare them — currently RagGuideGenerator."
        )
    print(f"- optimizing {len(seed)}: {sorted(seed)}")
    print(f"- frozen (untouched): {frozen}")
    with open(os.path.join(out_dir, "seed_prompts.json"), "w", encoding="utf-8") as f:
        json.dump(seed, f, indent=2, ensure_ascii=False)

    evaluator = PipelineEvaluator(
        build_pipeline=build,
        dataset=dataset,
        scratch_dir=scratch,
        num_workers=args.num_workers,
        reuse_pipeline=args.reuse_pipeline,
        keep_scratch=args.keep_scratch,
        progress=lambda message: print(f"  · {message}"),
    )
    optimizer = get_optimizer(
        args.optimizer,
        reflection_lm=build_reflection_lm(config, args),
        seed=args.seed,
    )

    started = time.time()
    try:
        result = optimizer.optimize(
            seed=seed,
            score=evaluator.evaluate,
            train_ids=split["train"],
            budget=args.budget,
            log=print,
        )
    finally:
        evaluator.close()
        if not args.keep_scratch:
            shutil.rmtree(scratch, ignore_errors=True)
    elapsed = time.time() - started

    with open(os.path.join(out_dir, "best_prompts.json"), "w", encoding="utf-8") as f:
        json.dump(result.best, f, indent=2, ensure_ascii=False)
    with open(os.path.join(out_dir, "result.json"), "w", encoding="utf-8") as f:
        json.dump({
            "backend": result.backend,
            "config": args.config,
            "seed_score": result.seed_score,
            "best_score": result.best_score,
            "improvement": result.improvement,
            "metric_calls": result.metric_calls,
            "budget": args.budget,
            "elapsed_s": round(elapsed, 1),
            "n_train": len(split["train"]),
            "max_docs": args.max_docs,
            "gen_limit": args.gen_limit,
            "reuse_pipeline": args.reuse_pipeline,
            "changed": sorted(k for k in result.best
                              if result.best[k] != seed.get(k)),
            "history": result.history,
        }, f, indent=2, ensure_ascii=False)

    print(f"\nseed {result.seed_score:.3f} -> best {result.best_score:.3f} "
          f"({result.improvement:+.3f}) in {result.metric_calls} rollouts, "
          f"{elapsed / 60:.1f} min")
    print("\nThe score above is on a shrunken corpus — measure the real effect "
          "on tasks the optimizer never saw:")
    print(f"  python run_ds1000.py -c <configs_dir> \\\n"
          f"      --prompts {out_dir}/best_prompts.json \\\n"
          f"      --exclude-split {out_dir}/split.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
