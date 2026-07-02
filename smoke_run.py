"""Pre-flight smoke test: build + run EVERY config on a tiny corpus + 1 prompt.

Goal: confirm every config actually launches end-to-end (components build, the
filter / generator runs — including the LLM filters and the generators, the
retriever + solver answer) BEFORE committing to the multi-day full run. NOT for
statistics — just "does it run without crashing".

To stay fast it limits each config's corpus to `--max-docs` documents (via
LocalDB.max_docs) and builds a throwaway index under `--vdb` (data/vdb_smoke),
removed afterwards, so the real data/vdb_exp5 is never touched.

RUN ON THE SERVER (needs embedder + LLM + the DBs; build dirty DB first):
  python smoke_run.py -c test_configs_experimental_5 --max-docs 40
"""

import argparse
import glob
import os
import shutil
import traceback
from pathlib import Path

# Embedder + LLM are served locally — never reach out to huggingface.co.
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

from src.pipelines.configs import ConfigLoader
from src.pipelines.pipeline_builder import PipelineBuilder

SAMPLE_PROMPT = (
    "Problem:\nGiven a 2D numpy array a, compute the mean of each column.\n"
    "A:\n<code>\n"
)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-c", "--config", required=True, help="configs dir")
    ap.add_argument("--max-docs", type=int, default=15)
    ap.add_argument("--max-tail", type=int, default=10,
                    help="also load this many tail docs (off-topic junk sits at the end) "
                         "so the F3 topic-LLM path is exercised")
    ap.add_argument("--gen-limit", type=int, default=5,
                    help="cap dataset-driven generators (paraphrase / code2doc, which "
                         "read the full ds1000 and ignore --max-docs) to this many tasks; "
                         "harmless for corpus-driven generators (codeeval / code2task)")
    ap.add_argument("--vdb", default="data/vdb_smoke")
    args = ap.parse_args()

    if os.path.isdir(args.vdb):
        shutil.rmtree(args.vdb, ignore_errors=True)

    files = sorted(glob.glob(os.path.join(args.config, "**", "*.yaml"), recursive=True))
    print(f"smoke: {len(files)} configs, max_docs={args.max_docs} (+tail {args.max_tail})\n",
          flush=True)
    ok = fail = 0
    for i, f in enumerate(files, 1):
        stem = Path(f).stem
        print(f"[{i}/{len(files)}] building & running {stem} ...", flush=True)
        try:
            cfg = ConfigLoader().load_from_yaml(path_to_cfg=f)
            db = cfg.components["data_base"]
            db.params["max_docs"] = args.max_docs
            db.params["max_docs_tail"] = args.max_tail
            db.params["path_to_vector_db"] = f"{args.vdb}/{stem}"

            # Dataset-driven generators (paraphrase/code2doc) read the full ds1000
            # and ignore max_docs -> cap them so the smoke stays fast. The extra
            # 'limit' key is ignored by generators without such a param.
            gen = cfg.components.get("generator")
            if gen is not None and not isinstance(gen, list):
                gen.params["limit"] = args.gen_limit

            pipe = PipelineBuilder().build(cfg)        # runs filter/generator on max_docs
            ans = pipe.run(SAMPLE_PROMPT)              # runs retriever + solver
            try:
                pipe.close()
            except Exception:                          # noqa: BLE001
                pass

            empty = ans is None or not str(ans).strip()
            print(f"    [{'EMPTY' if empty else 'OK'}] {stem}  (answer {len(str(ans or ''))} chars)",
                  flush=True)
            ok += 1
        except Exception as e:                         # noqa: BLE001
            print(f"    [FAIL] {stem}: {type(e).__name__}: {e}", flush=True)
            traceback.print_exc()
            fail += 1

    if os.path.isdir(args.vdb):
        shutil.rmtree(args.vdb, ignore_errors=True)
    print(f"\nsmoke: {ok} launched OK / {fail} FAILED  (of {len(files)} configs)")
    print("если все OK — можно запускать полный run_ds1000.py")


if __name__ == "__main__":
    main()
