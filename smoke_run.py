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
    ap.add_argument("--vdb", default="data/vdb_smoke")
    args = ap.parse_args()

    if os.path.isdir(args.vdb):
        shutil.rmtree(args.vdb, ignore_errors=True)

    files = sorted(glob.glob(os.path.join(args.config, "**", "*.yaml"), recursive=True))
    ok = fail = 0
    for f in files:
        stem = Path(f).stem
        try:
            cfg = ConfigLoader().load_from_yaml(path_to_cfg=f)
            db = cfg.components["data_base"]
            db.params["max_docs"] = args.max_docs
            db.params["max_docs_tail"] = args.max_tail
            db.params["path_to_vector_db"] = f"{args.vdb}/{stem}"

            pipe = PipelineBuilder().build(cfg)        # runs filter/generator on max_docs
            ans = pipe.run(SAMPLE_PROMPT)              # runs retriever + solver
            try:
                pipe.close()
            except Exception:                          # noqa: BLE001
                pass

            empty = ans is None or not str(ans).strip()
            print(f"[{'EMPTY' if empty else 'OK'}] {stem}  (answer {len(str(ans or ''))} chars)")
            ok += 1
        except Exception as e:                         # noqa: BLE001
            print(f"[FAIL] {stem}: {type(e).__name__}: {e}")
            traceback.print_exc()
            fail += 1

    if os.path.isdir(args.vdb):
        shutil.rmtree(args.vdb, ignore_errors=True)
    print(f"\nsmoke: {ok} launched OK / {fail} FAILED  (of {len(files)} configs)")
    print("если все OK — можно запускать полный run_ds1000.py")


if __name__ == "__main__":
    main()
