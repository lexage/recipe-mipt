"""Seed each exp16 vector index from the base one, so only new docs get embedded.

Every exp16 corpus is the base corpus plus/minus a handful of the 197 added
documents. `RecursiveChunker` derives a chunk id as f"{doc.id}_{i}", which
depends on the document alone — deleting OTHER documents cannot change it. And
`QdrantDocsAdapter.add` skips chunks whose id is already indexed. So a copy of
the base index is a valid warm start: the run embeds only the ~200 documents
that variant adds, instead of all ~60 000 chunks again.

Build the base index once (any config pointing at docs_base.db, e.g.
e0_reference/e0_baseline.yaml), then run this script before the rest.

    python run_ds1000.py -c test_configs_experimental_16/e0_reference --log-chunks
    python scripts/warm_start_exp16_vdbs.py

NOT applicable to the e6 corpora: inlining examples REWRITES documents.content,
so chunk "3776_2" holds different text there. Reusing the base index would keep
the stale vector and silently invalidate the experiment. Those two indexes are
built from scratch and this script refuses to touch them.
"""

import argparse
import os
import shutil


# variant slug -> vector dir suffix used in the configs
WARM_STARTABLE = [
    "aug_full",
    "only_guide", "only_migr", "only_howto",
    "minus_guide", "minus_migr", "minus_howto",
    "minus_t1", "minus_t1_t2",
    "aug_train_only",
]

# Same corpus as the base -> no copy needed, the configs point straight at it.
SHARES_BASE_INDEX = ["guide_prompt", "guide_prompt_verbatim"]

# Rewritten documents -> must be embedded from scratch.
MUST_BE_FRESH = ["base_inlined", "aug_inlined"]


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--base-slug", default="base")
    parser.add_argument("--prefix", default="vdb_exp16_")
    parser.add_argument("--suffix", default="_rc_1000")
    parser.add_argument("--force", action="store_true",
                        help="Overwrite a variant index that already exists.")
    args = parser.parse_args()

    def path_for(slug: str) -> str:
        return os.path.join(args.data_dir, f"{args.prefix}{slug}{args.suffix}")

    base_dir = path_for(args.base_slug)
    if not os.path.isdir(base_dir):
        raise SystemExit(
            f"base index not found: {base_dir}\n"
            f"Build it first:  python run_ds1000.py "
            f"-c test_configs_experimental_16/e0_reference --log-chunks"
        )

    size_mb = sum(
        os.path.getsize(os.path.join(root, name))
        for root, _, names in os.walk(base_dir) for name in names
    ) / (1024 * 1024)
    print(f"- base index: {base_dir} ({size_mb:.0f} MB)")

    for slug in WARM_STARTABLE:
        target = path_for(slug)
        if os.path.isdir(target):
            if not args.force:
                print(f"  skip   {slug:<16} (already exists, use --force)")
                continue
            shutil.rmtree(target)
        shutil.copytree(base_dir, target)
        print(f"  seeded {slug:<16} -> {target}")

    print("- share the base index, nothing to copy:")
    for slug in SHARES_BASE_INDEX:
        print(f"    {slug}")
    print("- must be embedded from scratch (documents rewritten):")
    for slug in MUST_BE_FRESH:
        print(f"    {slug}")


if __name__ == "__main__":
    main()
