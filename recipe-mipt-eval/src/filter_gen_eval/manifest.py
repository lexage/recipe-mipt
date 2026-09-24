"""run_manifest.json — what was run, from which code, on which models."""

import json
import os
import sys

from datetime import datetime, timezone

MANIFEST_NAME = "run_manifest.json"


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def write_manifest(run_dir: str, kind: str, recipe_root: str, **extra) -> dict:
    """Start the manifest; the shell script passes stand details through env."""
    manifest = {
        "kind": kind,
        "started_utc": _now(),
        "finished_utc": None,
        "status": "running",
        "host": os.environ.get("RUN_HOST", ""),
        "command": os.environ.get("RUN_COMMAND", ""),
        "recipe_root": recipe_root,
        "eval_root": os.environ.get("EVAL_ROOT", ""),
        "git": {
            "commit": os.environ.get("GIT_COMMIT", ""),
            "branch": os.environ.get("GIT_BRANCH", ""),
            "dirty": os.environ.get("GIT_DIRTY", "") == "true",
        },
        "image": os.environ.get("IMG_ASLLM", ""),
        "models": {
            "llm": os.environ.get("LLM_MODEL_DIR", ""),
            "embedder": os.environ.get("EMBED_API_NAME", ""),
        },
        "python": sys.version.split()[0],
        **extra,
    }
    _save(run_dir, manifest)
    return manifest


def finish_manifest(run_dir: str, status: str, **extra) -> None:
    path = os.path.join(run_dir, MANIFEST_NAME)
    manifest = {}
    if os.path.isfile(path):
        with open(path, encoding="utf-8") as handle:
            manifest = json.load(handle)
    manifest.update(extra)
    manifest["status"] = status
    manifest["finished_utc"] = _now()
    _save(run_dir, manifest)


def _save(run_dir: str, manifest: dict) -> None:
    os.makedirs(run_dir, exist_ok=True)
    with open(os.path.join(run_dir, MANIFEST_NAME), "w", encoding="utf-8") as handle:
        json.dump(manifest, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
