"""Where things are: recipe-mipt, config.toml, paths inside a run directory."""

import os
import sys
import tomllib


def recipe_root() -> str:
    root = os.path.abspath(os.environ.get("RECIPE_ROOT") or os.getcwd())
    if not os.path.isfile(os.path.join(root, "run_ds1000.py")):
        raise SystemExit(f"не найден recipe-mipt: {root} (задайте RECIPE_ROOT)")
    if root not in sys.path:
        sys.path.insert(0, root)
    return root


def load_config(path: str) -> dict:
    with open(path, "rb") as handle:
        return tomllib.load(handle)


def repo_path(root: str, path: str) -> str:
    """Absolute path; relative paths are taken from the recipe-mipt root."""
    return path if os.path.isabs(path) else os.path.join(root, path)


def stem(path: str) -> str:
    return os.path.splitext(os.path.basename(path))[0]
