"""Where things are: the component, config.toml, paths inside a run directory."""

import os
import sys
import tomllib

# Эксперименты испытаний; режим all прогоняет их все одним запуском.
EXPERIMENTS = ("filtration", "generation")


def component_root() -> str:
    root = os.path.abspath(os.environ.get("COMPONENT_ROOT") or os.getcwd())
    if not os.path.isfile(os.path.join(root, "run_ds1000.py")):
        raise SystemExit(f"не найден компонент: {root} (задайте COMPONENT_ROOT)")
    if root not in sys.path:
        sys.path.insert(0, root)
    return root


def load_config(path: str) -> dict:
    with open(path, "rb") as handle:
        return tomllib.load(handle)


def repo_path(root: str, path: str) -> str:
    """Absolute path; relative paths are taken from the component root."""
    return path if os.path.isabs(path) else os.path.join(root, path)


def stem(path: str) -> str:
    return os.path.splitext(os.path.basename(path))[0]
