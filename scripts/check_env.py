"""Проверка окружения прогона: пакеты и цепочка импортов пайплайна.

Запускается из корня репозитория интерпретатором venv. Пакет, помеченный
MISSING, означает, что семейство задач DS-1000 не исполнится, а для
qdrant_client, chromadb и fastembed — что пайплайн вообще не соберётся.

Вторая часть импортирует ровно те модули, которые поднимает run_ds1000.py с
нашими конфигами: так недостающая зависимость находится за секунды, а не через
час ожидания прогона.
"""

import importlib
import os
import sys

REQUIRED = [
    "numpy", "pandas", "scipy", "sklearn", "torch", "tensorflow", "matplotlib",
    "openai", "transformers", "yaml", "networkx", "pydantic", "qdrant_client",
    "chromadb", "fastembed", "tqdm", "psutil", "datasets",
]

PIPELINE_MODULES = [
    "src.pipelines.pipeline_builder",
    "src.pipelines.templates",
    "src.db.docs_db",
    "src.agents.general",
    "src.generation.prompt_rules",
    "src.benchmarks.ds1000",
    "src.utils.token_tracker",
]


def check(names, header):
    print(header)
    missing = []
    for name in names:
        try:
            module = importlib.import_module(name)
        except Exception as exc:
            missing.append(name)
            print(f"  {name:<28} MISSING ({type(exc).__name__}: {exc})")
        else:
            version = getattr(module, "__version__", "")
            print(f"  {name:<28} {version or 'ok'}")
    return missing


def main() -> int:
    sys.path.insert(0, os.getcwd())
    missing = check(REQUIRED, "=== пакеты ===")
    missing += check(PIPELINE_MODULES, "=== модули пайплайна ===")

    if missing:
        print("\nне хватает: " + ", ".join(missing))
        return 1
    print("\nвсё на месте")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
