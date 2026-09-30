"""Сборка архива поставки pk3.zip из закоммиченного состояния репозитория.

    python3 tools/make_archive.py <корень репозитория> <путь к pk3.zip>

В архив попадает только то, что нужно для испытаний по ПМИ, — файлы HEAD
(git archive) из белого списка ниже: исходный код компонента, конфиги ПМИ,
исходные данные (JSON-корпус и DS-1000) и набор испытаний. Всё, что строят
скрипты испытаний (SQLite из JSON, векторные индексы, venv, кеш, результаты),
в архив не кладётся; незакоммиченные правки тоже не попадают. Раскладка:

    pk3.zip
    ├── services/components/filtration_generation/   компонент
    └── self-service-filtration_generation-eval/     скрипты и модули испытаний

Архив распаковывается в рабочий каталог испытаний — по ПМИ это общий каталог
/bmcp_lvm_fs/data/shared/self-service-filtration_generation (до распаковки в
нём только pk3.zip).
Сборка прерывается, если какой-то пункт белого списка не нашёлся в HEAD или
слово recipe встретилось в имени файла либо в тексте кода и документации.

Совместим с питоном 3.6 (питон узла alibaba).
"""

import io
import os
import subprocess
import sys
import tarfile
import time
import zipfile

COMPONENT_DIR = "services/components/filtration_generation"
EVAL_DIR = "self-service-filtration_generation-eval"

# Компонент: пути от корня репозитория; каталог — с косой чертой на конце.
COMPONENT_FILES = (
    "run_ds1000.py",
    "filter_corpus.py",
    "generate_rules.py",
    "src/",
    "pmi_configs/",
    "scripts/setup_runner_env.sh",
    "scripts/constraints.txt",
    "scripts/check_env.py",
    "scripts/svc-qwen36-7217.sbatch",
    "scripts/svc-embed-7216.sbatch",
    "db_scripts/corpus_json.py",
    "data/docs_database_examples.json.gz",
    "data/ds1000/ds1000.jsonl.gz",
)
# Внутри src/ — копия чужого бенчмарка SWE-bench, пайплайном не используется.
COMPONENT_EXCLUDE = ("src/benchmarks/SWE-bench/",)

# Испытания: пути от каталога испытаний.
EVAL_FILES = (
    "README.md",
    ".env.example",
    "config.example.toml",
    "scripts/",
    "src/filter_gen_eval/",
)
# Сборка архива — инструмент разработчика, в испытаниях не используется.
EVAL_EXCLUDE = ("scripts/make-pk3-archive.sh",)

# Исходные данные: текст документации и задач не меняется, слово recipe в нём
# (например, «Numerical Recipes») — часть данных; имена файлов проверяются.
DATA_FILES = ("data/docs_database_examples.json.gz", "data/ds1000/ds1000.jsonl.gz")

FORBIDDEN = b"recipe"


def _matches(path, patterns):
    return any(path == p or (p.endswith("/") and path.startswith(p)) for p in patterns)


def _select(name):
    """Путь файла в архиве и пункт белого списка, или (None, None)."""
    if name.startswith(EVAL_DIR + "/"):
        rel = name[len(EVAL_DIR) + 1:]
        if _matches(rel, EVAL_FILES) and not _matches(rel, EVAL_EXCLUDE):
            pattern = next(p for p in EVAL_FILES if _matches(rel, (p,)))
            return name, EVAL_DIR + "/" + pattern
        return None, None
    if _matches(name, COMPONENT_FILES) and not _matches(name, COMPONENT_EXCLUDE):
        pattern = next(p for p in COMPONENT_FILES if _matches(name, (p,)))
        return COMPONENT_DIR + "/" + name, pattern
    return None, None


def main(argv):
    if len(argv) != 3:
        print(__doc__)
        return 2
    repo, out = os.path.abspath(argv[1]), os.path.abspath(argv[2])
    head = subprocess.run(["git", "-C", repo, "rev-parse", "HEAD"],
                          stdout=subprocess.PIPE, check=True).stdout.decode().strip()
    branch = subprocess.run(["git", "-C", repo, "rev-parse", "--abbrev-ref", "HEAD"],
                            stdout=subprocess.PIPE, check=True).stdout.decode().strip()
    dirty = subprocess.run(["git", "-C", repo, "status", "--porcelain", "--untracked-files=no"],
                           stdout=subprocess.PIPE, check=True).stdout.decode().strip()
    if dirty:
        print("ВНИМАНИЕ: есть незакоммиченные правки — в архив они не попадут:\n" + dirty,
              file=sys.stderr)
    tar_bytes = subprocess.run(["git", "-C", repo, "archive", "--format=tar", "HEAD"],
                               stdout=subprocess.PIPE, check=True).stdout

    entries = []            # (путь в архиве, режим, mtime, данные)
    used = set()
    problems = []
    with tarfile.open(fileobj=io.BytesIO(tar_bytes)) as tar:
        for member in tar.getmembers():
            if not member.isfile():
                continue
            arcname, pattern = _select(member.name)
            if arcname is None:
                continue
            used.add(pattern)
            data = tar.extractfile(member).read()
            if FORBIDDEN in arcname.lower().encode("utf-8"):
                problems.append("recipe в имени: " + arcname)
            elif member.name not in DATA_FILES and FORBIDDEN in data.lower():
                problems.append("recipe в тексте: " + arcname)
            entries.append((arcname, member.mode, member.mtime, data))

    wanted = set(COMPONENT_FILES) | {EVAL_DIR + "/" + p for p in EVAL_FILES}
    for pattern in sorted(wanted - used):
        problems.append("нет в HEAD: " + pattern)
    if problems:
        print("архив не собран:\n  " + "\n  ".join(problems), file=sys.stderr)
        return 1

    os.makedirs(os.path.dirname(out), exist_ok=True)
    tmp = out + ".tmp"
    counts = {COMPONENT_DIR: 0, EVAL_DIR: 0}
    size = 0
    with zipfile.ZipFile(tmp, "w", zipfile.ZIP_DEFLATED) as archive:
        for arcname, mode, mtime, data in sorted(entries):
            info = zipfile.ZipInfo(arcname, date_time=time.localtime(mtime)[:6])
            info.external_attr = (mode & 0o777) << 16
            info.compress_type = zipfile.ZIP_DEFLATED
            archive.writestr(info, data)
            counts[EVAL_DIR if arcname.startswith(EVAL_DIR + "/") else COMPONENT_DIR] += 1
            size += len(data)
        # В архиве нет .git: коммит записывается в файл, его читают скрипты испытаний.
        info = zipfile.ZipInfo(COMPONENT_DIR + "/DEPLOYED_COMMIT", date_time=time.localtime()[:6])
        info.external_attr = 0o644 << 16
        info.compress_type = zipfile.ZIP_DEFLATED
        archive.writestr(info, "{}\n{}\nfalse\n".format(head, branch))
        archive.comment = ("pk3: компонент фильтрации и генерации, commit " + head).encode("utf-8")
    os.replace(tmp, out)
    print("архив: {}".format(out))
    print("commit: {}".format(head))
    print("файлов: {} {}, {} {}; распакованный объём {:.1f} МБ; архив {:.1f} МБ".format(
        COMPONENT_DIR, counts[COMPONENT_DIR] + 1, EVAL_DIR, counts[EVAL_DIR],
        size / 1e6, os.path.getsize(out) / 1e6))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
