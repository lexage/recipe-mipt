"""The three runs of pk3: standalone filtration, standalone rule generation and
the pipeline itself (an experiment: baseline + method, or all experiments).

Each run calls the component's own entry point from recipe-mipt as a
subprocess — ``filter_corpus.py``, ``generate_rules.py``, ``run_ds1000.py`` —
so what is tested is exactly what the component ships.
"""

import json
import os
import subprocess
import sys

from filter_gen_eval.configs import prepare_config
from filter_gen_eval.manifest import finish_manifest, write_manifest
from filter_gen_eval.settings import EXPERIMENTS, repo_path, stem
from filter_gen_eval.summary import summarize

FILTERED_NAME = "filtered.json.gz"
FILTRATION_REPORT = "filtration_report.json"


def _log(message: str) -> None:
    print(f"[filter_gen_eval] {message}", file=sys.stderr, flush=True)


def _run(cmd, cwd) -> int:
    _log("запуск: " + " ".join(cmd))
    return subprocess.run(cmd, cwd=cwd).returncode


# ------------------------------------------------------------------ filtration
def run_filtration(root: str, config: dict, out_dir: str,
                   filter_config: str = None, corpus: str = None) -> tuple:
    """Standalone filtration into ``out_dir``; returns (exit code, report)."""
    experiment = config["experiments"]["filtration"]
    filter_config = repo_path(root, filter_config or experiment["filter_config"])
    corpus = repo_path(root, corpus or experiment["corpus"])
    os.makedirs(out_dir, exist_ok=True)
    report_path = os.path.join(out_dir, FILTRATION_REPORT)
    code = _run([sys.executable, os.path.join(root, "filter_corpus.py"),
                 "-c", filter_config, "-i", corpus,
                 "-o", os.path.join(out_dir, FILTERED_NAME),
                 "--report", report_path], cwd=root)
    report = None
    if os.path.isfile(report_path):
        with open(report_path, encoding="utf-8") as handle:
            report = json.load(handle)
    return code, report


def filtration(root: str, config: dict, run_dir: str,
               filter_config: str = None, corpus: str = None) -> int:
    write_manifest(run_dir, "filtration", root)
    code, report = run_filtration(root, config, run_dir, filter_config, corpus)
    finish_manifest(run_dir, "ok" if code == 0 else "error", exit_code=code)
    if report:
        p = report.get("p_success") or {}
        _log(f"Pусп = {p.get('value')} ({p.get('n_success')} из {p.get('n_total')}); "
             f"отчёт: {os.path.join(run_dir, FILTRATION_REPORT)}")
    return code


# ------------------------------------------------------------------ generation
def generation(root: str, config: dict, run_dir: str, rules_config: str = None) -> int:
    write_manifest(run_dir, "rules-generation", root)
    rules_config = repo_path(root, rules_config or config["experiments"]["generation"]["rules_config"])
    code = _run([sys.executable, os.path.join(root, "generate_rules.py"),
                 "-c", rules_config, "-o", run_dir], cwd=root)
    finish_manifest(run_dir, "ok" if code == 0 else "error", exit_code=code)
    return code


# ------------------------------------------------------------------ evaluation
def _materialize(path: str) -> None:
    # JSON -> SQLite заранее, чтобы преобразование не попадало во время
    # построения индекса ни у бейзлайна, ни у метода.
    from src.db.json_corpus import resolve_sqlite_path
    _log(f"подготовка SQLite из {path}")
    resolve_sqlite_path(path)


def evaluate(root: str, config: dict, run_dir: str, experiment: str,
             workers: int, limit: int = 0, log_chunks: bool = True,
             max_docs: int = 0) -> int:
    """One experiment (baseline + method) or ``all`` of them in one run.

    Configs of every selected experiment are copied into one folder and
    run_ds1000.py runs that folder once, in alphabetical order.
    """
    experiments = EXPERIMENTS if experiment == "all" else (experiment,)
    manifest = write_manifest(
        run_dir, f"evaluation-{experiment}", root,
        experiment=experiment, workers=workers, limit=limit, max_docs=max_docs,
        configs={name: {"baseline": config["experiments"][name]["baseline"],
                        "method": config["experiments"][name]["method"]}
                 for name in experiments})
    configs_dir = os.path.join(run_dir, "configs")

    stems = []
    for name in experiments:
        settings = config["experiments"][name]
        baseline, method = stem(settings["baseline"]), stem(settings["method"])
        if baseline >= method:
            # run_ds1000.py запускает конфиги папки по алфавиту: бейзлайн — первым.
            raise SystemExit(f"{name}: имя конфига бейзлайна должно идти по алфавиту "
                             "раньше имени конфига метода")
        stems += [baseline, method]
    if len(set(stems)) != len(stems):
        raise SystemExit("имена конфигов экспериментов совпадают: " + ", ".join(stems))

    changes = {}
    for name in experiments:
        code, changes[name] = _prepare_experiment(root, config, run_dir, name,
                                                  configs_dir, max_docs)
        if code != 0:
            finish_manifest(run_dir, "error", exit_code=code, failed_step="filtration")
            _log("отдельная фильтрация завершилась с ошибкой — прогоны пайплайна не запускаются")
            return code

    cmd = [sys.executable, os.path.join(root, "run_ds1000.py"),
           "-c", configs_dir, "-s", os.path.join(run_dir, "runs"), "-n", str(workers)]
    if log_chunks:
        cmd.append("--log-chunks")
    if limit:
        cmd += ["-l", str(limit)]
    code = _run(cmd, cwd=root)

    summary = summarize(run_dir, experiment, config, {**manifest, "config_changes": changes})
    finish_manifest(run_dir, "ok" if code == 0 and summary["complete"] else "error",
                    exit_code=code, config_changes=changes, verdict=summary["verdict"])
    with open(os.path.join(run_dir, "summary.md"), encoding="utf-8") as handle:
        print(handle.read())
    if code != 0:
        return code
    return 0 if summary["complete"] else 3


def _prepare_experiment(root: str, config: dict, run_dir: str, experiment: str,
                        configs_dir: str, max_docs: int) -> tuple:
    """Copies of the experiment's configs in ``configs_dir``; returns (code, changes)."""
    settings = config["experiments"][experiment]
    baseline_src = repo_path(root, settings["baseline"])
    method_src = repo_path(root, settings["method"])
    changes = {}

    if experiment == "filtration":
        filtration_dir = os.path.join(run_dir, "filtration")
        code, report = run_filtration(root, config, filtration_dir)
        if code != 0:
            return code, changes
        corpus = repo_path(root, settings["corpus"])
        filtered = os.path.join(filtration_dir, FILTERED_NAME)
        _materialize(corpus)
        _materialize(filtered)
        changes["baseline"] = prepare_config(
            baseline_src, os.path.join(configs_dir, stem(baseline_src) + ".yaml"),
            path_to_db=corpus,
            path_to_vector_db=os.path.join(run_dir, "vdb", stem(baseline_src)),
            max_docs=max_docs)
        changes["method"] = prepare_config(
            method_src, os.path.join(configs_dir, stem(method_src) + ".yaml"),
            path_to_db=filtered,
            path_to_vector_db=os.path.join(run_dir, "vdb", stem(method_src)),
            max_docs=max_docs)
    else:
        shared = settings.get("shared_index", True)

        def index(name: str) -> str:
            # Корпус у бейзлайна и метода один — индекс строится один раз.
            return os.path.join(run_dir, "vdb", "generation" if shared else name)

        _materialize(_config_db(baseline_src, root))
        changes["baseline"] = prepare_config(
            baseline_src, os.path.join(configs_dir, stem(baseline_src) + ".yaml"),
            path_to_vector_db=index(stem(baseline_src)), max_docs=max_docs)
        changes["method"] = prepare_config(
            method_src, os.path.join(configs_dir, stem(method_src) + ".yaml"),
            path_to_vector_db=index(stem(method_src)),
            rules_path=os.path.join(run_dir, "rules", stem(method_src) + ".jsonl"),
            max_docs=max_docs)
    return 0, changes


def _config_db(config_path: str, root: str) -> str:
    import yaml
    with open(config_path, encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    return repo_path(root, config["components"]["data_base"]["params"]["path_to_db"])
