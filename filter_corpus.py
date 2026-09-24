"""Отдельный запуск метода фильтрации — без пайплайна и без моделей.

На вход — JSON-корпус (вся база документов), на выход — JSON-корпус того же
формата, в котором остались только документы, оставленные фильтром. Этот
файл подаётся в пайплайн как path_to_db вместо исходного, и шаг фильтрации в
пайплайне уже не нужен.

    python filter_corpus.py -c pmi_configs/filtration/f1_api_genre_inline.yaml \
        -o results/filtration/filtered.json.gz

Метод и его параметры берутся из того же конфига пайплайна, что и при
фильтрации внутри пайплайна (компонент components.document_filter), а
документы читаются тем же адаптером SQLite, что и в LocalDB. Сам метод не
меняется: скрипт только вызывает его apply().

Что считается:
  - решение по каждому документу: apply([документ]) отдельно для каждого, чтобы
    сбой на одном документе не скрывал остальные;
  - вызов apply() на всём корпусе — ровно как в пайплайне; его результат и
    записывается в выходной JSON;
  - Pусп (п. 3.5.4.1 ТЗ) = доля документов, по которым фильтр вынес решение без
    сбоя и которые корректно отражены в выходном JSON (оставленный — есть и
    совпадает с исходным побайтно, отброшенный — отсутствует);
  - проверка эквивалентности: документы, прочитанные из выходного JSON адаптером
    пайплайна, совпадают с тем, что пайплайн проиндексировал бы после фильтра.

Отчёт (JSON) печатается в stdout и пишется в --report.

Коды возврата: 0 — успех; 2 — входные данные или конфиг не прошли проверку;
3 — сбой при выполнении (метод упал на корпусе, проверка выхода не прошла);
1 — непредвиденная внутренняя ошибка.
"""

import argparse
import json
import logging
import os
import shutil
import sys
import tempfile
import time
import traceback

REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, REPO_ROOT)

from src.db.json_corpus import (  # noqa: E402
    CorpusFormatError,
    corpus_stats,
    corpus_to_sqlite,
    file_sha256,
    is_json_path,
    read_json,
    subset_documents,
    write_json,
)
from src.pipelines.standalone import (  # noqa: E402
    ConfigError,
    build_component,
    component_section,
    load_config,
)

EXIT_OK, EXIT_INTERNAL, EXIT_INVALID_INPUT, EXIT_RUNTIME = 0, 1, 2, 3
MAX_LISTED_FAILURES = 20

log = logging.getLogger("filter_corpus")


class RuntimeFailure(RuntimeError):
    """The method or the output check failed on valid input."""


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Отдельный запуск метода фильтрации: JSON-корпус -> JSON-корпус.")
    parser.add_argument("-c", "--config", required=True,
                        help="конфиг пайплайна с компонентом фильтра")
    parser.add_argument("-i", "--input",
                        help="входной JSON-корпус (по умолчанию path_to_db из data_base)")
    parser.add_argument("-o", "--output", required=True,
                        help="выходной JSON-корпус (.json или .json.gz)")
    parser.add_argument("--report",
                        help="куда записать отчёт (по умолчанию filtration_report.json "
                             "рядом с выходным корпусом)")
    parser.add_argument("--component", default="document_filter",
                        help="имя компонента фильтра в конфиге (по умолчанию document_filter)")
    return parser.parse_args(argv)


def _pct(before: int, after: int) -> float:
    return round(100.0 * (before - after) / before, 2) if before else 0.0


def _doc_signature(doc) -> tuple:
    return doc.id, doc.text, json.dumps(doc.metadata, sort_keys=True, ensure_ascii=False)


def _load_documents(sqlite_path: str):
    # Тот же вызов, что делает LocalDB.get_documents() при index_sources=[documents].
    from src.utils.adapters.sqlite_adapter import SQLiteDocsDBAdapter
    return SQLiteDocsDBAdapter(path_to_db=sqlite_path).get_docs()


def run(args) -> dict:
    timing = {}
    started = time.time()

    config = load_config(args.config)
    filter_type, filter_params = component_section(config, args.component)
    _, db_params = component_section(config, "data_base")
    if db_params.get("index_sources") not in (None, ["documents"]):
        raise ConfigError("отдельная фильтрация поддерживает только index_sources: "
                          "[documents] — как в конфигах ПМИ")
    if db_params.get("max_docs"):
        raise ConfigError("отдельная фильтрация не поддерживает max_docs")

    input_path = args.input or db_params.get("path_to_db")
    if not input_path:
        raise ConfigError("не задан входной корпус: нет --input и data_base.path_to_db")
    if not is_json_path(input_path):
        raise ConfigError(f"вход должен быть JSON-корпусом (.json или .json.gz): {input_path}")
    if not is_json_path(args.output):
        raise ConfigError(f"выход должен быть JSON-корпусом (.json или .json.gz): {args.output}")

    t0 = time.time()
    corpus = read_json(input_path)
    input_sha = file_sha256(input_path)
    timing["load_input"] = time.time() - t0

    method = build_component(filter_type, filter_params,
                             where=f"components.{args.component}.params")

    output_dir = os.path.dirname(os.path.abspath(args.output))
    os.makedirs(output_dir, exist_ok=True)
    workdir = tempfile.mkdtemp(prefix=".filter-work-", dir=output_dir)
    try:
        t0 = time.time()
        input_db = os.path.join(workdir, "input.db")
        corpus_to_sqlite(corpus, input_db)
        documents = _load_documents(input_db)
        timing["prepare_documents"] = time.time() - t0

        # Решение по каждому документу отдельно — для подсчёта Pусп.
        t0 = time.time()
        decisions, failures = {}, []
        for doc in documents:
            try:
                decisions[doc.id] = bool(method.apply([doc]))
            except Exception as exc:  # сбой на одном документе не останавливает остальные
                failures.append({"id": doc.id, "error": f"{type(exc).__name__}: {exc}"})
        timing["per_document"] = time.time() - t0

        # Вызов на всём корпусе — ровно как в пайплайне; он и определяет выход.
        t0 = time.time()
        try:
            kept_docs = method.apply(list(documents))
        except Exception as exc:
            raise RuntimeFailure(f"метод фильтрации упал на корпусе: "
                                 f"{type(exc).__name__}: {exc}") from exc
        timing["filter"] = time.time() - t0
        kept_ids = [doc.id for doc in kept_docs]
        batch_consistent = (set(kept_ids)
                            == {doc_id for doc_id, keep in decisions.items() if keep})

        t0 = time.time()
        meta = {
            "derived_from": {
                "sha256": input_sha,
                "filter": {"type": filter_type, "params": filter_params},
            },
        }
        write_json(subset_documents(corpus, kept_ids, meta=meta), args.output)
        timing["write_output"] = time.time() - t0

        # Проверка выхода: перечитываем записанный файл.
        t0 = time.time()
        written = read_json(args.output)
        written_rows = {row[0]: row for row in written["tables"]["documents"]["rows"]}
        kept_set = set(kept_ids)
        failed_ids = {f["id"] for f in failures}
        n_success = 0
        for row in corpus["tables"]["documents"]["rows"]:
            doc_id = row[0]
            if doc_id in failed_ids or doc_id not in decisions:
                continue
            if doc_id in kept_set:
                n_success += written_rows.get(doc_id) == row
            else:
                n_success += doc_id not in written_rows

        output_db = os.path.join(workdir, "output.db")
        corpus_to_sqlite(written, output_db)
        from_output = [_doc_signature(d) for d in _load_documents(output_db)]
        equivalent = from_output == [_doc_signature(d) for d in kept_docs]
        timing["verify"] = time.time() - t0
    finally:
        shutil.rmtree(workdir, ignore_errors=True)

    stats_in, stats_out = corpus_stats(corpus), corpus_stats(written)
    n_total = stats_in["documents"]
    timing["total"] = time.time() - started

    report = {
        "status": "ok",
        "component": "filtration",
        "method": {"type": filter_type, "params": filter_params},
        "config": args.config,
        "input": {"path": input_path, "sha256": input_sha, **stats_in},
        "output": {"path": args.output, "sha256": file_sha256(args.output), **stats_out},
        "decisions": {"kept": len(kept_ids), "dropped": n_total - len(kept_ids),
                      "failed": len(failures)},
        "p_success": {"n_total": n_total, "n_success": n_success,
                      "value": round(n_success / n_total, 6) if n_total else 0.0},
        "reduction_pct": {
            "documents": _pct(stats_in["documents"], stats_out["documents"]),
            "document_chars": _pct(stats_in["document_chars"], stats_out["document_chars"]),
        },
        "checks": {"batch_consistent": batch_consistent,
                   "equivalent_to_pipeline": equivalent},
        "timing_s": {key: round(value, 3) for key, value in timing.items()},
        "failures": failures[:MAX_LISTED_FAILURES],
    }
    if not equivalent:
        report["status"] = "error"
        report["error"] = {"type": "verification_failed",
                           "message": "документы из выходного JSON не совпадают с тем, "
                                      "что пайплайн проиндексировал бы после фильтра"}
    return report


def _emit(report: dict, path: str) -> None:
    text = json.dumps(report, ensure_ascii=False, indent=2)
    print(text)
    if path:
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        with open(path, "w", encoding="utf-8") as handle:
            handle.write(text + "\n")


def main(argv=None) -> int:
    logging.basicConfig(level=logging.INFO, stream=sys.stderr,
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    args = parse_args(argv)
    report_path = args.report or os.path.join(
        os.path.dirname(os.path.abspath(args.output)), "filtration_report.json")
    try:
        report = run(args)
        code = EXIT_OK if report["status"] == "ok" else EXIT_RUNTIME
    except (ConfigError, CorpusFormatError) as exc:
        report = {"status": "error", "component": "filtration",
                  "error": {"type": "invalid_input", "message": str(exc)}}
        code = EXIT_INVALID_INPUT
    except RuntimeFailure as exc:
        report = {"status": "error", "component": "filtration",
                  "error": {"type": "runtime_failure", "message": str(exc)}}
        code = EXIT_RUNTIME
    except Exception as exc:
        traceback.print_exc()
        report = {"status": "error", "component": "filtration",
                  "error": {"type": "internal_error",
                            "message": f"{type(exc).__name__}: {exc}"}}
        code = EXIT_INTERNAL
    _emit(report, report_path)
    return code


if __name__ == "__main__":
    raise SystemExit(main())
