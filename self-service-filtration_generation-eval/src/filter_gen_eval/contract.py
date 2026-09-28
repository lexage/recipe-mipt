"""Check of the built-in input validation (ТЗ 3.5.5.1).

Feeds the component's standalone entry points a valid input and a series of
invalid ones and checks that every invalid input is refused in a controlled
way: exit code 2, a JSON answer with ``status: error`` and
``error.type: invalid_input``, and no traceback. The valid input must be
processed normally (exit code 0, ``status: ok``).

Inputs checked:
  - filtration (filter_corpus.py): the JSON corpus — syntax, structure, types,
    ranges, admissible characters, references, empty corpus — and the filter's
    parameters in the config;
  - generation (generate_rules.py): the generator's parameters in the config.
    These are rejected before any call to the model, so no model is needed;
    the valid generation input is exercised by run-rules-generation.sh.
"""

import copy
import gzip
import json
import os
import subprocess
import sys

import yaml

SMALL_CORPUS_DOCS = 200


def _small_corpus(corpus: dict) -> dict:
    """A valid corpus of the first documents, with their examples."""
    small = copy.deepcopy(corpus)
    documents = small["tables"]["documents"]["rows"][:SMALL_CORPUS_DOCS]
    keep = {row[0] for row in documents}
    small["tables"]["documents"]["rows"] = documents
    small["tables"]["examples"]["rows"] = [
        row for row in small["tables"]["examples"]["rows"] if row[1] in keep]
    small.pop("meta", None)
    return small


def _write_json(path: str, obj) -> None:
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(obj, handle, ensure_ascii=False)


def _write_bytes(path: str, data: bytes) -> None:
    with open(path, "wb") as handle:
        handle.write(data)


def _mutated(base: dict, change) -> dict:
    corpus = copy.deepcopy(base)
    change(corpus)
    return corpus


def _config_with(source: str, target: str, component: str, change) -> None:
    with open(source, encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    change(config["components"][component].setdefault("params", {}), config)
    with open(target, "w", encoding="utf-8") as handle:
        yaml.safe_dump(config, handle, sort_keys=False, allow_unicode=True)


def build_cases(root: str, config: dict, work: str) -> list:
    from src.db.json_corpus import read_json

    filtration = config["experiments"]["filtration"]
    filter_config = os.path.join(root, filtration["filter_config"])
    rules_config = os.path.join(root, config["experiments"]["generation"]["rules_config"])
    base = _small_corpus(read_json(os.path.join(root, filtration["corpus"])))
    docs = lambda c: c["tables"]["documents"]["rows"]  # noqa: E731

    def corpus_file(name, obj=None, raw: bytes = None):
        path = os.path.join(work, name)
        if raw is not None:
            _write_bytes(path, raw)
        else:
            _write_json(path, obj)
        return path

    def filter_case(key, title, path, expect_ok=False, cfg=None):
        out = os.path.join(work, f"out-{key}", "filtered.json")
        return {"key": key, "title": title, "component": "filtration",
                "cmd": [sys.executable, os.path.join(root, "filter_corpus.py"),
                        "-c", cfg or filter_config, "-i", path, "-o", out,
                        "--report", os.path.join(work, f"out-{key}", "report.json")],
                "expect_ok": expect_ok}

    def empty_corpus(c):
        c["tables"]["documents"]["rows"] = []
        c["tables"]["examples"]["rows"] = []

    cases = [
        filter_case("valid", "корректный корпус (200 документов)",
                    corpus_file("valid.json", base), expect_ok=True),
        filter_case("invalid_json", "некорректный JSON (обрыв файла)",
                    corpus_file("invalid.json", raw=json.dumps(base).encode()[:5000])),
        filter_case("empty_file", "пустой файл", corpus_file("empty.json", raw=b"")),
        filter_case("not_utf8", "файл не в UTF-8", corpus_file("latin.json", raw=b"\xff\xfe{\x00}")),
        filter_case("broken_gzip", "повреждённый gzip-архив",
                    corpus_file("broken.json.gz", raw=gzip.compress(b"{}")[:12])),
        filter_case("not_object", "верхний уровень не объект", corpus_file("list.json", [1, 2, 3])),
        filter_case("missing_field", "нет обязательного поля tables",
                    corpus_file("no_tables.json", _mutated(base, lambda c: c.pop("tables")))),
        filter_case("missing_table", "нет таблицы documents",
                    corpus_file("no_documents.json",
                                _mutated(base, lambda c: c["tables"].pop("documents")))),
        filter_case("unknown_format", "чужой формат (поле format)",
                    corpus_file("format.json", _mutated(base, lambda c: c.update(format="other")))),
        filter_case("row_length", "строка таблицы короче схемы",
                    corpus_file("short_row.json", _mutated(base, lambda c: docs(c)[0].pop()))),
        filter_case("wrong_type", "текст документа — число, а не строка",
                    corpus_file("type.json", _mutated(base, lambda c: docs(c)[0].__setitem__(3, 12345)))),
        filter_case("out_of_range", "отрицательный id документа",
                    corpus_file("range.json", _mutated(base, lambda c: docs(c)[0].__setitem__(0, -5)))),
        filter_case("duplicate_id", "повтор id документа",
                    corpus_file("dup.json", _mutated(base, lambda c: docs(c)[1].__setitem__(0, docs(c)[0][0])))),
        filter_case("forbidden_char", "недопустимый символ (U+0000) в тексте",
                    corpus_file("nul.json", _mutated(base, lambda c: docs(c)[0].__setitem__(3, "a\u0000b")))),
        filter_case("dangling_ref", "ссылка на несуществующую секцию",
                    corpus_file("ref.json", _mutated(base, lambda c: docs(c)[0].__setitem__(1, 999999)))),
        filter_case("empty_corpus", "пустой корпус (ни одного документа)",
                    corpus_file("empty_corpus.json", _mutated(base, empty_corpus))),
    ]

    valid_path = os.path.join(work, "valid.json")
    bad_type_cfg = os.path.join(work, "filter_bad_type.yaml")
    _config_with(filter_config, bad_type_cfg, "document_filter",
                 lambda p, c: p.__setitem__("max_sig_len", "много"))
    unknown_cfg = os.path.join(work, "filter_unknown_param.yaml")
    _config_with(filter_config, unknown_cfg, "document_filter",
                 lambda p, c: p.__setitem__("threshold", 0.5))
    cases += [
        filter_case("filter_param_type", "параметр фильтра недопустимого типа",
                    valid_path, cfg=bad_type_cfg),
        filter_case("filter_param_unknown", "неизвестный параметр фильтра",
                    valid_path, cfg=unknown_cfg),
    ]

    def generation_case(key, title, change):
        cfg = os.path.join(work, f"rules_{key}.yaml")
        _config_with(rules_config, cfg, "rule_writer", change)
        return {"key": key, "title": title, "component": "generation",
                "cmd": [sys.executable, os.path.join(root, "generate_rules.py"),
                        "-c", cfg, "-o", os.path.join(work, f"out-{key}")],
                "expect_ok": False}

    cases += [
        generation_case("gen_mode", "генератор: неизвестный режим",
                        lambda p, c: p.__setitem__("mode", "all_at_once")),
        generation_case("gen_problem_set", "генератор: неизвестный набор проблем",
                        lambda p, c: p.__setitem__("problem_set", "everything")),
        generation_case("gen_n_rules", "генератор: число правил вне диапазона (0)",
                        lambda p, c: p.__setitem__("n_rules", 0)),
        generation_case("gen_temperature", "генератор: температура вне диапазона (5)",
                        lambda p, c: p.__setitem__("temperature", 5)),
        generation_case("gen_url", "генератор: адрес модели не http(s)",
                        lambda p, c: p.__setitem__("url", "localhost:7217")),
        generation_case("gen_missing", "генератор: нет обязательного параметра problem_set",
                        lambda p, c: p.pop("problem_set")),
    ]
    return cases


def run_case(case: dict, root: str) -> dict:
    proc = subprocess.run(case["cmd"], cwd=root, capture_output=True, text=True)
    try:
        answer = json.loads(proc.stdout)
    except json.JSONDecodeError:
        answer = {}
    traceback_seen = "Traceback" in proc.stderr
    error_type = (answer.get("error") or {}).get("type")
    if case["expect_ok"]:
        passed = proc.returncode == 0 and answer.get("status") == "ok" and not traceback_seen
        expected = "код 0, status ok"
    else:
        passed = (proc.returncode == 2 and answer.get("status") == "error"
                  and error_type == "invalid_input" and not traceback_seen)
        expected = "код 2, status error, invalid_input"
    return {
        "key": case["key"], "title": case["title"], "component": case["component"],
        "expected": expected,
        "exit_code": proc.returncode, "status": answer.get("status"),
        "error_type": error_type,
        "message": (answer.get("error") or {}).get("message", ""),
        "traceback": traceback_seen,
        "passed": passed,
    }


def run_contract(root: str, config: dict, run_dir: str) -> int:
    work = os.path.join(run_dir, "cases")
    os.makedirs(work, exist_ok=True)
    results = [run_case(case, root) for case in build_cases(root, config, work)]
    passed = all(r["passed"] for r in results)

    width = max(len(r["title"]) for r in results)
    for r in results:
        mark = "OK  " if r["passed"] else "FAIL"
        detail = r["message"] if r["message"] else f"status={r['status']}"
        print(f"{mark} {r['title']:<{width}}  код {r['exit_code']}  {detail}")
    report = {"status": "passed" if passed else "failed",
              "cases_total": len(results),
              "cases_passed": sum(r["passed"] for r in results),
              "cases": results}
    with open(os.path.join(run_dir, "input_contract_report.json"), "w", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    print(f"\nПроверок: {report['cases_total']}, пройдено: {report['cases_passed']}")
    print(report["status"])
    return 0 if passed else 1
