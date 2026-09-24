"""Отдельный запуск метода генерации правил — без пайплайна.

Метод пишет блок правил для системного промпта решателя (компонент
PROMPT_RULE_GENERATOR, src/generation/prompt_rules.py). В пайплайне блок
дописывается к системному промпту сразу после сборки индекса; этот скрипт
вызывает тот же компонент с теми же параметрами из того же конфига, но сам по
себе, и сохраняет результат в JSON.

    python generate_rules.py -c pmi_configs/generation/g1_rules.yaml -o results/rules

Нужна поднятая модель, которая пишет правила (url из конфига). Правила
генерируются заново всегда (rebuild: True), журнал всех вызовов метода пишется
в <out>/rules_dump.jsonl, итог — в <out>/rules.json (он же печатается в stdout).

Pусп (п. 3.5.4.1 ТЗ) = доля вызовов генератора, на которые получен разобранный
ответ: новое правило (accepted) или ссылка на уже принятое правило, которое
покрывает проблему (covered). Ответ, из которого правило не извлеклось
(rejected), и вызов, не состоявшийся из-за сбоя, — неуспех. В режиме batch
вызов один, и он успешен, если из ответа извлеклось хотя бы одно правило.

Коды возврата: 0 — успех; 2 — конфиг не прошёл проверку; 3 — сбой при
выполнении (модель недоступна, ни одного правила); 1 — внутренняя ошибка.
"""

import argparse
import json
import logging
import os
import sys
import time
import traceback

REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, REPO_ROOT)

from src.pipelines.standalone import (  # noqa: E402
    ConfigError,
    build_component,
    component_section,
    load_config,
)

EXIT_OK, EXIT_INTERNAL, EXIT_INVALID_INPUT, EXIT_RUNTIME = 0, 1, 2, 3
GENERATOR_TYPE = "PROMPT_RULE_GENERATOR"
DUMP_NAME = "rules_dump.jsonl"
RESULT_NAME = "rules.json"

log = logging.getLogger("generate_rules")


def _number(value) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _integer(value) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


# Допустимые значения параметров генератора: (проверка, описание правила).
PARAM_RULES = {
    "url": (lambda v: isinstance(v, str) and v.startswith(("http://", "https://")),
            "адрес вида http(s)://хост:порт/v1"),
    "model_name": (lambda v: isinstance(v, str) and bool(v.strip()), "непустая строка"),
    "mode": (lambda v: v in ("per_problem", "batch"), "per_problem или batch"),
    "n_rules": (lambda v: _integer(v) and 1 <= v <= 100, "целое от 1 до 100"),
    "max_prompt_tokens": (lambda v: _integer(v) and 1 <= v <= 100_000, "целое от 1 до 100000"),
    "temperature": (lambda v: _number(v) and 0 <= v <= 2, "число от 0 до 2"),
    "seed": (lambda v: _integer(v) and 0 <= v <= 2 ** 31 - 1, "целое от 0 до 2147483647"),
    "max_tokens": (lambda v: _integer(v) and 1 <= v <= 32_768, "целое от 1 до 32768"),
    "enable_thinking": (lambda v: v is None or isinstance(v, bool), "true, false или null"),
    "name": (lambda v: isinstance(v, str) and bool(v.strip()), "непустая строка"),
    "rules_path": (lambda v: isinstance(v, str), "строка"),
    "rebuild": (lambda v: isinstance(v, bool), "true или false"),
}
REQUIRED_PARAMS = ("url", "model_name", "mode", "problem_set")


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Отдельный запуск метода генерации правил системного промпта.")
    parser.add_argument("-c", "--config", required=True,
                        help="конфиг пайплайна с компонентом генератора правил")
    parser.add_argument("-o", "--out-dir", required=True,
                        help="каталог для rules.json и журнала вызовов")
    parser.add_argument("--component", default="rule_writer",
                        help="имя компонента генератора в конфиге (по умолчанию rule_writer)")
    return parser.parse_args(argv)


def check_generator_params(params: dict, where: str) -> None:
    from src.generation.prompt_rules import PROBLEM_SETS
    for name in REQUIRED_PARAMS:
        if name not in params:
            raise ConfigError(f"{where}: не задан обязательный параметр {name}")
    for name, value in params.items():
        if name == "problem_set":
            if value not in PROBLEM_SETS:
                raise ConfigError(f"{where}.problem_set: {value!r}, допустимо: "
                                  f"{', '.join(sorted(PROBLEM_SETS))}")
            continue
        rule = PARAM_RULES.get(name)
        if rule and not rule[0](value):
            raise ConfigError(f"{where}.{name}: {value!r} — ожидается {rule[1]}")


def count_calls(records, mode: str, n_problems: int, completed: bool) -> dict:
    """Split the writing calls by outcome, from the generator's call records.

    ``records`` are the generator's own records (or the lines of its dump);
    ``completed`` is False when the run stopped before the last call.
    """
    calls = [r for r in records if r.get("record") == "call"]
    if mode == "batch":
        accepted = sum(1 for r in calls if r.get("status") == "accepted")
        total, success = 1, int(accepted > 0)
        return {"total": total, "success": success, "accepted_rules": accepted,
                "covered": 0, "rejected": total - success if completed else 0,
                "not_completed": 0 if completed else total - success}
    by_status = {"accepted": 0, "covered": 0, "rejected": 0}
    for record in calls:
        by_status[record.get("status")] = by_status.get(record.get("status"), 0) + 1
    success = by_status["accepted"] + by_status["covered"]
    return {"total": n_problems, "success": success, "accepted_rules": by_status["accepted"],
            "covered": by_status["covered"], "rejected": by_status["rejected"],
            "not_completed": max(0, n_problems - len(calls))}


def read_dump(dump_path: str) -> list:
    """All records of a generator dump (jsonl); unreadable lines are skipped."""
    records = []
    if os.path.isfile(dump_path):
        with open(dump_path, encoding="utf-8") as handle:
            for line in handle:
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    return records


def _final_record(dump_path: str) -> dict:
    finals = [r for r in read_dump(dump_path) if r.get("record") == "final"]
    return finals[-1] if finals else {}


def run(args) -> dict:
    started = time.time()
    config = load_config(args.config)
    type_name, params = component_section(config, args.component)
    where = f"components.{args.component}.params"
    if type_name != GENERATOR_TYPE:
        raise ConfigError(f"components.{args.component}: ожидался {GENERATOR_TYPE}, "
                          f"указан {type_name}")
    check_generator_params(params, where)

    out_dir = os.path.abspath(args.out_dir)
    os.makedirs(out_dir, exist_ok=True)
    dump_path = os.path.join(out_dir, DUMP_NAME)
    run_params = dict(params, rules_path=dump_path, rebuild=True)
    generator = build_component(type_name, run_params, where=where)

    error = None
    text = ""
    try:
        text = generator.rules()
    except ValueError as exc:            # метод отказался: ни одного годного правила
        error = {"type": "no_rules", "message": str(exc)}
    except Exception as exc:             # модель недоступна, ошибка API и т. п.
        error = {"type": "runtime_failure", "message": f"{type(exc).__name__}: {exc}"}

    calls = count_calls(generator._records, generator.mode, len(generator.problems),
                        completed=error is None)
    final = _final_record(dump_path) if error is None else {}
    rules = [r["rule"] for r in generator._records
             if r.get("record") == "call" and r.get("status") == "accepted"]
    report = {
        "status": "ok" if error is None else "error",
        "component": "generation",
        "method": {"type": type_name,
                   "params": {k: v for k, v in params.items() if k not in ("rules_path", "rebuild")}},
        "config": args.config,
        "mode": generator.mode,
        "problem_set": generator.problem_set,
        "model": generator.model_name,
        "seed": generator.seed,
        "calls": calls,
        "p_success": {"n_total": calls["total"], "n_success": calls["success"],
                      "value": round(calls["success"] / calls["total"], 6) if calls["total"] else 0.0},
        "n_rules": final.get("n_rules", len(rules) if error is None else 0),
        "est_tokens": final.get("est_tokens"),
        "rules": rules if error is None else [],
        "text": text,
        "dump": dump_path,
        "timing_s": {"total": round(time.time() - started, 3)},
    }
    if error:
        report["error"] = error
    return report


def main(argv=None) -> int:
    logging.basicConfig(level=logging.INFO, stream=sys.stderr,
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    args = parse_args(argv)
    try:
        report = run(args)
        code = EXIT_OK if report["status"] == "ok" else EXIT_RUNTIME
    except ConfigError as exc:
        report = {"status": "error", "component": "generation",
                  "error": {"type": "invalid_input", "message": str(exc)}}
        code = EXIT_INVALID_INPUT
    except Exception as exc:
        traceback.print_exc()
        report = {"status": "error", "component": "generation",
                  "error": {"type": "internal_error",
                            "message": f"{type(exc).__name__}: {exc}"}}
        code = EXIT_INTERNAL
    text = json.dumps(report, ensure_ascii=False, indent=2)
    print(text)
    os.makedirs(os.path.abspath(args.out_dir), exist_ok=True)
    with open(os.path.join(os.path.abspath(args.out_dir), RESULT_NAME), "w",
              encoding="utf-8") as handle:
        handle.write(text + "\n")
    return code


if __name__ == "__main__":
    raise SystemExit(main())
