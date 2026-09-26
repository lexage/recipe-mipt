"""summary.json and summary.md of a pk3 test run.

Every number is read from files the run left behind: ``results.csv`` and
``runtime_stats.json`` of each pipeline run (written by run_ds1000.py), the
standalone filtration report and the rule generator's dump.
"""

import csv
import json
import os
import re

from collections import Counter

from filter_gen_eval.settings import EXPERIMENTS, stem

OUTCOMES = ("passed", "failed", "timed out")
MET, NOT_MET, NO_DATA = "выполнено", "не выполнено", "нет данных"


# ------------------------------------------------------------------ readers
def read_pipeline_run(runs_dir: str, config_stem: str):
    """Numbers of one run_ds1000.py run, or None if it left nothing."""
    base = os.path.join(runs_dir, config_stem)
    if not os.path.isdir(base):
        return None
    subdirs = sorted(d for d in os.listdir(base) if os.path.isdir(os.path.join(base, d)))
    if not subdirs:
        return None
    path = os.path.join(base, subdirs[-1])
    results = os.path.join(path, "results.csv")
    if not os.path.isfile(results):
        return {"dir": path, "complete": False}

    scores, outcomes = [], Counter()
    with open(results, newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            scores.append(float(row["score"]))
            label = re.split(r"[:(]", row.get("result") or "", maxsplit=1)[0].strip()
            outcomes[label if label in OUTCOMES else "other"] += 1

    stats = {}
    stats_path = os.path.join(path, "runtime_stats.json")
    if os.path.isfile(stats_path):
        with open(stats_path, encoding="utf-8") as handle:
            stats = json.load(handle)

    return {
        "dir": path,
        "complete": bool(scores) and bool(stats),
        "tasks": len(scores),
        "passed": int(sum(scores)),
        "pass_at_1": round(sum(scores) / len(scores), 4) if scores else None,
        "outcomes": dict(outcomes),
        "index_build_time_s": stats.get("init_time_s"),
        "solve_time_s": stats.get("total_time_s"),
        "full_run_time_s": stats.get("config_time_s"),
        "document_filter_time_s": stats.get("document_filter_apply_time_s"),
        "mean_tokens_per_task": stats.get("eval_mean_total_tokens"),
        "peak_rss_mb": stats.get("peak_rss_mb"),
    }


def _load_json(path: str):
    if not os.path.isfile(path):
        return None
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)


# ------------------------------------------------------------------ helpers
def _faster_pct(before, after):
    if before in (None, 0) or after is None:
        return None
    return round(100.0 * (before - after) / before, 2)


def _pp(value):
    return None if value is None else round(100.0 * value, 2)


def _check(value, threshold, at_least: bool) -> str:
    if value is None:
        return NO_DATA
    ok = value >= threshold if at_least else value <= threshold
    return MET if ok else NOT_MET


def _metric(key, name, value, unit, threshold, at_least, baseline=None, method=None):
    return {"key": key, "name": name, "baseline": baseline, "method": method,
            "value": value, "unit": unit, "threshold": threshold,
            "threshold_kind": "не меньше" if at_least else "не больше",
            "verdict": _check(value, threshold, at_least)}


# --------------------------------------------------------------- experiments
def summarize_filtration(run_dir: str, config: dict) -> dict:
    experiment = config["experiments"]["filtration"]
    thresholds = config["thresholds"]["filtration"]
    p_min = config["thresholds"]["p_success_min"]
    runs_dir = os.path.join(run_dir, "runs")

    report = _load_json(os.path.join(run_dir, "filtration", "filtration_report.json")) or {}
    base = read_pipeline_run(runs_dir, stem(experiment["baseline"])) or {}
    method = read_pipeline_run(runs_dir, stem(experiment["method"])) or {}

    p = (report.get("p_success") or {})
    drop = None
    if base.get("pass_at_1") is not None and method.get("pass_at_1") is not None:
        drop = _pp(base["pass_at_1"] - method["pass_at_1"])

    metrics = [
        _metric("a_pass_at_1_drop", "а) Падение PASS@1 на DS-1000 при включённой фильтрации",
                drop, "п.п.", thresholds["pass_at_1_drop_max_pp"], at_least=False,
                baseline=base.get("pass_at_1"), method=method.get("pass_at_1")),
        _metric("b_index_build_speedup", "б) Ускорение подготовки базы знаний (построение индекса)",
                _faster_pct(base.get("index_build_time_s"), method.get("index_build_time_s")),
                "%", thresholds["index_build_speedup_min_pct"], at_least=True,
                baseline=base.get("index_build_time_s"), method=method.get("index_build_time_s")),
        _metric("c_full_run_speedup", "в) Ускорение полного прогона пайплайна",
                _faster_pct(base.get("full_run_time_s"), method.get("full_run_time_s")),
                "%", thresholds["full_run_speedup_min_pct"], at_least=True,
                baseline=base.get("full_run_time_s"), method=method.get("full_run_time_s")),
        _metric("d_corpus_reduction", "г) Сокращение объёма базы знаний (символы текста документов)",
                (report.get("reduction_pct") or {}).get("document_chars"), "%",
                thresholds["corpus_reduction_min_pct"], at_least=True,
                baseline=(report.get("input") or {}).get("document_chars"),
                method=(report.get("output") or {}).get("document_chars")),
    ]
    p_success = _metric("p_success", "Вероятность успешного выполнения функции назначения "
                        "(доля документов, обработанных фильтром без сбоя)",
                        p.get("value"), "", p_min, at_least=True,
                        baseline=p.get("n_total"), method=p.get("n_success"))
    return {
        "experiment": "filtration",
        "p_success": p_success,
        "metrics": metrics,
        "filtration": {
            "status": report.get("status"),
            "input": report.get("input"),
            "output": report.get("output"),
            "decisions": report.get("decisions"),
            "reduction_pct": report.get("reduction_pct"),
            "checks": report.get("checks"),
            "timing_s": report.get("timing_s"),
        },
        "runs": {"baseline": {"config": experiment["baseline"], **base},
                 "method": {"config": experiment["method"], **method}},
    }


def summarize_generation(run_dir: str, config: dict) -> dict:
    from generate_rules import count_calls, read_dump
    from src.generation.prompt_rules import PROBLEM_SETS

    experiment = config["experiments"]["generation"]
    thresholds = config["thresholds"]["generation"]
    p_min = config["thresholds"]["p_success_min"]
    runs_dir = os.path.join(run_dir, "runs")

    base = read_pipeline_run(runs_dir, stem(experiment["baseline"])) or {}
    method = read_pipeline_run(runs_dir, stem(experiment["method"])) or {}

    records = read_dump(os.path.join(run_dir, "rules", stem(experiment["method"]) + ".jsonl"))
    finals = [r for r in records if r.get("record") == "final"]
    final = finals[-1] if finals else {}
    first = next((r for r in records if r.get("record") == "call"), {})
    mode = final.get("mode") or first.get("mode")
    problem_set = final.get("problem_set") or first.get("problem_set")
    calls = None
    if mode:
        calls = count_calls(records, mode, len(PROBLEM_SETS.get(problem_set, [])),
                            completed=bool(final.get("text")))

    gain = None
    if base.get("pass_at_1") is not None and method.get("pass_at_1") is not None:
        gain = _pp(method["pass_at_1"] - base["pass_at_1"])

    value = round(calls["success"] / calls["total"], 6) if calls and calls["total"] else None
    return {
        "experiment": "generation",
        "p_success": _metric("p_success", "Вероятность успешного выполнения функции назначения "
                             "(доля вызовов генератора с разобранным ответом)",
                             value, "", p_min, at_least=True,
                             baseline=calls["total"] if calls else None,
                             method=calls["success"] if calls else None),
        "metrics": [
            _metric("pass_at_1_gain", "Прирост PASS@1 на DS-1000 с правилами в системном промпте",
                    gain, "п.п.", thresholds["pass_at_1_gain_min_pp"], at_least=True,
                    baseline=base.get("pass_at_1"), method=method.get("pass_at_1")),
        ],
        "generation": {
            "mode": mode, "problem_set": problem_set, "calls": calls,
            "n_rules": final.get("n_rules"), "est_tokens": final.get("est_tokens"),
            "model": final.get("model"), "seed": final.get("seed"),
            "text": final.get("text", ""),
        },
        "runs": {"baseline": {"config": experiment["baseline"], **base},
                 "method": {"config": experiment["method"], **method}},
    }


# ------------------------------------------------------------------ writing
def _fmt(value, unit=""):
    if value is None:
        return "—"
    if isinstance(value, float):
        # доли (PASS@1, Pусп) — до 4 знаков, проценты и п.п. — до 2, секунды — 1
        digits = 4 if abs(value) < 1 else 2 if abs(value) < 100 else 1
        text = f"{value:.{digits}f}"
        if digits > 1:
            text = text.rstrip("0").rstrip(".")
    else:
        text = str(value)
    return f"{text} {unit}".strip()


TITLES = {"filtration": "фильтрация", "generation": "генерация",
          "all": "фильтрация и генерация"}


def render_markdown(summary: dict, manifest: dict) -> str:
    git = manifest.get("git", {})
    lines = [
        f"# Сводка испытаний компонента pk3: {TITLES[summary['experiment']]}",
        "",
        f"- Каталог прогона: {summary['run_dir']}",
        f"- Начало (UTC): {manifest.get('started_utc', '—')}",
        f"- Код: {git.get('commit', '—')}"
        + (" (есть незакоммиченные правки)" if git.get("dirty") else ""),
        f"- Модели: {manifest.get('models', {}).get('llm', '—')}; "
        f"эмбеддер {manifest.get('models', {}).get('embedder', '—')}",
        f"- Итог: {summary['verdict']}",
    ]
    if summary["experiment"] != "all":
        return "\n".join(lines + _render_experiment(summary, "##")) + "\n"
    for name in EXPERIMENTS:
        part = summary["parts"][name]
        lines += ["", f"## Эксперимент: {TITLES[name]}", "", f"- Итог: {part['verdict']}"]
        lines += _render_experiment(part, "###")
    return "\n".join(lines) + "\n"


def _render_experiment(summary: dict, h: str) -> list:
    lines = ["", f"{h} Вероятность успешного выполнения функции назначения (п. 3.5.4.1 ТЗ)", ""]
    p = summary["p_success"]
    lines += [
        f"- {p['name']}",
        f"- Nобщ = {_fmt(p['baseline'])}, Nусп = {_fmt(p['method'])}, "
        f"Pусп = {_fmt(p['value'])} (порог: {p['threshold_kind']} {p['threshold']}) — {p['verdict']}",
        "",
        f"{h} Метрики качества",
        "",
        "| Метрика | Бейзлайн | Метод | Значение | Порог | Итог |",
        "|---|---|---|---|---|---|",
    ]
    for m in summary["metrics"]:
        lines.append(f"| {m['name']} | {_fmt(m['baseline'])} | {_fmt(m['method'])} | "
                     f"{_fmt(m['value'], m['unit'])} | {m['threshold_kind']} "
                     f"{m['threshold']} {m['unit']} | {m['verdict']} |")

    lines += ["", f"{h} Прогоны пайплайна", "",
              "| Роль | Конфиг | Задач | Решено | PASS@1 | Индекс, с | Решение, с | Всего, с | "
              "Токенов на задачу | Исходы |",
              "|---|---|---|---|---|---|---|---|---|---|"]
    for role, run in (("бейзлайн", summary["runs"]["baseline"]), ("метод", summary["runs"]["method"])):
        outcomes = ", ".join(f"{k}: {v}" for k, v in sorted((run.get("outcomes") or {}).items()))
        lines.append(f"| {role} | {run['config']} | {_fmt(run.get('tasks'))} | "
                     f"{_fmt(run.get('passed'))} | {_fmt(run.get('pass_at_1'))} | "
                     f"{_fmt(run.get('index_build_time_s'))} | {_fmt(run.get('solve_time_s'))} | "
                     f"{_fmt(run.get('full_run_time_s'))} | {_fmt(run.get('mean_tokens_per_task'))} | "
                     f"{outcomes or '—'} |")

    if summary["experiment"] == "filtration":
        f = summary["filtration"]
        inp, out = f.get("input") or {}, f.get("output") or {}
        dec, checks = f.get("decisions") or {}, f.get("checks") or {}
        lines += ["", f"{h} Отдельный запуск фильтрации", "",
                  f"- Вход: {inp.get('path', '—')} — документов {_fmt(inp.get('documents'))}, "
                  f"примеров {_fmt(inp.get('examples'))}, символов {_fmt(inp.get('document_chars'))}",
                  f"- Выход: {out.get('path', '—')} — документов {_fmt(out.get('documents'))}, "
                  f"примеров {_fmt(out.get('examples'))}, символов {_fmt(out.get('document_chars'))}",
                  f"- Решения: оставлено {_fmt(dec.get('kept'))}, отброшено {_fmt(dec.get('dropped'))}, "
                  f"сбоев {_fmt(dec.get('failed'))}",
                  f"- Проверки: решения по одному документу совпадают с вызовом на всём корпусе — "
                  f"{checks.get('batch_consistent')}; документы из выходного JSON совпадают с тем, "
                  f"что пайплайн проиндексировал бы после фильтра — {checks.get('equivalent_to_pipeline')}",
                  f"- Время фильтрации (метод, весь корпус): "
                  f"{_fmt((f.get('timing_s') or {}).get('filter'), 'с')}"]
    else:
        g = summary["generation"]
        calls = g.get("calls") or {}
        lines += ["", f"{h} Генерация правил в прогоне с методом", "",
                  f"- Режим: {g.get('mode')}, набор проблем: {g.get('problem_set')}, "
                  f"модель: {g.get('model')}, seed: {g.get('seed')}",
                  f"- Вызовов: {_fmt(calls.get('total'))}; новых правил: {_fmt(calls.get('accepted_rules'))}; "
                  f"проблем, покрытых принятым правилом: {_fmt(calls.get('covered'))}; "
                  f"ответов без правила: {_fmt(calls.get('rejected'))}; "
                  f"не выполнено: {_fmt(calls.get('not_completed'))}",
                  f"- Правил в блоке: {_fmt(g.get('n_rules'))}, оценка длины ≈ {_fmt(g.get('est_tokens'))} токенов",
                  "", "Блок правил, дописанный к системному промпту решателя:", "", "```text",
                  g.get("text") or "(нет)", "```"]
    return lines


NOT_COMPLETE = "прогон не завершён: нет результатов одного из конфигов"
ALL_MET, NOT_ALL_MET = "все пороги достигнуты", "не все пороги достигнуты"


def _summarize_experiment(run_dir: str, experiment: str, config: dict) -> dict:
    if experiment == "filtration":
        summary = summarize_filtration(run_dir, config)
    else:
        summary = summarize_generation(run_dir, config)
    runs = summary["runs"]
    complete = all(runs[role].get("complete") for role in ("baseline", "method"))
    verdicts = [summary["p_success"]["verdict"]] + [m["verdict"] for m in summary["metrics"]]
    if not complete:
        summary["verdict"] = NOT_COMPLETE
    elif all(v == MET for v in verdicts):
        summary["verdict"] = ALL_MET
    else:
        summary["verdict"] = NOT_ALL_MET
    summary["complete"] = complete
    return summary


def summarize(run_dir: str, experiment: str, config: dict, manifest: dict) -> dict:
    if experiment == "all":
        parts = {name: _summarize_experiment(run_dir, name, config) for name in EXPERIMENTS}
        complete = all(part["complete"] for part in parts.values())
        if not complete:
            verdict = NOT_COMPLETE
        elif all(part["verdict"] == ALL_MET for part in parts.values()):
            verdict = ALL_MET
        else:
            verdict = NOT_ALL_MET
        summary = {"experiment": "all", "verdict": verdict, "complete": complete,
                   "parts": parts}
    else:
        summary = _summarize_experiment(run_dir, experiment, config)
    summary["run_dir"] = run_dir

    with open(os.path.join(run_dir, "summary.json"), "w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    markdown = render_markdown(summary, manifest)
    with open(os.path.join(run_dir, "summary.md"), "w", encoding="utf-8") as handle:
        handle.write(markdown)
    return summary
