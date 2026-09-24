"""python -m filter_gen_eval <команда> — вызывается скриптами из scripts/."""

import argparse
import json
import os

from filter_gen_eval import runs
from filter_gen_eval.contract import run_contract
from filter_gen_eval.settings import load_config, recipe_root
from filter_gen_eval.summary import summarize


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="filter_gen_eval",
                                     description="Испытания компонента pk3.")
    parser.add_argument("--config-toml", required=True, help="config.toml испытаний")
    parser.add_argument("--run-dir", required=True, help="каталог прогона")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("filtration", help="отдельный запуск фильтрации")
    p.add_argument("--filter-config", help="конфиг с компонентом фильтра")
    p.add_argument("--input", help="входной JSON-корпус")

    p = sub.add_parser("generation", help="отдельный запуск генерации правил")
    p.add_argument("--rules-config", help="конфиг с компонентом генератора")

    p = sub.add_parser("evaluate", help="эксперимент: бейзлайн и метод в пайплайне")
    p.add_argument("--experiment", required=True, choices=("filtration", "generation"))
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--limit", type=int, default=0, help="только N задач DS-1000 (0 — все)")
    p.add_argument("--no-log-chunks", action="store_true")

    sub.add_parser("contract", help="проверка встроенных средств проверки ввода")

    p = sub.add_parser("summarize", help="пересобрать summary.md по готовому прогону")
    p.add_argument("--experiment", required=True, choices=("filtration", "generation"))

    args = parser.parse_args(argv)
    root = recipe_root()
    config = load_config(args.config_toml)
    run_dir = os.path.abspath(args.run_dir)
    os.makedirs(run_dir, exist_ok=True)

    if args.command == "filtration":
        return runs.filtration(root, config, run_dir, args.filter_config, args.input)
    if args.command == "generation":
        return runs.generation(root, config, run_dir, args.rules_config)
    if args.command == "evaluate":
        return runs.evaluate(root, config, run_dir, args.experiment, args.workers,
                             args.limit, log_chunks=not args.no_log_chunks)
    if args.command == "contract":
        return run_contract(root, config, run_dir)
    if args.command == "summarize":
        manifest_path = os.path.join(run_dir, "run_manifest.json")
        manifest = {}
        if os.path.isfile(manifest_path):
            with open(manifest_path, encoding="utf-8") as handle:
                manifest = json.load(handle)
        summary = summarize(run_dir, args.experiment, config, manifest)
        with open(os.path.join(run_dir, "summary.md"), encoding="utf-8") as handle:
            print(handle.read())
        return 0 if summary["complete"] else 3
    return 1
