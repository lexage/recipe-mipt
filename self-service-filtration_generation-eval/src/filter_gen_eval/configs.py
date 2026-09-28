"""Per-run copies of the PMI configs.

A run must not reuse an index or a rule dump from an earlier run: an existing
index directory is reused without re-embedding (the index-build time would be
meaningless and a filtered corpus would land in an index that already holds
the full one), and ``rebuild: True`` overwrites the dump at ``rules_path``. So
every run gets copies of the configs whose paths point inside its own run
directory; nothing else in them changes.
"""

import os

import yaml


def prepare_config(source: str, target: str, *, path_to_db: str = None,
                   path_to_vector_db: str = None, rules_path: str = None,
                   max_docs: int = 0) -> dict:
    """Write ``target`` = ``source`` with the given paths replaced; return changes."""
    with open(source, encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    components = config["components"]
    changes = {}

    db_params = components["data_base"].setdefault("params", {})
    if path_to_db:
        changes["data_base.path_to_db"] = [db_params.get("path_to_db"), path_to_db]
        db_params["path_to_db"] = path_to_db
    if path_to_vector_db:
        changes["data_base.path_to_vector_db"] = [db_params.get("path_to_vector_db"),
                                                  path_to_vector_db]
        db_params["path_to_vector_db"] = path_to_vector_db
    if max_docs:
        # Проверка стенда: индекс только по первым документам корпуса.
        changes["data_base.max_docs"] = [db_params.get("max_docs"), max_docs]
        db_params["max_docs"] = max_docs
    if rules_path:
        writer = components["rule_writer"].setdefault("params", {})
        changes["rule_writer.rules_path"] = [writer.get("rules_path"), rules_path]
        writer["rules_path"] = rules_path
        writer["rebuild"] = True

    header = [f"# Копия {source} для одного прогона испытаний pk3."]
    header += [f"# Изменено: {key}: {old} -> {new}" for key, (old, new) in changes.items()]
    os.makedirs(os.path.dirname(os.path.abspath(target)), exist_ok=True)
    with open(target, "w", encoding="utf-8") as handle:
        handle.write("\n".join(header) + "\n\n")
        yaml.safe_dump(config, handle, sort_keys=False, allow_unicode=True)
    return changes
