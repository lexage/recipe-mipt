"""One component of a pipeline config, built on its own.

The standalone filtration and rule-generation runs must use exactly the
parameters the pipeline would, so they read the same YAML and build the
component through the same registry and factory as ``PipelineBuilder``.

Unlike the pipeline factory, which silently skips keys a component does not
take (CLAUDE.md gotcha #3), this builder rejects unknown keys and values of the
wrong type: a standalone run is the component's input check (ТЗ 3.5.5.1).
"""

import inspect
import os

from typing import Any, Dict, Optional, Tuple, Union, get_args, get_origin

import yaml

from src.pipelines.constants import ComponentNames
from src.pipelines.factory import ComponentFactory
from src.pipelines.registry import ComponentRegistry


class ConfigError(ValueError):
    """The config or a parameter in it is not acceptable; the message says why."""


def load_config(path: str) -> dict:
    if not os.path.isfile(path):
        raise ConfigError(f"конфиг не найден: {path}")
    try:
        with open(path, "r", encoding="utf-8") as handle:
            config = yaml.safe_load(handle)
    except yaml.YAMLError as exc:
        raise ConfigError(f"{path}: некорректный YAML — {exc}") from None
    if not isinstance(config, dict) or not isinstance(config.get("components"), dict):
        raise ConfigError(f"{path}: ожидался конфиг пайплайна с разделом components")
    return config


def component_section(config: dict, key: str) -> Tuple[str, Dict[str, Any]]:
    """(type name, params) of ``components.<key>``."""
    section = config["components"].get(key)
    if not isinstance(section, dict) or "type" not in section:
        raise ConfigError(f"в конфиге нет компонента components.{key} с полем type")
    params = section.get("params") or {}
    if not isinstance(params, dict):
        raise ConfigError(f"components.{key}.params должен быть словарём")
    return section["type"], dict(params)


def _accepts(annotation, value) -> bool:
    if annotation is inspect.Parameter.empty or annotation is Any:
        return True
    origin = get_origin(annotation)
    if origin is Union:
        return any(_accepts(arg, value) for arg in get_args(annotation))
    if annotation is type(None):
        return value is None
    if annotation is bool:
        return isinstance(value, bool)
    if annotation is int:
        return isinstance(value, int) and not isinstance(value, bool)
    if annotation is float:
        return isinstance(value, (int, float)) and not isinstance(value, bool)
    if annotation is str:
        return isinstance(value, str)
    if inspect.isclass(annotation):
        return isinstance(value, annotation)
    return True


def check_params(component_class, params: Dict[str, Any], where: str) -> None:
    """Reject keys the class does not take and values of the wrong type."""
    signature = inspect.signature(component_class.__init__)
    accepted = {name: p for name, p in signature.parameters.items() if name != "self"}
    unknown = sorted(set(params) - set(accepted))
    if unknown:
        raise ConfigError(f"{where}: неизвестные параметры {', '.join(unknown)} "
                          f"(допустимы: {', '.join(accepted)})")
    for name, value in params.items():
        parameter = accepted[name]
        annotation = parameter.annotation
        if annotation is inspect.Parameter.empty and parameter.default not in (None, inspect.Parameter.empty):
            annotation = type(parameter.default)
        if not _accepts(annotation, value):
            raise ConfigError(f"{where}.{name}: недопустимый тип значения "
                              f"{type(value).__name__} ({value!r})")


def build_component(type_name: str, params: Dict[str, Any],
                    where: Optional[str] = None):
    """Build a dependency-free component exactly as the pipeline factory would."""
    where = where or type_name
    try:
        component_type = ComponentNames[type_name]
    except KeyError:
        raise ConfigError(f"{where}: неизвестный тип компонента {type_name!r}") from None
    registry = ComponentRegistry()
    registry.load_modules([component_type])
    info = registry.get_component_info(component_type)
    check_params(info.class_, params, where)
    missing = [name for name, p in info.params.items()
               if p.is_dependency and not p.has_default]
    if missing:
        raise ConfigError(f"{where}: компонент требует зависимостей "
                          f"({', '.join(missing)}) и не собирается отдельно")
    return ComponentFactory().create_component(
        component_info=info,
        available_dependencies={},
        deps_mapping={},
        config_params=params,
    )
