import inspect
import yaml
import importlib

from dataclasses import dataclass
from typing import Dict, Type, List, get_origin, get_args
from pathlib import Path

from src.pipelines.constants import (
    ComponentNames,
    PipelinesNames,
    PIPELINES_IMPORT_PATH,
)
from src.agent_constructor.core import Block


@dataclass
class ParamInfo:
    is_dependency: bool
    has_default: bool


@dataclass
class ComponentInfo:
    class_: Type
    params: Dict[str, ParamInfo]


class ComponentRegistry:
    _registry: Dict[ComponentNames | PipelinesNames, ComponentInfo] = {}

    def load_modules(
        self, components_names: List[ComponentNames | PipelinesNames]
    ) -> None:
        for component_name in components_names:

            if component_name in self._registry:
                continue

            if isinstance(component_name, PipelinesNames):
                module_path = PIPELINES_IMPORT_PATH
                class_name = component_name.value
            else:
                import_path = component_name.value
                module_path, class_name = import_path.rsplit(".", 1)

            module = importlib.import_module(module_path)
            component_class = getattr(module, class_name)

            params = self._extract_params(component_class)

            self._registry[component_name] = ComponentInfo(
                class_=component_class,
                params=params,
            )

    @staticmethod
    def _extract_params(component_class: Type) -> Dict[str, ParamInfo]:
        sig = inspect.signature(component_class.__init__)

        params = {}
        for param in sig.parameters.values():

            if param.name == "self":
                continue

            param_is_dep = ComponentRegistry._is_dependency_annotation(param.annotation)

            param_has_default = param.default != inspect.Parameter.empty

            params[param.name] = ParamInfo(
                is_dependency=param_is_dep,
                has_default=param_has_default,
            )

        return params

    @staticmethod
    def _is_dependency_annotation(annotation) -> bool:
        if annotation == inspect.Parameter.empty:
            return False

        if inspect.isclass(annotation) and issubclass(annotation, Block):
            return True

        origin = get_origin(annotation)
        if origin is not None:
            args = get_args(annotation)
            for arg in args:
                if ComponentRegistry._is_dependency_annotation(arg):
                    return True

        return False

    def get_component_info(
        self, component_type: ComponentNames | PipelinesNames
    ) -> ComponentInfo:
        return self._registry.get(component_type, None)

    def get_component_deps(
        self, component_type: ComponentNames | PipelinesNames
    ) -> Dict[str, ParamInfo]:
        component_info = self._registry.get(component_type, None)
        if component_info is None or not hasattr(component_info, "params"):
            return {}
        return {
            param_name: param_info
            for param_name, param_info in component_info.params.items()
            if getattr(param_info, "is_dependency", True)
        }

    def component_exist(self, component_type: ComponentNames):
        return component_type in self._registry