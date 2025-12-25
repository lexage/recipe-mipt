import inspect
import yaml
import importlib

from dataclasses import dataclass
from typing import Dict, Type, List
from pathlib import Path

from src.pipelines.constants import ComponentNames
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
    _registry: Dict[ComponentNames, ComponentInfo] = {}

    def load_modules(self, components_names: List[ComponentNames]) -> None:
        for component_name in components_names:

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
    def _extract_params(component_class: Type) -> List[str]:
        sig = inspect.signature(component_class.__init__)
        
        params = {}
        for param in sig.parameters.values():
            
            if param.name == 'self':
                continue
            
            param_is_dep = (
                param.annotation != inspect.Parameter.empty 
                and inspect.isclass(param.annotation) 
                and issubclass(param.annotation, Block)
            )
            
            param_has_default = param.default != inspect.Parameter.empty

            params[param.name] = ParamInfo(
                is_dependency=param_is_dep, 
                has_default=param_has_default, 
            )

        return params

    def get_component_info(self, component_type: ComponentNames) -> ComponentInfo:
        return self._registry.get(component_type, None)

    def get_component_deps(self, component_type: ComponentNames) -> Dict:
        component_info = self._registry.get(component_type, None)
        if component_info is None or not hasattr(component_info, "params"):
            return []
        return {
            param_name: param_info
            for param_name, param_info in component_info.params.items()
            if getattr(param_info, "is_dependency", True)
        }

    def component_exist(self, component_type: ComponentNames):
        return component_type in self._registry
