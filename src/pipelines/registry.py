import inspect

from dataclasses import dataclass
from typing import Dict, Type, List

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

    def __init__(self) -> None:
        self._registry = type(self)._registry

    @classmethod
    def register_component(cls, component_type: ComponentNames):
        
        def wrapper(component_class: Type):
            params = cls._extract_params(component_class)
            cls._registry[component_type] = ComponentInfo(
                class_=component_class,
                params=params,
            )
            return component_class
        
        return wrapper
    
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

    def get_component_deps(self, component_type: ComponentNames) -> List[str]:
        component_info = self._registry.get(component_type, None)
        if component_info is None or not hasattr(component_info, "params"):
            return []
        return [
            param_name
            for param_name, param_info in component_info.params.items()
            if getattr(param_info, "is_dependency", True)
        ]

    def component_exist(self, component_type: ComponentNames):
        return component_type in self._registry