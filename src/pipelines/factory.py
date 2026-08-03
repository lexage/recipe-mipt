from typing import Dict, Any

from src.pipelines.registry import ComponentInfo


class ComponentFactory:

    def create_component(
        self,
        component_info: ComponentInfo,
        available_dependencies: Dict[str, Any],
        deps_mapping: Dict[str, str],
        config_params: Dict[str, Any],
    ):

        params = {}

        for param_name, param_info in component_info.params.items():

            dep_name = deps_mapping.get(param_name, param_name)

            if param_info.is_dependency and dep_name in available_dependencies:
                params[param_name] = available_dependencies[dep_name]

            elif param_name in config_params:
                params[param_name] = config_params[param_name]

            elif param_info.has_default:
                continue

            else:
                raise ValueError(
                    f"Required parameter '{param_name}' not found for {component_info.class_}. "
                    f"Available: params={list(config_params)}, "
                )

        return component_info.class_(**params)