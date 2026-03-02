import yaml

from typing import Dict, Any, List, Union

from pydantic import BaseModel

from src.pipelines.constants import ComponentNames, PipelinesNames


class ComponentConfig(BaseModel):
    type: ComponentNames
    params: Dict[str, Any] = {}
    deps_mapping: Dict[str, str] = {}


class PipelineConfig(BaseModel):
    type: PipelinesNames
    components: Dict[str, Union[ComponentConfig, List[ComponentConfig]]]
    params: Dict[str, Any] = {}


class ConfigLoader:
    def __init__(self):
        pass

    def load_from_yaml(self, path_to_cfg: str) -> PipelineConfig:
        
        with open(path_to_cfg, 'r') as file:
            config = yaml.safe_load(file)

        pipeline_type = config["type"]
        pipeline_params = config.get("params", {})

        pipeline_components = {}

        for component_name, component_config in config["components"].items():
            pipeline_components.setdefault(
                component_name,
                self._parse_component_config(component_config)
            )

        return PipelineConfig(
            type=PipelinesNames[pipeline_type],
            params=pipeline_params,
            components=pipeline_components,
        )
        
    def _parse_component_config(self, config: List | Dict) -> ComponentConfig | List[ComponentConfig]:
        
        if isinstance(config, Dict):
            return ComponentConfig(
                type=ComponentNames[config["type"]],
                params=config.get("params", {}),
                deps_mapping=config.get("deps_mapping", {})
            )
        elif isinstance(config, List):
            return [
                self._parse_component_config(item_config) 
                for item_config in config
            ]
