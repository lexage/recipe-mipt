import networkx as nx

from typing import List

from src.pipelines.configs import PipelineConfig
from src.pipelines.factory import ComponentFactory
from src.pipelines.registry import ComponentRegistry
from src.agent_constructor.pipeline import Pipeline


class PipelineBuilder:
    
    def __init__(self):
        self.registry = ComponentRegistry()
        self.factory = ComponentFactory()
        
    # def _get_build_order_v0(self, config: PipelineConfig) -> List[str]:
        
    #     graph = nx.DiGraph()
    #     for component_name, component_config in config.components.items():
    #         if not self.registry.component_exist(component_config.type):
    #             raise ValueError(f"Unregistied component: '{component_config.type}'")
            
    #         graph.add_node(component_name)

    #         deps = self.registry.get_component_deps(component_config.type)

    #         for dep_name in deps:
    #             dep_name = component_config.deps_mapping.get(dep_name, dep_name)
    #             if dep_name in config.components:
    #                 graph.add_edge(dep_name, component_name)
    #             elif not deps[dep_name].has_default:
    #                 raise ValueError(f"Missing '{dep_name}' for '{component_name}'")
        
    #     if not nx.is_directed_acyclic_graph(graph):
    #         raise ValueError("Cyclic dependencies have been discovered")
        
    #     return list(nx.topological_sort(graph))
    
    def _get_build_order(self, config: PipelineConfig) -> List[str]:
        
        graph = nx.DiGraph()
        for component_name, component_config in config.components.items():
            
            configs_list = component_config if isinstance(component_config, list) else [component_config]
            
            for cfg in configs_list:
                if not self.registry.component_exist(cfg.type):
                    raise ValueError(f"Unregistered component: '{cfg.type}'")

            graph.add_node(component_name)
            
            for cfg in configs_list:
            
                deps = self.registry.get_component_deps(cfg.type)
                
                for dep_name in deps:
                    dep_name = cfg.deps_mapping.get(dep_name, dep_name)
                    if dep_name in config.components:
                        graph.add_edge(dep_name, component_name)
                    elif not deps[dep_name].has_default:
                        raise ValueError(f"Missing '{dep_name}' for '{component_name}'")
            
        if not nx.is_directed_acyclic_graph(graph):
            raise ValueError("Cyclic dependencies have been discovered")
        
        return list(nx.topological_sort(graph))
    
    def _check_config(self, config: PipelineConfig):
        pipeline_components = self.registry.get_component_deps(config.type)
        required_components = [name for name, info in pipeline_components.items() if not info.has_default]
        for name in required_components:
            if not name in config.components:
                raise ValueError(f"Missing '{name}' component for '{config.type.value}'")

    def build(self, pipeline_config: PipelineConfig) -> Pipeline:
        
        modules = []
        for item in pipeline_config.components.values():
            if isinstance(item, list):
                modules.extend(cfg.type for cfg in item)
            else:
                modules.append(item.type)
        modules.append(pipeline_config.type)
        

        self.registry.load_modules(modules)
        self._check_config(config=pipeline_config)
        build_order = self._get_build_order(pipeline_config)

        components = {}

        for component_name in build_order:
            
            raw_config = pipeline_config.components.get(component_name)
            
            component_config = raw_config if isinstance(raw_config, list) else [raw_config]
                
            components_list = []
                
            for item_config in component_config:
                
                component_info = self.registry.get_component_info(item_config.type)
                
                component = self.factory.create_component(
                    component_info=component_info, 
                    available_dependencies=components,
                    deps_mapping=item_config.deps_mapping,
                    config_params=item_config.params,
                )
                
                components_list.append(component)
                
            components[component_name] = components_list if isinstance(raw_config, list) else components_list[0]

        pipeline = self.factory.create_component(
            component_info=self.registry.get_component_info(pipeline_config.type),
            available_dependencies=components,
            deps_mapping={},
            config_params=pipeline_config.params,
        )
        
        return pipeline
