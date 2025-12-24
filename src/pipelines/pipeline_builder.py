import networkx as nx

from typing import List

from src.pipelines.configs import PipelineConfig
from src.pipelines.factory import ComponentFactory
from src.pipelines.registry import ComponentRegistry
from src.pipelines.templates import SimplePipeline, REWOOPipeline, MAPSPipeline
from src.agent_constructor.pipeline import Pipeline


class PipelineBuilder:
    
    def __init__(self, config: PipelineConfig, registry_cfg_path: str):
        self.config = config
        self.registry = ComponentRegistry(
            registry_cfg_path=registry_cfg_path
        )
        components_name = [component.type for component in config.components.values()]
        self.registry.load_modules(components_name)

        self.factory = ComponentFactory()
        self._components = {}
    
    def _build_order(self, config: PipelineConfig) -> List[str]:
        
        graph = nx.DiGraph()
        for component_name, component_config in config.components.items():
            if not self.registry.component_exist(component_config.type):
                raise ValueError(f"Unregistied component: '{component_config.type}'")
            
            graph.add_node(component_name)

            deps = self.registry.get_component_deps(component_config.type)

            for dep_name in deps:
                dep_name = component_config.deps_mapping.get(dep_name, dep_name)
                if dep_name in config.components:
                    graph.add_edge(dep_name, component_name)
                elif not deps[dep_name].has_default:
                    raise ValueError(f"Missing '{dep_name}' for '{component_name}'")
        
        if not nx.is_directed_acyclic_graph(graph):
            raise ValueError("Cyclic dependencies have been discovered")
        
        return list(nx.topological_sort(graph))

    def build(self) -> Pipeline:

        build_order = self._build_order(self.config)

        for component_name in build_order:
            
            component_config = self.config.components.get(component_name)
            
            component_info = self.registry.get_component_info(component_config.type)

            component = self.factory.create_component(
                component_info=component_info, 
                available_dependencies=self._components,
                deps_mapping=component_config.deps_mapping,
                config_params=component_config.params,
            )

            self._components[component_name] = component
        if self.config.pipeline_type == "simple":
            return SimplePipeline(
                retriever=self._components["retriever"],
                agent=self._components["agent"],
                context_assembler=self._components.get("context_assembler", None)
            )
        elif self.config.pipeline_type == "rewoo":
            return REWOOPipeline(
                planner=self._components["planner"],
                worker=self._components["worker"],
                solver=self._components["solver"],
            )
        elif self.config.pipeline_type == "maps":
            return MAPSPipeline(
                manager=self._components["manager"],
                solver=self._components["solver"],
                scholar=self._components["scholar"],
                critic=self._components["critic"],
                user_proxy=self._components["user_proxy"],
                aligner=self._components["aligner"],
                max_iterations=self.config.params.get("max_iterations", 1)
            )
        else:
            raise ValueError(f"Unknown pipeline type: {self.config.pipeline_type}")
