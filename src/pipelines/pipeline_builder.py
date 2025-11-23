from src.pipelines.configs import PipelineConfig
from src.pipelines.factory import ComponentFactory
from src.pipelines.templates import SimplePipeline, REWOOPipeline, MAPSPipeline
from src.agent_constructor.pipeline import Pipeline
from src.pipelines.registry import get_component_dependencies
from typing import List
import networkx as nx


class PipelineBuilder:
    def __init__(self, config: PipelineConfig):
        self.config = config
        self.factory = ComponentFactory()
        self._components = {}
    

    @staticmethod
    def _build_order(config: PipelineConfig) -> List[str]:
        
        graph = nx.DiGraph()
        for component_name in config.components:
            graph.add_node(component_name)

        for component_name, component_config in config.components.items():
            for dep_name in get_component_dependencies(component_config):
                if dep_name in config.components:
                    graph.add_edge(dep_name, component_name)
                else:
                    raise ValueError(f"Missing '{dep_name}' for '{component_name}'")
        
        if not nx.is_directed_acyclic_graph(graph):
            raise ValueError("Cyclic dependencies have been discovered")
        
        return list(nx.topological_sort(graph))

    def build(self) -> Pipeline:

        build_order = self._build_order(self.config)


        # Создаём компоненты в правильном порядке
        for component_name in build_order:
            component_config = self.config.components[component_name]
            
            # Создаём компонент
            component = self.factory.create_component(
                component_config, 
                dependencies=self._components
            )

            self._components[component_name] = component
        if self.config.pipeline_type == "simple":
            return SimplePipeline(
                retriever=self._components["retriever"],
                agent=self._components["agent"],
                context_assembler=self._components["context_assembler"]
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