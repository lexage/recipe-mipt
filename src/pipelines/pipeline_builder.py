from src.agent_constructor.agent import Agent
from src.agent_constructor.db import IDB
from src.agent_constructor.chunkers import Chunker
from src.agent_constructor.context_engine import Retriever
from src.agent_constructor.filters import Filter
from src.agent_constructor.configs import PipelineConfig
from src.agent_constructor.factory import ComponentFactory


class SimplePipeline:
    def __init__(self, retriever: Retriever, agent: Agent):
        self.retriever = retriever
        self.agent = agent
            
    def run(self, task: str) -> str:
        context = self.retriever.retrieve(query=task)[0][1][0].text

        return self.agent.run(context, task)

class PipelineBuilder:
    def __init__(self, config: PipelineConfig):
        self.config = config
        self.factory = ComponentFactory()
        self._components = {}
    
    def build(self) -> SimplePipeline:
        # Создаём компоненты в правильном порядке
        for component_name in self.config.execution_order:
            component_config = self.config.components[component_name]
            
            # Собираем зависимости для текущего компонента
            dependencies = {}
            if hasattr(component_config, 'dependencies'):
                for dep_name in component_config.dependencies:
                    if dep_name not in self._components:
                        raise ValueError(f"Dependency {dep_name} not found for {component_name}")
                    dependencies[dep_name] = self._components[dep_name]
            
            # Создаём компонент
            component = self.factory.create_component(
                component_config, 
                dependencies
            )

            self._components[component_name] = component
        
        return SimplePipeline(
            retriever=self._components["retriever"],
            agent=self._components["agent"],
        )