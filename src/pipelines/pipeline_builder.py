from src.pipelines.configs import PipelineConfig
from src.pipelines.factory import ComponentFactory
from src.pipelines.templates import SimplePipeline, REWOOPipeline, MAPSPipeline
from src.agent_constructor.pipeline import Pipeline

class PipelineBuilder:
    def __init__(self, config: PipelineConfig):
        self.config = config
        self.factory = ComponentFactory()
        self._components = {}
    
    def build(self) -> Pipeline:
        # Создаём компоненты в правильном порядке
        for component_name in self.config.execution_order:
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