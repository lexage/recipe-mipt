from src.agent_constructor.agent import Agent
from src.agent_constructor.core import Text
from src.pipelines.registry import register_component
from src.pipelines.constants import ComponentNames

@register_component(ComponentNames.DUMMY_AGENT)
class DummyAgent(Agent):
    def __init__(self, name: str):
        super().__init__(name)
    
    def run(self, task: Text, *args, **kwargs):
        return f"Dummy answer on query:\n'''{task}'''\n"
