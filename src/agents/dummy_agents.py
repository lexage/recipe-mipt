from src.agent_constructor.agent import Agent
from src.agent_constructor.core import Text

class DummyAgent(Agent):
    def __init__(self, name: str):
        super().__init__(name)
    
    def run(self, task: Text, *args, **kwargs):
        return f"Dummy answer on query:\n'''{task}'''\n"
