from src.agents.agent_constructor.pipeline import Agent


class EmbeddingFunctionWrapper:
    def __init__(self, embedding_agent: Agent):
        self.agent = embedding_agent
    
    def __call__(self, input):
        return self.agent.run(input)
    
    def embed_query(self, input):
        return self.__call__(input)
    
    def name(self):
        return "custom-embedding"
