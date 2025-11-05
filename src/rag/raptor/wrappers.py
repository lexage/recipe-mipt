from src.rag.raptor.core import BaseEmbeddingModel, BaseSummarizationModel, BaseQAModel
from src.agent_constructor.agent import Agent

class QAModelWrapper(BaseQAModel):
    def __init__(self, agent: Agent):
        super().__init__()
        self.agent = agent
    
    def answer_question(self, context, question):
        return self.agent.run(context, question)


class SummarizationWrapper(BaseSummarizationModel):
    def __init__(self, agent: Agent):
        super().__init__()
        self.agent = agent

    def summarize(self, context, max_tokens=150):
        return self.agent.run(context)


class EmbeddingWrapper(BaseEmbeddingModel):
    def __init__(self, agent: Agent):
        self.agent = agent

    def create_embedding(self, text):
        return self.agent.run(text)[0]
