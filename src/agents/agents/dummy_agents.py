from src.agents.agent_constructor.agent import Agent


class DummyPlanner(Agent):
    def __init__(self, name: str, max_subqueries: int):
        super().__init__(name)
        self.max_num_queries = max_subqueries
    
    def run(self, query: str):
        return [f"#{i} Subquery of query '{query}'" for i in range(self.max_num_queries)]


class DummyQAAgent(Agent):
    def __init__(self, name):
        super().__init__(name)

    def run(self, context, question):
        return f"Answer on {question} using context:\n\n{context}"


class DummySummarization(Agent):
    def __init__(self, name):
        super().__init__(name)

    def run(self, context, max_tokens=150):
        return "Summarization"
