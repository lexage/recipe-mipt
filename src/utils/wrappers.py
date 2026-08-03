from src.agent_constructor.agent import Agent


class EmbeddingFunctionWrapper:
    def __init__(self, embedding_agent):
        # Если пришёл кортеж, берём первый элемент
        if isinstance(embedding_agent, tuple) and len(embedding_agent) > 0:
            self.agent = embedding_agent[0]
        else:
            self.agent = embedding_agent
    
    def __call__(self, input):
        # Проверяем, что агент действительно имеет метод run
        if hasattr(self.agent, 'run'):
            return self.agent.run(input)
        else:
            # Если нет run, возвращаем заглушку
            print(f"Предупреждение: агент {type(self.agent)} не имеет метода run")
            return [0.0] * 384
    
    def embed_query(self, input):
        return self.__call__(input)
    
    def name(self):
        return "custom_embedding_function"