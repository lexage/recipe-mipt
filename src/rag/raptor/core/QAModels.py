from abc import ABC, abstractmethod

class BaseQAModel(ABC):
    @abstractmethod
    def answer_question(self, context, question):
        pass
