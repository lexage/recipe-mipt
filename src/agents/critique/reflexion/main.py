from abc import ABC, abstractmethod
from .prompts_code import *


class Agent(ABC):
    """Base class for all agents in the system."""

    def __init__(self, name: str):
        self.name = name

    @abstractmethod
    def run(self, *args, **kwargs):
        raise NotImplementedError

    def __str__(self):
        return f"{self.__class__.__name__}({self.name})"


class Reflexion(Agent):
    """
    Agent that implements Reflexion: Language Agents
    with Verbal Reinforcement Learning.
    """

    # Prompts in original paper are task-specific.

    def __init__(
        self,
        name: str = "Reflexion",
        do_eval: bool = False,
    ):
        super().__init__(name)
        self.do_eval = do_eval

    def llm(self, prompt: str | list[dict]) -> str:
        # TODO: implement actual LLM call
        return ""

    def make_evaluate_prompt(
        self, question: str, answer: str,
    ) -> str | list[dict]:
        prompt = (
            f"{PY_EVALUATE_INSTRUCTION}\n{PY_EVALUATE_FEW_SHOT}\n"
            f"Problem: {question}\n\n Answer: {answer}\n\n\Evaluation: "
        )
        return prompt

    def make_reflect_prompt(
        self, question: str, answer: str, evaluation: str
    ) -> str | list[dict]:
        prompt = (
            f"{PY_SELF_REFLECTION_INSTRUCTION}\n{PY_SELF_REFLECTION_FEW_SHOT}\n"
            f" Problem: {question}\n\n Answer: {answer}\n\n Evaluation: {evaluation}\n\Reflection:"
        )
        return prompt

    def make_actor_prompt(
        self, question: str, answer: str, reflection: str
    ) -> str | list[dict]:
        prompt = (
            f"{PY_ACTOR_FEW_SHOT}\n{PY_ACTOR_INSTRUCTION}\n""
            f"Problem: {question}\n\n Answer: {answer}\n\n Reflection: {reflection}\n\Fine Answer:"
        )
        return prompt

    def reflect(self, question: str, answer: str, evaluation: str) -> str:
        prompt = self.make_reflect_prompt(question, answer, evaluation)
        result = self.llm(prompt)
        return result

    def evaluate(self, question: str, answer: str) -> str:
        # TODO: implement tests/judge/etc.

        prompt = self.make_evaluate_prompt(question, answer)
        result = self.llm(prompt)
        return result

    def actor(self, question: str, answer: str, evaluation: str, reflect: str) -> str:
        # TODO: implement tests/judge/etc.
        prompt = self.make_actor_prompt(question, answer, reflect)
        result = self.llm(prompt)
        return result

    def run(self, question: str, answer: str):
        if self.do_eval:
            feedback = self.evaluate(question, answer)
            reflect_text = self.reflect(question, answer, feedback)
            return self.actor(question, answer, reflect_text)
        return answer
