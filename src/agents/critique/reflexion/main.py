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
        is_chat: bool = False,
        chat_few_shot: bool = False,
    ):
        super().__init__(name)
        self.do_eval = do_eval
        self.is_chat = is_chat
        self.chat_few_shot = chat_few_shot

    def llm(self, prompt: str | list[dict]) -> str:
        # TODO: implement actual LLM call
        return ""

    def make_prompt(
        self, question: str, answer: str, evaluation: str
    ) -> str | list[dict]:

        if self.is_chat:
            if self.chat_few_shot is not None:
                messages = [
                    dict(
                        role="system",
                        content=PY_SELF_REFLECTION_CHAT_INSTRUCTION,
                    ),
                    dict(
                        role="user",
                        content=f"{PY_SELF_REFLECTION_FEW_SHOT}\n\n"
                        f"[task]:\n{question}\n\n[function impl]:\n{answer}\n\n"
                        f"[unit test results]:\n{evaluation}\n\n[self-reflection]:",
                    ),
                ]
                return messages
            else:
                messages = [
                    dict(
                        role="system",
                        content=PY_SELF_REFLECTION_CHAT_INSTRUCTION,
                    ),
                    dict(
                        role="user",
                        content=f"[task]:\n{question}\n\n[function impl]:\n{answer}\n\n"
                        f"[unit test results]:\n{evaluation}\n\n[self-reflection]:",
                    ),
                ]
                return messages
        else:
            prompt = (
                f"{PY_SELF_REFLECTION_COMPLETION_INSTRUCTION}\n"
                f"{question}\n\n{answer}\n\n{evaluation}\n\nExplanation:"
            )
            return prompt  # type: ignore

    def reflect(self, question: str, answer: str, evaluation: str) -> str:
        prompt = self.make_prompt(question, answer, evaluation)
        result = self.llm(prompt)
        return result

    def evaluate(self, question: str, answer: str) -> str:
        # TODO: implement tests/judge/etc.

        if self.do_eval:
            pass

        return ""

    def run(self, question: str, answer: str):
        evaluation = self.evaluate(question, answer)
        feedback = self.reflect(question, answer, evaluation)
        return feedback
