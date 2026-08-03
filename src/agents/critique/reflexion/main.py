from abc import ABC, abstractmethod
from .prompts_code import *
from src.agent_constructor.agent import Agent
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage


class Reflexion(Agent):
    """
    Agent that implements Reflexion: Language Agents
    with Verbal Reinforcement Learning.
    Agent generate criticism of the implementation of the problem.
    At the first stage agent generate an evaluation of the implementation. (Evaluation is an integer an integer between one and five)
    Then the agent generates a criticism of the implementation using this evaluation.
    """

    # Prompts in original paper are task-specific.

    def __init__(
        self,
        name: str = "Reflexion",
        model_name: str = "Qwen/Qwen1.5-32B-Chat-AWQ",
        openai_api_base_url="http://localhost:7215/v1",
    ):
        super().__init__(name)
        self.llm_model = ChatOpenAI(
            model=model_name,
            openai_api_base=openai_api_base_url,
            openai_api_key="fake-key",
            temperature=0.7,
        )

    def llm(self, message) -> str:

        messages = [HumanMessage(content=message)]

        response = self.llm_model.invoke(messages)
        return response.content

    def make_evaluate_prompt(
        self,
        question: str,
        answer: str,
    ) -> str | list[dict]:
        prompt = rf"""{PY_EVALUATE_INSTRUCTION}\n{PY_EVALUATE_FEW_SHOT}\n  Problem: {question}\n\n Implementation: {answer}\n\n\Evaluation: """
        return prompt

    def make_reflect_prompt(
        self, question: str, answer: str, evaluation: str
    ) -> str | list[dict]:
        prompt = rf"""{PY_SELF_REFLECTION_INSTRUCTION}\n{PY_SELF_REFLECTION_FEW_SHOT}\n  Problem: {question}\n\n Implementation: {answer}\n\n Evaluation: {evaluation}\n\Reflection:"""
        return prompt

    def reflect(self, question: str, answer: str, evaluation: str) -> str:
        prompt = self.make_reflect_prompt(question, answer, evaluation)
        result = self.llm(prompt)
        return result

    def evaluate(self, question: str, answer: str) -> str:
        prompt = self.make_evaluate_prompt(question, answer)
        result = self.llm(prompt)
        return result

    def run(self, question: str, answer: str):
        feedback = self.evaluate(question, answer)
        return self.reflect(question, answer, feedback)
