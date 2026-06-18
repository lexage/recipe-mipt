import importlib
from abc import ABC, abstractmethod
from src.agent_constructor.agent import Agent
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage

from .examples import Example


def _load_self_refine_prompts(dataset: str):
    """Load the Self-Refine prompt module for the given dataset (ds1000, codemmlu, ...).

    Each ``prompts_<dataset>`` module exposes the same public names, so switching
    benchmarks changes only which module is imported — selected via the ``dataset`` param.
    """
    return importlib.import_module(f"src.agents.critique.self_refine.prompts_{dataset}")


class SelfRefine(Agent):
    """
    Agent that implements Self-Refine: Iterative Refinement with Self-Feedback.

    The agent uses several examples of Python solution critiques to generate criticism of the problem's implementation.
    """

    def __init__(
        self,
        name: str = "Self-Refine",
        model_name: str = "Qwen/Qwen1.5-32B-Chat-AWQ",
        openai_api_base_url="http://localhost:7215/v1",
        dataset: str = "ds1000",
    ):
        super().__init__(name)
        self.dataset = dataset
        self.prompts = _load_self_refine_prompts(dataset)
        self.llm_model = ChatOpenAI(
            model=model_name,
            openai_api_base=openai_api_base_url,
            openai_api_key="fake-key",
            temperature=0.7,
        )
        self.refrain_feedback: str = self.prompts.FEEDBACK_INSTRUCTION
        self.refrain_refine: str = self.prompts.REFINE_INSTRUCTION

    def get_examples(self) -> list[Example]:
        return self.prompts.FEWSHOT_EXAMPLES

    def llm(self, message) -> str:

        messages = [HumanMessage(content=message)]

        response = self.llm_model.invoke(messages)
        return response.content

    def feedback_prompt(
        self, question: str, answer: str, examples: list[Example] = []
    ) -> str:
        few_shot: str = """Examples:
        """
        for example in examples:
            few_shot += f"""
            Problem: {example.question}
            Implementation: {example.answer}
            {self.refrain_feedback}
            Feedback: {example.feedback}
            """

        prompt = f""" END OF EXAMPLES
        Problem: {question}
        Implementation: {answer}
        {self.refrain_feedback}
        Feedback:
        """

        final_prompt = few_shot + prompt
        return final_prompt

    def feedback(self, question: str, answer: str, examples: list[Example] = []) -> str:
        feedback_prompt = self.feedback_prompt(question, answer, examples)
        result = self.llm(feedback_prompt)
        return result

    def refine_prompt(
        self, question: str, answer: str, feedback: str, examples: list[Example] = []
    ) -> str:
        few_shot: str = """Examples:
         """
        for example in examples:
            few_shot += f"""
            Problem: {example.question}
            Implementation: {example.answer}
            {self.refrain_feedback}
            Feedback: {example.feedback}
            {self.refrain_refine}
            Refined Implementation: {example.refined}
            """

        prompt = f""" END OF EXAMPLES
        Problem: {question}
        Implementation: {answer}
        {self.refrain_feedback}
        Feedback: {feedback}
        {self.refrain_refine}
        Refined Implementation:
        """

        final_prompt = few_shot + prompt
        return final_prompt

    def run(self, question: str, answer: str):
        feedback = self.feedback(question, answer, self.get_examples())
        return feedback
