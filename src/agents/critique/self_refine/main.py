from abc import ABC, abstractmethod
from src.agent_constructor.agent import Agent


class Example(str):
    question: str
    answer: str
    feedback: str
    refined: str


class SelfRefine(Agent):
    """
    Agent that implements Self-Refine: Iterative Refinement with Self-Feedback.

    Prompts in original paper are task-specific.
    """

    def __init__(self, name: str = "Self-Refine", critique_only: bool = True):
        super().__init__(name)
        self.critique_only = critique_only
        self.refrain_feedback: str = (
            "Can you give a suggestion to improve the answer? "
            "Don't fix it, just give a suggestion."
        )
        self.refrain_refine: str = "Now fix the answer."

    def llm(self, prompt: str) -> str:
        # TODO: implement actual LLM call
        return ""

    def feedback_prompt(
        self, question: str, answer: str, examples: list[Example] = []
    ) -> str:
        few_shot: str = ""
        for example in examples:
            few_shot += f"""
Question: {example.question}
Proposed Answer: {example.answer}
{self.refrain_feedback}
Feedback: {example.feedback}
"""

        prompt = f"""
Question: {question}
Proposed Answer: {answer}
{self.refrain_feedback}
Feedback:
"""

        final_prompt = few_shot + prompt
        return final_prompt

    def feedback(self, question: str, answer: str, examples: list[Example] = []) -> str:
        feedback_prompt = self.feedback_prompt(question, answer, examples)
        # print(feedback_prompt)
        result = self.llm(feedback_prompt)
        return result

    def refine_prompt(
        self, question: str, answer: str, feedback: str, examples: list[Example] = []
    ) -> str:
        few_shot: str = ""
        for example in examples:
            few_shot += f"""
Question: {example.question}
Proposed Answer: {example.answer}
{self.refrain_feedback}
Feedback: {example.feedback}
{self.refrain_refine}
Refined answer: {example.refined}
"""

        prompt = f"""
Question: {question}
Answer: {answer}
{self.refrain_feedback}
Feedback: {feedback}
{self.refrain_refine}
Refined answer:
"""

        final_prompt = few_shot + prompt
        return final_prompt

    def refine(
        self, question: str, answer: str, feedback: str, examples: list[Example] = []
    ) -> str:
        refine_prompt = self.refine_prompt(question, answer, feedback, examples)
        # print(refine_prompt)
        result = self.llm(refine_prompt)
        return result

    def run(self, question: str, answer: str):
        feedback = self.feedback(question, answer)
        if self.critique_only:
            return feedback

        improved_answer = self.refine(question, answer, feedback)
        return improved_answer
