from abc import ABC, abstractmethod


class Agent(ABC):
    """Base class for all agents in the system."""

    def __init__(self, name: str):
        self.name = name

    @abstractmethod
    def run(self, *args, **kwargs):
        raise NotImplementedError

    def __str__(self):
        return f"{self.__class__.__name__}({self.name})"


class Example(str):
    question: str
    answer: str
    critique: str


class Critic(Agent):
    """
    Agent that implements CRITIC: Large Language Models Can Self-Correct with Tool-Interactive Critiquing.
    Finds faults in model response via external tools or few-shot prompting.

    Few-shot examples in original paper are task-specific.
    """

    def __init__(
        self,
        name: str = "CRITIC",
        use_tools: bool = False,
        tools: list = [],
        examples: list[Example] = [],
    ):
        super().__init__(name)
        self.refrain = "What's the problem with the above answer?"
        self.use_tools = use_tools
        self.tools = tools
        self.examples = examples

    def llm(self, prompt: str) -> str:
        # TODO: Replace with actual LLM call
        return ""

    def make_prompt(self, question: str, answer: str, examples: list[Example]) -> str:
        few_shot: str = ""
        for example in examples:
            few_shot += f"""
Question: {example.question}
Proposed Answer: {example.answer}

{self.refrain}

{example.critique}
"""

        prompt = f"""
Question: {question}
Proposed Answer: {answer}

{self.refrain}
"""

        final_prompt = few_shot + prompt
        return final_prompt

    def tool_critique(self, question: str, response: str) -> str:
        # TODO: implement tool calls
        tool_criticisms = []
        for tool in self.tools:
            res = ""
            tool_criticisms.append(res)
        return "\n".join(tool_criticisms)

    def llm_critique(self, question: str, response: str) -> str:
        prompt = self.make_prompt(question, response, self.examples)
        # print(prompt)
        res = self.llm(prompt)
        return res

    def run(self, question: str, response: str) -> str:
        # TODO: verify then correct

        if self.use_tools:
            return self.tool_critique(question, response)
        else:
            return self.llm_critique(question, response)
