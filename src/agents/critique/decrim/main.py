import importlib

from src.agent_constructor.agent import Agent
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage


def _load_decrim_prompts(dataset: str):
    """Load DeCRIM prompt templates for the given dataset (ds1000, codemmlu, ...).

    Each ``prompts_<dataset>`` module exposes the same public names, so switching
    benchmarks changes only which module is imported — selected via the ``dataset`` param.
    """
    module = importlib.import_module(
        f"src.agents.critique.decrim.prompts_{dataset}"
    )
    return module.DECOMPOSE_PROMPT, module.CRITIQUE_PROMPT, module.CONSTRAINTS_HEADER


class Decrim(Agent):
    """DeCRIM agent implementing the Decompose, Critique, and Refine pipeline."""

    def __init__(
        self,
        name: str = "DeCRIM",
        model_name: str = "Qwen/Qwen1.5-32B-Chat-AWQ",
        openai_api_base_url="http://localhost:7215/v1",
        dataset: str = "ds1000",
    ):
        super().__init__(name)
        self.dataset = dataset
        (
            self.decompose_prompt,
            self.critique_prompt,
            self.constraints_header,
        ) = _load_decrim_prompts(dataset)
        self.llm_model = ChatOpenAI(
            model=model_name,
            openai_api_base=openai_api_base_url,
            openai_api_key="fake-key",
            temperature=0.7,
        )

    def llm(self, message: str) -> str:
        # TODO: different models for decomposition and feedback?

        messages = [HumanMessage(content=message)]

        response = self.llm_model.invoke(messages)
        return response.content

    def _parse_constraints(self, constraints_response: str) -> list[str]:
        """Parse constraints from LLM response."""
        # Extract numbered constraints from the response
        lines = constraints_response.strip().split("\n")
        constraint_list = []

        for line in lines:
            line = line.strip()
            # Look for lines starting with numbers (1., 2., etc.)
            if line and line[0].isdigit():
                # Remove the number and any following punctuation
                # TODO: парсит, только если нумерация с точкой.
                constraint = line.split(".", 1)[1].strip() if "." in line else line
                constraint_list.append(constraint)

        return (
            constraint_list
            if constraint_list
            else ["Follow all instructions carefully"]
        )

    def _create_decompose_prompt(self, question: str) -> str:
        """Build prompt for decomposition step (dataset-specific template)."""
        return self.decompose_prompt.format(question=question)

    def _create_critique_prompt(
        self, question: str, constraints_text: str, answer: str
    ) -> str:
        """Build prompt for critique step (dataset-specific template)."""
        return self.critique_prompt.format(
            question=question, constraints_text=constraints_text, answer=answer
        )

    def _decompose(self, question: str) -> str:
        decompose_prompt = self._create_decompose_prompt(question)
        constraints = self.llm(decompose_prompt)

        constraint_list = self._parse_constraints(constraints)
        constraints_text = "\n".join(
            [f"{i+1}. {constraint}" for i, constraint in enumerate(constraint_list)]
        )
        constraints_text = f"{self.constraints_header}\n" + constraints_text

        return constraints_text

    def _critique(self, question: str, answer: str, constraints_text: str) -> str:
        critique_prompt = self._create_critique_prompt(
            question, constraints_text, answer
        )
        critique_result = self.llm(critique_prompt)

        return f"{constraints_text}\n\n{critique_result}"

    def run(self, question: str, answer: str) -> str:
        """
        Args:
            question: User instruction with multiple constraints
            answer: LLM response

        Returns:
            Feedback
        """

        constraints = self._decompose(question)
        feedback = self._critique(question, answer, constraints)

        return feedback
