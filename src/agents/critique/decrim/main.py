from src.agent_constructor.agent import Agent
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage


class Decrim(Agent):
    """DeCRIM agent implementing the Decompose, Critique, and Refine pipeline."""

    def __init__(
        self,
        name: str = "Decrim",
        max_iterations: int = 10,
        model_name: str = "Qwen/Qwen1.5-32B-Chat-AWQ",
        openai_api_base_url="http://localhost:7215/v1",
    ):
        super().__init__(name)
        self.max_iterations = max_iterations
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

    def _create_critique_prompt(
        self, instruction: str, constraints: list[str], response: str
    ) -> str:
        """Build prompt for critique step."""
        constraints_text = "\n".join(
            [f"{i+1}. {constraint}" for i, constraint in enumerate(constraints)]
        )

        return f"""
        You are an assistant whose job is to help me perform tasks.
        I will give you an instruction and an AI assistant response.
        The instruction includes some constraints to be followed by AI assistant while generating response.
        Your task is to check and let me know which of the constraints are satisfied by the AI assistant response.
        Please state short reasons on whether constraint is satisfied in the response or not.
        Also include final answer as "Constraint followed" or "Constraint not followed" for each constraint.

        Instruction: {instruction}

        Constraints:
        {constraints_text}

        Assistant Response: {response}

        Please analyze each constraint one by one and provide your critique:
        """

    def _all_constraints_satisfied(self, critique_result: str) -> bool:
        """Check if all constraints are satisfied based on critique result."""
        # TODO: Add real parsing
        unsatisfied_indicators = [
            "not followed",
            "not satisfied",
            "unsatisfied",
            "failed",
            "missing",
        ]

        for indicator in unsatisfied_indicators:
            if indicator in critique_result.lower():
                return False
        return True

    def _extract_unsatisfied_constraints(
        self, critique_result: str, all_constraints: list[str]
    ) -> list[str]:
        """Extract list of unsatisfied constraints from critique response."""
        # TODO: Add real extraction
        unsatisfied = []
        lower_result = critique_result.lower()

        for i, constraint in enumerate(all_constraints):
            constraint_lower = constraint.lower()
            if any(
                indicator in lower_result
                for indicator in [f"constraint {i+1} not", f"{constraint_lower} not"]
            ):
                unsatisfied.append(constraint)

        return unsatisfied if unsatisfied else all_constraints

    def _decompose(self, question: str) -> list[str]:
        # TODO: add few-shot examples
        decompose_prompt = f"""
        You are an assistant whose job is to help me perform tasks.
        I will give you an instruction that implicitly contains constraints to be followed.
        Your task is to list the constraints provided by the user in an enumerated list format.

        Original Instruction: {question}

        Provided Constraints:
        """
        constraints = self.llm(decompose_prompt, model_type="decompose")
        constraint_list = self._parse_constraints(constraints)
        return constraint_list

    def _critique(self, question: str, answer: str, constraints: list[str]) -> str:
        critique_prompt = self._create_critique_prompt(question, constraints, answer)
        critique_result = self.llm(critique_prompt)

        # TODO: edit or delete conclusion?
        if self._all_constraints_satisfied(critique_result):
            conclusion = "All constraints satisfied"
        else:
            unsatisfied_constraints = self._extract_unsatisfied_constraints(
                critique_result, constraints
            )
            conclusion = (
                f"Unsatisfied constraints: {', '.join(unsatisfied_constraints)}"
            )

        return f"{critique_result}\n{conclusion}"

    def run(self, question: str, answer: str) -> str:
        """
        Args:
            question: User instruction with multiple constraints
            answer: LLM response

        Returns:
            Feedback
        """

        constraints = self._decompose(question, answer)
        feedback = self._critique(question, answer, constraints)

        return feedback
