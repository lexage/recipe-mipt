from src.agent_constructor.agent import Agent
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage


class Decrim(Agent):
    """DeCRIM agent implementing the Decompose, Critique, and Refine pipeline."""

    def __init__(
        self,
        name: str = "DeCRIM",
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
        """Build prompt for decomposition step."""
        # TODO: add few-shot examples
        # decompose_prompt = f"""
        # You are an assistant whose job is to help me perform tasks.
        # I will give you a coding instruction that implicitly contains constraints to be followed.
        # Your task is to list the constraints provided by the user in an enumerated list format.
        # The constraints should help validate that the instruction will be carried out correctly,
        # and the resulting code will be correct and follow the syntax of the specified programming language.

        # Original Instruction: {question}

        # Provided Constraints:
        # """

        decompose_prompt = f"""
        You are an assistant whose job is to help me perform tasks.
        I will give you a coding instruction that implicitly contains constraints to be followed.
        Your task is to extract and list the constraints in a STRICT enumerated format.

        FORMAT REQUIREMENTS:
        - Use ONLY a numbered list where each line starts with a digit followed immediately by a dot and a space (e.g., "1. ", "2. ", "3. ").
        - Each constraint must be on a separate line.
        - Do NOT use markdown formatting, parentheses, dashes, Roman numerals, or any other numbering styles.
        - Output ONLY the list itself, without introductory text, explanations, or concluding remarks.

        Original Instruction: {question}

        Provided Constraints:
        """
        return decompose_prompt

    def _create_critique_prompt(
        self, question: str, constraints_text: str, answer: str
    ) -> str:
        """Build prompt for critique step."""

        critique_prompt = f"""
        You are an assistant whose job is to help me perform tasks.
        I will give you a coding instruction and an AI assistant response.
        The response should be a valid and correct piece of code which follows the syntax of the programming language.
        The instruction includes some constraints to be followed by AI assistant while generating response.
        Your task is to check and let me know which of the constraints are satisfied by the AI assistant response.
        Please state short reasons on whether constraint is satisfied in the response or not.
        Also include final answer as "Constraint followed" or "Constraint not followed" for each constraint.

        Instruction: {question}

        {constraints_text}

        Assistant Response: {answer}

        Please analyze each constraint one by one and provide your critique:
        """
        return critique_prompt

    def _decompose(self, question: str) -> str:
        decompose_prompt = self._create_decompose_prompt(question)
        constraints = self.llm(decompose_prompt)

        constraint_list = self._parse_constraints(constraints)
        constraints_text = "\n".join(
            [f"{i+1}. {constraint}" for i, constraint in enumerate(constraint_list)]
        )
        constraints_text = "Constraints:\n" + constraints_text

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
