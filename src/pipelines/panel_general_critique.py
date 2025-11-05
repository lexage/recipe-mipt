import logging
from typing import List

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage

from src.agent_constructor.agent import Agent


_LOG_SEPARATOR = f"\n{'_' * 20}\n"


class PanelAgentV2(Agent):
    def __init__(
        self,
        name: str = "Panel",
        model_name: str = "Qwen/Qwen1.5-32B-Chat-AWQ",
        url="http://localhost:7215/v1",
        temperature: float = 0.6,
        num_candidates: int = 5,
    ):
        super().__init__(name)
        self.temperature = temperature
        self.num_candidates = num_candidates
        self.llm_model = ChatOpenAI(
            model=model_name,
            openai_api_base=url,
            openai_api_key="fake-key",
            temperature=self.temperature,
        )

    def llm(self, message) -> str:
        messages = [HumanMessage(content=message)]
        response = self.llm_model.invoke(messages)
        return response.content

    def _create_sampling_prompt(
        self, problem: str, context: str, sampling_type: str
    ) -> str:
        """Create prompt for candidate generation."""
        base_prompt = f"""
You are an expert programmer solving the following task or problem:

PROBLEM/TASK: {problem}

CURRENT CONTEXT: {context}

Generate the required code block.
Focus on:
1. Logical correctness
2. Code quality and readability
3. Progress towards solving the problem

Respond with the answer
directly with no extra words."""

        if sampling_type == "greedy":
            base_prompt += "\n[Generate the most logical and correct code snippet]"
        else:  # random sampling
            base_prompt += "\n[Generate a diverse but valid code snippet]"

        return base_prompt

    def _sample_candidates(self, problem: str, context: str) -> List[str]:
        """Generate candidate reasoning steps using dual sampling strategy."""
        candidates = []

        # Greedy decoding
        greedy_prompt = self._create_sampling_prompt(problem, context, "greedy")
        greedy_candidate = self.llm(greedy_prompt)
        candidates.append(greedy_candidate)

        # Random sampling with temperature
        for i in range(self.num_candidates - 1):
            random_prompt = self._create_sampling_prompt(problem, context, "random")
            random_candidate = self.llm(random_prompt)
            candidates.append(random_candidate)

        return candidates

    def _create_general_critique_prompt(
        self, candidates: List[str], problem: str, context: str
    ) -> str:
        """Create prompt for generating a general critique of all candidates."""
        candidates_text = ""
        for i, cand in enumerate(candidates):
            candidates_text += f"CANDIDATE {i}:\n{cand}\n\n"

        return f"""
You are an expert programmer specializing in code review and debugging.
You are given several candidate next steps for solving a programming problem.

PROBLEM: {problem}
CURRENT CONTEXT: {context}

CANDIDATES:
{candidates_text}

Your task is to provide a GENERAL CRITIQUE that:
1. Identifies common strengths across the candidates.
2. Highlights recurring errors, logical flaws, or misunderstandings.
3. Points out important ideas that appear in multiple candidates.
4. Summarizes the overall quality and progress of the set of candidates.

Do NOT simply repeat each candidate's critique. Instead, synthesize the information.
Focus on patterns, common issues, and general observations that apply to multiple candidates.

GENERAL CRITIQUE:"""

    def _generate_general_critique(
        self, candidates: List[str], problem: str, context: str
    ) -> str:
        """Generate a single general critique summarizing all candidates."""
        prompt = self._create_general_critique_prompt(candidates, problem, context)
        return self.llm(prompt)

    def _create_selection_prompt(
        self, candidates: List[str], general_critique: str
    ) -> str:
        """Create prompt for candidate selection based on the general critique."""
        candidates_text = ""
        for i, cand in enumerate(candidates):
            candidates_text += f"CANDIDATE {i}:\n{cand}\n\n"

        return f"""You are an expert programmer. Select the best next reasoning step based on the candidates and the general critique below.

GENERAL CRITIQUE (common strengths, errors, and patterns):
{general_critique}

CANDIDATES:
{candidates_text}

Based on the general critique, select the candidate that:
1. Has the fewest logical errors
2. Makes the most progress toward solving the problem
3. Has the best code quality
4. Avoids the common pitfalls mentioned in the general critique

Return only the index number (0-based) of the selected candidate:"""

    def _extract_candidate_index(self, selection_response: str) -> int:
        prompt = f"""You are a Python programmer. You are given a reasoning about the index of the best candidate. 
                    You should return only one integer (the index of the best candidate) (without any other words) from this reasoning.
                    I will have to insert it into int(index). Reasoning : {selection_response}"""
        response_index = self.llm(prompt)
        return int(response_index)

    def _select_best_candidate(
        self, candidates: List[str], general_critique: str
    ) -> str:
        """Select the best candidate based on the general critique."""
        if not candidates:
            return ""

        selection_prompt = self._create_selection_prompt(candidates, general_critique)
        selection_response = self.llm(selection_prompt)
        selected_index = self._extract_candidate_index(selection_response)

        return candidates[selected_index]

    def run(self, task: str, current_context: str = "") -> str:
        """
        Args:
            task: The programming problem to solve
            current_context: Current reasoning context/code so far

        Returns:
            Selected next reasoning step/code
        """
        # Step 1: generate N candidates
        candidates = self._sample_candidates(task, current_context)

        logging.info(f"TASK: {task}")
        logging.info(_LOG_SEPARATOR)

        for idx, candidate in enumerate(candidates):
            logging.info(f"CANDIDATE {idx+1}: {candidate}\n")

        logging.info(_LOG_SEPARATOR)

        # Step 2: generate a single general critique for all candidates
        general_critique = self._generate_general_critique(candidates, task, current_context)

        logging.info(f"GENERAL CRITIQUE:\n{general_critique}")
        logging.info(_LOG_SEPARATOR)

        # Step 3: select the best candidate using the general critique
        selected_candidate = self._select_best_candidate(candidates, general_critique)

        logging.info(f"SELECTED CANDIDATE: {selected_candidate}")
        logging.info(_LOG_SEPARATOR)

        return selected_candidate