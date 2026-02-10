import random
from typing import List
from src.agent_constructor.agent import Agent
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, ToolMessage


class Panel(Agent):
    def __init__(self, 
                 name: str = "Panel", 
                model_name: str = "Qwen/Qwen1.5-32B-Chat-AWQ",
                openai_api_base_url="http://localhost:7215/v1"):
        
        super().__init__(name)
        self.temperature = 0.6
        self.num_candidates = 5
        self.llm_model = ChatOpenAI(
            model=model_name,
            openai_api_base=openai_api_base_url,
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
You are an expert programmer solving the following problem:

PROBLEM: {problem}

CURRENT CONTEXT: {context}

Generate the next reasoning step or code snippet. Focus on:
1. Logical correctness
2. Code quality and readability
3. Progress towards solving the problem

Next step:"""

        if sampling_type == "greedy":
            base_prompt += "\n[Generate the most logical and correct next step]"
        else:  # random sampling
            base_prompt += "\n[Generate a diverse but valid next step]"

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

    def _create_critique_prompt(
        self, problem: str, context: str, candidate: str
    ) -> str:
        """Create prompt for NL self-critique of programming steps."""
        return f"""
You are an expert programmer specializing in code review and debugging. Your task is to critique the next proposed reasoning step for solving a programming problem.

PROBLEM: {problem}
CURRENT CONTEXT: {context}
PROPOSED NEXT STEP: {candidate}

Evaluate this step for:
1. Logical errors in reasoning
2. Syntax and semantic correctness
3. Conceptual misunderstandings
4. Potential bugs or edge cases
5. Alignment with problem requirements

Provide a concise natural language critique focusing on the core issues. If the step is correct, explain why. If incorrect, pinpoint the specific errors.

CRITIQUE:"""

    def _generate_critiques(
        self, candidates: List[str], problem: str, context: str
    ) -> List[str]:
        """Generate natural language critiques for each candidate."""
        critiques = []

        for candidate in candidates:
            critique_prompt = self._create_critique_prompt(problem, context, candidate)
            critique = self.llm(critique_prompt)
            critiques.append(critique)

        return critiques

    def _create_selection_prompt(
        self, candidates: List[str], critiques: List[str]
    ) -> str:
        """Create prompt for candidate selection based on critiques."""
        prompt = "You are an expert programmer. Select the best next reasoning step based on the following critiques:\n\n"

        for i, (candidate, critique) in enumerate(zip(candidates, critiques)):
            prompt += f"CANDIDATE {i}:\n{candidate}\n"
            prompt += f"CRITIQUE {i}:\n{critique}\n\n"

        prompt += """Based on the critiques, select the candidate that:
1. Has the fewest logical errors
2. Makes the most progress toward solving the problem
3. Has the best code quality

Return only the index number (0-based) of the selected candidate:"""

        return prompt

    def _extract_candidate_index(
        self, selection_response: str
    ) -> int:
        prompt = f"""You are a Python programmer. You are given a reasoning about the index of the best candidate. 
                    You should return only one integer (the index of the best candidate) (without any other words) from this reasoning.
                    I will have to insert it into int(index). Reasoning : {selection_response}"""
        responce_index = self.llm(prompt)
        return int(responce_index)

    def _select_best_candidate(
        self, candidates: List[str], critiques: List[str]
    ) -> str:
        """Select the best candidate based on NL critiques."""
        if not candidates:
            return ""

        selection_prompt = self._create_selection_prompt(candidates, critiques)
        selection_response = self.llm(selection_prompt)
        selected_index = self._extract_candidate_index(
            selection_response
        )

        return candidates[selected_index]

    def run(self, task: str, current_context: str = "") -> str:
        """
        Args:
            task: The programming problem to solve
            current_context: Current reasoning context/code so far

        Returns:
            Selected next reasoning step/code
        """
        candidates = self._sample_candidates(task, current_context)

        critiques = self._generate_critiques(candidates, task, current_context)

        selected_candidate = self._select_best_candidate(candidates, critiques)

        return selected_candidate
