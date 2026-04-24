import os
import sys
sys.path.insert(0, os.getcwd())

from openai import OpenAI
import json
import re
from typing import List, Optional, Dict, Union
from src.agent_constructor.agent import Agent


class LeastToMost(Agent):
    """
    Agent that implements Least-to-Most Prompting for automated programming tasks.
    Breaks down complex programming problems into simpler subproblems.
    """
    def __init__(
        self, 
        examples: Optional[Union[str, List[Dict[str, str]]]],
        name: str = "L2M_Planner",
        model_url: str = "http://localhost:7215/v1",
        model_name: str = "Qwen/Qwen1.5-32B-Chat-AWQ",
        temperature: float = 0
    ):
        super().__init__(name)
        self.examples = self.load_examples(examples)
        self.model_name = model_name
        self.temperature = temperature
        
        self.client = OpenAI(
            base_url=model_url,
            api_key="vllm"
        )
        
    def load_examples(
        self, 
        examples: Optional[Union[str, List[Dict[str, str]]]]
    ) -> List[str]:
        """Load few-shot examples from file or use provided examples."""
        if isinstance(examples, str) and examples.endswith('.jsonl'):
            examples_list = []
            with open(examples, 'r', encoding='utf-8') as f:
                for line in f:
                    if line.strip():
                        examples_list.append(json.loads(line))
            return [example["example"] for example in examples_list]
        elif isinstance(examples, list):
            return [example["example"] for example in examples]
        else:
            print(f"Warning: Invalid examples type.")
            return []
        
    def generate(self, prompt: str) -> str:
        system_prompt = (
            "You are an expert at decomposing programming tasks into a sequence of simpler, self‑contained subproblems. "
            "Given a programming problem, break it down so that: (1) each subproblem is solvable on its own, "
            "(2) the solution of each subproblem can be used in subsequent ones, "
            "(3) subproblems progress from simple to complex, and (4) the last subproblem is the original task. "
            "Output only the subproblems as a numbered list, each on a new line, starting with the number and a period. "
            "Do not add any extra commentary."
        )
        
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt}
            ],
            temperature=self.temperature,
        )
        
        return response.choices[0].message.content
    
    def extract_subproblems(self, response: str) -> List[str]:
        subproblems = []
        for line in response.splitlines():
            match = re.match(r'^\s*\d+\.\s+(.*)', line)
            if match:
                subproblem = match.group(1).rstrip()
                subproblems.append(subproblem)
        return subproblems

    
    def remove_problem_prefix(self, text: str) -> str:
        return re.sub(r'^Problem:\s*', '', text, flags=re.IGNORECASE)
    
    def make_prompt(self, task: str) -> str:
        clean_task = self.remove_problem_prefix(task)
        prompt_parts = []
        
        prompt_parts.append("Here are some examples of decomposing programming problems into subproblems:\n")
        
        for i, example in enumerate(self.examples, 1):
                prompt_parts.append(f"Example {i}:\n")
                prompt_parts.append(f"{example}\n")
                
        prompt_parts.append(
            "You are an expert at decomposing programming tasks. "
            "Break down the following programming task into a sequence of simpler subproblems that need to be solved in order.\n\n"
            f"Problem:\n{clean_task}\n"
        )
        prompt_parts.append(
            "Break down the task into subproblems such that:\n"
            "1. Each subproblem is self-contained and solvable\n"
            "2. The solution to each subproblem can be used in subsequent subproblems\n"
            "3. The last subproblem should be the original task\n"
            "4. Subproblems should progress from simple to complex\n"
        )
        prompt_parts.append("Response format - simply list the subproblems, each on a new line, starting with a number and period.\n")
        prompt_parts.append("Subproblems:")
                
        return "\n".join(prompt_parts)
    
    def decompose(self, task: str) -> List[str]:
        """
        Decompose the main task into a sequence of simpler subproblems.
        Returns a list of subproblems.
        """
        decomposition_prompt = self.make_prompt(task)
        response = self.generate(decomposition_prompt)
        subproblems = self.extract_subproblems(response)
        
        return subproblems
    
    def run(self, task: str) -> List[str]:
        plan = self.decompose(task)
        return plan