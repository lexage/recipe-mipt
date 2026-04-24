import os
import sys
sys.path.insert(0, os.getcwd())

from openai import OpenAI
import re
import json
from typing import List, Optional, Dict, Union
from src.agent_constructor.agent import Agent


class PlanAndSolve(Agent):
    def __init__(
        self, 
        examples: Optional[Union[str, List[Dict[str, str]]]],
        name: str = "PlanAndSolve",
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
        
    def remove_problem_prefix(self, text: str) -> str:
        return re.sub(r'^Problem:\s*', '', text, flags=re.IGNORECASE)
    
    def make_plan_prompt(self, task: str) -> str:
        clean_task = self.remove_problem_prefix(task)
        prompt_parts = []
        
        prompt_parts.append("Here are some examples of decomposing programming problems into subproblems:\n")
        
        for i, example in enumerate(self.examples, 1):
                prompt_parts.append(f"Example {i}:\n")
                prompt_parts.append(f"{example}\n")
        
        prompt_parts.append(
            "Generate a detailed plan to solve the following programming task. "
            "Break the task down into smaller, well-defined subtasks. "
            "Present the plan as a numbered list, with each subtask on a new line. "
            "Use numbering with a dot (e.g., 1., 2., 3., ...). "
            "Ensure that the subtasks are logically ordered and cover all necessary steps to complete the original task. "
            "Provide only the plan; do not generate the actual code or solution for the task.\n"
        )
        
        prompt_parts.append(f"Problem:\n{clean_task}\n")
        prompt_parts.append("Plan:")
        
        return "\n".join(prompt_parts)
    
    def generate(self, prompt: str) -> str:
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=[
                {"role": "user", "content": prompt}
            ],
            temperature=self.temperature,
        )
        
        return response.choices[0].message.content
    
    def run(self, task: str) -> str:
        plan_prompt = self.make_plan_prompt(task)
        print(plan_prompt)
        plan = self.generate(plan_prompt)
        return plan