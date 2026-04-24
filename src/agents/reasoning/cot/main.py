from typing import Optional, List, Union, Dict
import json
from openai import OpenAI
import os
import sys
sys.path.insert(0, os.getcwd())
from src.agent_constructor.agent import Agent


class CoT(Agent):
    def __init__(
        self, 
        name: str = "CoT", 
        mode: str = "zero-shot", 
        few_shot_examples: Optional[Union[str, List[Dict[str, str]]]] = None,
        model_url: str = "http://localhost:7215/v1",
        model_name: str = "Qwen/Qwen1.5-32B-Chat-AWQ",
        temperature: float = 0.2,
        top_p: float = 0.95, 
        max_tokens: int = 1024,
        stop_tokens: List[str] = ["</code>", "# SOLUTION END"]
    ):
        """
        Args:
            name: method name
            mode: "zero-shot" or "few-shot"
            few_shot_examples: Either path to file with examples or list of examples
        """
        super().__init__(name)
        self.mode = mode
        self.examples = self.load_examples(few_shot_examples)
        self.model_name = model_name
        self.temperature = temperature
        self.top_p=top_p
        self.max_tokens=max_tokens
        self.stop_tokens=stop_tokens
        
        self.client = OpenAI(
            base_url=model_url,
            api_key="vllm"
        )
    
    def load_examples(
        self, 
        few_shot_examples: Optional[Union[str, List[Dict[str, str]]]]
    ) -> List[str]:
        """Load few-shot examples from file or use provided examples."""
        if self.mode != "few-shot" or few_shot_examples is None:
            return []

        if isinstance(few_shot_examples, str) and few_shot_examples.endswith('.jsonl'):
            examples_list = []
            with open(few_shot_examples, 'r', encoding='utf-8') as f:
                for line in f:
                    if line.strip():
                        examples_list.append(json.loads(line))
            return [example["example"] for example in examples_list]
        elif isinstance(few_shot_examples, list):
            return [example["example"] for example in few_shot_examples]
        else:
            print(f"Warning: Invalid examples type. Using zero-shot mode.")
            return []
    
    def make_prompt(self, task: str) -> str:
        """Build the prompt based on the mode."""
        prompt_parts = []
        
        if self.mode == "few-shot" and self.examples:
            # Add few-shot examples
            prompt_parts.append("Here are some examples of solving programming problems step by step:")
            for i, example in enumerate(self.examples, 1):
                prompt_parts.append(f"Example {i}:\n")
                prompt_parts.append(f"{example}\n")
            prompt_parts.append(
                "\nNow analyze the following problem. Provide step-by-step reasoning. "
                "After your reasoning, write the solution code after the line 'Solution:'.\n"
            )
        else:
            # Zero-shot mode
            prompt_parts.append("Think step by step to analyze this programming problem.")
            prompt_parts.append(
                "After your reasoning, write the solution code after the line 'Solution:'.\n"
            )
        
        # Add the task
        prompt_parts.append(task)
        prompt_parts.append("\nReasoning:")
        
        return "\n".join(prompt_parts)
    
    def generate(self, prompt: str) -> str:
        system_prompt = (
            "You are a helpful programming assistant. When given a problem, you should first think step by step "
            "and then provide the solution code after the line 'Solution:'. "
            "Make sure to include exactly 'Solution:' on its own line followed by the code. "
            "Do not add any extra text after the code."
        )
        
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt}
            ],
            temperature=self.temperature,
            top_p=self.top_p,
            max_tokens=self.max_tokens,
            stop=self.stop_tokens
        )
        
        return response.choices[0].message.content
    
    def extract_solution(self, response: str) -> str:
        marker = "Solution:\n"
        idx = response.find(marker)
        if idx == -1:
            return "Solution not found"

        return response[idx + len(marker):]

        
    def run(self, task: str) -> str:
        """
        Run the Chain-of-Thought process for the given task.
        """
        prompt = self.make_prompt(task)
        response = self.generate(prompt)
        solution = self.extract_solution(response)
        
        return solution