from typing import Optional, List, Union, Dict
from src.agent_constructor.agent import Agent
import json


class CoT(Agent):
    def __init__(
        self, 
        name: str = "CoT", 
        mode: str = "zero-shot", 
        few_shot_examples: Optional[Union[str, List[Dict[str, str]]]] = None
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
            prompt_parts.append("NOTE: Only show your reasoning process. Do NOT provide final solution.")
            for i, example in enumerate(self.examples, 1):
                prompt_parts.append(f"Example {i}:")
                prompt_parts.append(example)
            prompt_parts.append("\nNow analyze the following problem. Provide ONLY step-by-step reasoning:")
        else:
            # Zero-shot mode
            prompt_parts.append("Think step by step to analyze this programming problem.")
            prompt_parts.append("Provide ONLY your reasoning process. Do NOT write final solution.")
        
        # Add the task
        prompt_parts.append(f"Problem: {task}")
        prompt_parts.append("\nReasoning:")
        
        return "\n".join(prompt_parts)
    
    def llm(self, prompt: str) -> str:
        # TODO: Replace with actual LLM call
        return ""
        
    def run(self, task: str) -> str:
        """
        Run the Chain-of-Thought process for the given task.
        """
        prompt = self.make_prompt(task)
        reasoning = self.llm(prompt)
        
        return reasoning