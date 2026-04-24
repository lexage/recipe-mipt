import os
import sys
sys.path.insert(0, os.getcwd())

from openai import OpenAI
from typing import List, Dict, Union, Optional
import json
import re
from src.agent_constructor.agent import Agent


class ContrastiveCoT(Agent):
    def __init__(
        self,
        demonstrations: Union[str, List[Dict[str, str]]],
        name: str = "ContrastiveCoT",
        model_url: str = "http://localhost:7215/v1",
        model_name: str = "Qwen/Qwen1.5-32B-Chat-AWQ",
        temperature: float = 0.2,
        top_p: float = 0.95, 
        max_tokens: int = 1024,
        stop_tokens: List[str] = ["</code>", "# SOLUTION END"]
    ):
        super().__init__(name)
        self.demonstrations = self.load_demonstrations(demonstrations)
        self.model_name = model_name
        self.temperature = temperature
        self.top_p=top_p
        self.max_tokens=max_tokens
        self.stop_tokens=stop_tokens
        
        self.client = OpenAI(
            base_url=model_url,
            api_key="vllm"
        )
        
    def load_demonstrations(
        self, 
        demonstrations: Union[str, List[Dict[str, str]]]
    ) -> List[Dict[str, str]]:
        """
        Args:
            demonstrations: Either a list of example dictionaries or a path to JSONL file
                            Each example should have:
                            - problem: str
                            - correct_explanation: str
                            - correct_code: str
                            - incorrect_explanation: str (optional)
                            - incorrect_code: str (optional)
        """
        if isinstance(demonstrations, str) and demonstrations.endswith('.jsonl'):
            examples_list = []
            with open(demonstrations, 'r', encoding='utf-8') as f:
                for line in f:
                    if line.strip():
                        examples_list.append(json.loads(line))
            return examples_list
        elif isinstance(demonstrations, list):
            return demonstrations
        else:
            print(f"Warning: Invalid demonstrations type.")
            return []
        
    def extract_objects(self, code: str, explanation: str) -> List[str]:
        """
        Extract programming-related objects from code and explanation.
        """
         # TODO: Think about which entities should be extracted. Add real implementation.
        
        return [""]
    
    def create_incorrect_explanation(
        self,
        correct_explanation: str,
        correct_code: str,
        incorrect_explanation: Optional[str] = None
    ) -> str:
        """
        Create incoherent explanation by shuffling programming entities.
        """
        # TODO: Add implementation
        if incorrect_explanation:
            return incorrect_explanation
        
        objects = self.extract_objects(correct_code, correct_explanation)
        
        return ""
    
    def create_incorrect_code(
        self,
        correct_code: str,
        incorrect_code: Optional[str] = None
    ) -> str:
        """
        Create incorrect version of code by introducing common programming errors.
        """
        # TODO: Add real implementation
        if incorrect_code:
            return incorrect_code
        
        return ""
    
    def create_contrastive_demonstration(
        self, 
        problem: str,
        correct_explanation: str,
        correct_code: str,
        incorrect_explanation: Optional[str] = None,
        incorrect_code: Optional[str] = None
    ) -> Dict:
        """
        Create contrastive demonstration for programming task.
        """
        incorrect_explanation = self.create_incorrect_explanation(
            correct_explanation, correct_code, incorrect_explanation
        )
        incorrect_code = self.create_incorrect_code(correct_code, incorrect_code)
        
        return {
            "problem": problem,
            "correct_explanation": correct_explanation,
            "correct_code": correct_code,
            "incorrect_explanation": incorrect_explanation,
            "incorrect_code": incorrect_code
        }
        
    def remove_problem_prefix(self, text: str) -> str:
        return re.sub(r'^Problem:\s*', '', text, flags=re.IGNORECASE)
    
    def make_prompt(self, demonstrations: List[Dict], task: str) -> str:
        """
        Format prompt for programming tasks with contrastive examples.
        """
        prompt = """You are a programming assistant. For each problem, first reason step by step, then provide the solution code after the line 'Correct solution:'. Do not add any extra text after the code.

Below are examples showing both correct and incorrect approaches, each with reasoning and the corresponding solution.

"""
        
        for i, demo in enumerate(demonstrations, 1):
            prompt += f"Example {i}:\n\n"
            clean_problem = self.remove_problem_prefix(demo['problem'])
            prompt += f"Problem:\n{clean_problem}\n\n"
            
            prompt += f"Correct reasoning:\n{demo['correct_explanation']}\n\n"
            prompt += f"Correct solution:\n{demo['correct_code']}\n\n"
            
            prompt += f"Incorrect reasoning:\n{demo['incorrect_explanation']}\n\n"
            prompt += f"Incorrect solution:\n{demo['incorrect_code']}\n\n"
            prompt += "---\n\n"
            
        clean_task = self.remove_problem_prefix(task)
        prompt += f"""Now analyze the following problem. Remember to provide step‑by‑step reasoning, then write the solution code after the line 'Correct solution:'.

Problem:\n{clean_task}

Correct reasoning:"""
        
        return prompt
    
    def generate(self, prompt: str) -> str:
        system_prompt = (
            "You are a helpful programming assistant. When given a problem, you should first think step by step "
            "and then provide the solution code after the line 'Correct solution:'. "
            "Make sure to include exactly 'Correct solution:' on its own line followed by the code. "
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
        marker = "Correct solution:\n"
        idx = response.find(marker)
        if idx == -1:
            return "Solution not found"

        return response[idx + len(marker):]
    
    def run(self, task: str) -> str:
        """
        Args:
            task: Description of the programming task to solve
        """
        contrastive_demos = []
        for demo in self.demonstrations:
            contrastive_demo = self.create_contrastive_demonstration(
                demo["problem"],
                demo["correct_explanation"],
                demo["correct_code"],
                demo.get("incorrect_explanation"),
                demo.get("incorrect_code")
            )
            contrastive_demos.append(contrastive_demo)
        
        prompt = self.make_prompt(contrastive_demos, task)
        response = self.generate(prompt)
        solution = self.extract_solution(response)
        
        return solution