import random
# import spacy
from typing import List, Dict, Union, Optional
import json
from src.agent_constructor.agent import Agent


class ContrastiveCoT(Agent):
    def __init__(
        self,
        demonstrations: Union[str, List[Dict[str, str]]],
        name: str = "ContrastiveCoT"
    ):
        super().__init__(name)
        self.demonstrations = self.load_demonstrations(demonstrations)
        # entity recognition model
        # "en_core_web_trf" is used in the paper
        # self.nlp = spacy.load("en_core_web_sm")
        
    def load_demonstrations(
        self, 
        demonstrations: Union[str, List[Dict[str, str]]]
    ) -> List[Dict[str, str]]:
        """
        Args:
            demonstrations: Either a list of example dictionaries or a path to JSON file
                            Each example should have:
                            - problem: str
                            - correct_explanation: str
                            - correct_code: str
                            - incorrect_explanation: str (optional)
                            - incorrect_code: str (optional)
        """
        if isinstance(demonstrations, str):
            with open(demonstrations, 'r', encoding='utf-8') as f:
                data = json.load(f)
                return data
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
    
    def format_programming_prompt(self, demonstrations: List[Dict], task: str) -> str:
        """
        Format prompt for programming tasks with contrastive examples.
        """
        prompt = """Solve the following programming problem using step-by-step reasoning.
I'll show you both correct and incorrect approaches to help you avoid common mistakes.

"""
        
        for i, demo in enumerate(demonstrations, 1):
            prompt += f"Example {i}:\n"
            prompt += f"Problem: {demo['problem']}\n\n"
            
            prompt += f"Correct reasoning: {demo['correct_explanation']}\n\n"
            prompt += f"Correct answer: {demo['correct_code']}\n\n"
            
            prompt += f"Incorrect reasoning: {demo['incorrect_explanation']}\n\n"
            prompt += f"Incorrect answer: {demo['incorrect_code']}\n\n"
            prompt += "---\n\n"
        
        prompt += f"""Now solve this programming problem. Think step by step and provide:
1. Your reasoning process
2. The complete code solution

Problem: {task}

Reasoning:"""
        
        return prompt
    
    def llm(self, prompt: str) -> str:
        # TODO: Replace with actual LLM call
        return ""
    
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
        
        prompt = self.format_programming_prompt(contrastive_demos, task)
        response = self.llm(prompt)
        
        return response