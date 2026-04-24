import os
import sys
sys.path.insert(0, os.getcwd())

from openai import OpenAI

from typing import List, Dict, Optional
import json
import re
from src.agent_constructor.agent import Agent


def remove_problem_prefix(text: str) -> str:
    return re.sub(r'^Problem:\s*', '', text, flags=re.IGNORECASE)


class Selection:
    """Selection module for selecting relevant facts from context."""
    
    def __init__(
        self, 
        facts_from_context_examples: Optional[str] = None,
        facts_from_facts_examples: Optional[str] = None,
        model_url: str = "http://localhost:7215/v1",
        model_name: str = "Qwen/Qwen1.5-32B-Chat-AWQ",
        temperature: float = 0.
    ):
        """
        Args:
            examples: Few-shot examples for selection
            mode: 'simple' for direct generation or 'scoring' for scoring-based selection
        """
        self.facts_from_context_examples = self.load_examples(
            facts_from_context_examples,
            examples_type="context"
        )
        self.facts_from_facts_examples = self.load_examples(
            facts_from_facts_examples,
            examples_type="facts"
        )
        
        self.model_name = model_name
        self.temperature = temperature
        
        self.client = OpenAI(
            base_url=model_url,
            api_key="vllm"
        )
        
    def _format_context_examples(self, examples_list: List[Dict[str, str]]) -> str:
        formatted = ""
        for i, example in enumerate(examples_list):
            formatted += f"Example {i + 1}:\n\n"
            clean_task = remove_problem_prefix(example['task'])
            formatted += f"Task:\n{clean_task}\n\n"
            formatted += f"Facts:\n{example['facts']}\n\n"
            
        return formatted
    
    def _format_facts_examples(self, examples_list: List[Dict[str, str]]) -> str:
        formatted = ""
        for i, example in enumerate(examples_list):
            formatted += f"Example {i + 1}:\n\n"
            clean_task = remove_problem_prefix(example['task'])
            formatted += f"Task:\n{clean_task}\n\n"
            formatted += f"Facts:\n{example['facts']}\n\n"
            formatted += f"Selected facts:\n{example['selection']}\n\n"
            
        return formatted
        
    def load_examples(
        self, 
        examples: str, 
        examples_type: str = "context" # or "facts"
    ) -> str:
        """
        Load examples.
        
        Args:
            examples: Examples as string or path to file
            
        Returns:
            Formatted examples string
        """
        if examples is not None and examples.endswith('.jsonl'):
            examples_list = []
            with open(examples, 'r', encoding='utf-8') as f:
                for line in f:
                    if line.strip():
                        examples_list.append(json.loads(line))
            if examples_type == "context":
                return self._format_context_examples(examples_list)
            else:
                return self._format_facts_examples(examples_list)
        else:
            return []
        
    def _build_facts_from_context_prompt(self, question: str) -> str:
        prompt_parts = []
        
        prompt_parts.append("Here are some examples of extracting relevant facts from task context:\n")
        prompt_parts.append(f"{self.facts_from_context_examples}\n")
        
        clean_task = remove_problem_prefix(question)
        prompt_parts.append(f"Task:\n{clean_task}\n")
        prompt_parts.append("Facts:")
        
        return "\n".join(prompt_parts)
    
    def generate(self, prompt: str, system_prompt: str) -> str:
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt}
            ],
            temperature=self.temperature,
        )
        
        return response.choices[0].message.content
        
    def extract_facts_from_context(self, question: str) -> str:
        prompt = self._build_facts_from_context_prompt(question)
        
        system_prompt = "You are an AI assistant specialized in extracting facts from text. Your task is to read the provided programming task description, understand what needs to be done in this task, and extract all facts that are useful for solving it. Facts may include information about the problem statement, the format of the result, Python libraries/modules/functions that could be useful for the solution, as well as any other information you deem relevant. Output format: print only the facts themselves, each on a new line, without any list markers. If a fact contains any pieces of code, use special formatting (triple backticks for multi-line code, or inline backticks for short snippets))."

        facts = self.generate(prompt, system_prompt)
        return facts
        
    def _build_selection_prompt(self, facts: str, question: str, current_selection: str = "") -> str:
        """Build selection prompt with few-shot examples."""
        prompt = f"{self.facts_from_facts_examples}\n"
        clean_question = remove_problem_prefix(question)
        prompt += f"Task:\n{clean_question}\n\n"
        prompt += f"Facts:\n{facts}\n\n"
        
        if current_selection:
            prompt += f"Selected facts:\n{current_selection}"
        else:
            prompt += "Selected facts:"

        return prompt
        
    def select_simple(self, facts_list: List[str], question: str) -> str:
        """
        Simple selection using direct generation.
        
        Args:
            facts: Current context with facts and rules
            question: Question to answer
            
        Returns:
            Selected facts string
        """
        facts = "\n".join(facts_list)
        prompt = self._build_selection_prompt(facts, question)
        
        system_prompt = (
            "You are an AI assistant specialized in selecting the most relevant facts to solve a given task. " 
            "You will receive a task description and a list of facts. " 
            "Your job is to identify and return facts that are the most useful for solving the task at this stage. " 
            "Useful facts are those that provide critical information, directly address the task's requirements, " 
            "or significantly reduce the solution complexity. " 
            "Do not select facts that are redundant, trivial, or irrelevant. " 
            "Return the selected facts, each on a new line, exactly as they appear in the input list. " 
            "Do not include any additional commentary, explanations, or formatting."""
        )
    
        selection = self.generate(prompt, system_prompt)
        return f"We know that:\n{selection}"
    
    def _score_completion(self, prompt: str) -> float:
        """Score completion using LLM."""
        response = self.client.completions.create(
            model=self.model_name,
            prompt=prompt,
            max_tokens=0,
            temperature=0,
            logprobs=True,
            echo=True
        )
        
        logprobs_list = response.choices[0].logprobs.token_logprobs
        total_logprob = sum(lp for lp in logprobs_list if lp is not None)
        
        return total_logprob
    
    def select_scoring(self, facts_list: List[str], question: str, max_facts: int = 2) -> str:
        """
        Scoring-based selection using log-likelihood scoring.
        
        Args:
            facts: Current context with facts and rules  
            question: Question to answer
            max_facts: Maximum number of facts to select
            
        Returns:
            Selected facts string
        """
        facts = "\n".join(facts_list)
        selected_facts = []
        
        for _ in range(max_facts):
            best_fact = None
            best_score = float('-inf')
            
            for fact in facts_list:
                if fact in selected_facts:
                    continue
                    
                current_selection = "\n".join(selected_facts + [fact])
                prompt = self._build_selection_prompt(facts, question, current_selection)
                score = self._score_completion(prompt)
                
                if score > best_score:
                    best_score = score
                    best_fact = fact
                    
            if best_fact:
                selected_facts.append(best_fact)
        
        return "We know that:\n" + "\n".join(selected_facts)


class Inference:
    """Inference module for generating new facts from selections."""
    
    def __init__(
        self, 
        examples: str,
        model_url: str = "http://localhost:7215/v1",
        model_name: str = "Qwen/Qwen1.5-32B-Chat-AWQ",
        temperature: float = 0.
    ):
        self.examples = self.load_examples(examples)
        self.model_name = model_name
        self.temperature = temperature
        
        self.client = OpenAI(
            base_url=model_url,
            api_key="vllm"
        )
        
    def _format_inference_examples(self, examples_list: List[Dict[str, str]]) -> str:
        """Format inference examples from structured data."""
        formatted = ""
        for i, example in enumerate(examples_list):
            formatted += f"Example {i + 1}:\n\n"
            formatted += f"{example['selection']}\nTherefore,\n{example['inference']}\n\n"
        
        return formatted
        
    def load_examples(
        self, 
        examples: Optional[str] = None
    ) -> str:
        """
        Load examples.
        
        Args:
            examples: Examples as string or path to file
            
        Returns:
            Formatted examples string
        """
        if examples is not None and examples.endswith('.jsonl'):
            examples_list = []
            with open(examples, 'r', encoding='utf-8') as f:
                for line in f:
                    if line.strip():
                        examples_list.append(json.loads(line))

            return self._format_inference_examples(examples_list)
        else:
            return []
        
    def _build_inference_prompt(self, selection: str) -> str:
        """Build inference prompt with few-shot examples."""
        prompt = f"{self.examples}"
        prompt += f"{selection} Therefore,"
        return prompt
    
    def generate(self, prompt: str, system_prompt: str) -> str:
        response = self.client.chat.completions.create(
            model=self.model_name,
           messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt}
            ],
            temperature=self.temperature,
        )
        
        return response.choices[0].message.content
        
    def infer(self, selection: str) -> str:
        """
        Generate inference from selection.
        
        Args:
            selection: Selected facts string
            
        Returns:
            Newly inferred fact
        """
        prompt = self._build_inference_prompt(selection)
        
        system_prompt = (
            "You are an AI assistant specialized in logical deduction and step-by-step problem-solving. " 
            "You will receive a set of facts related to a programming task. " 
            "Your task is to determine whether the current facts are sufficient to produce a complete solution. " 
            "If they are, output exactly:\n\n"
            "The problem is solved:\n"
            "[the solution in the format required by the problem, e.g., a code block]\n\n"
            "If the facts are not yet sufficient, derive a single new fact that logically follows from the given facts. " 
            "This new fact must be a concise, standalone statement that advances the reasoning toward the solution. " 
            "The new fact may include code snippets if relevant (as shown in the examples). " 
            "Output only that fact, with no additional text, commentary, or formatting."
        )

        inference = self.generate(prompt, system_prompt)
        return inference


class SelectionInference(Agent):
    def __init__(
        self, 
        facts_from_context_examples: Optional[str] = None,
        facts_from_facts_examples: Optional[str] = None,
        inference_examples: Optional[str] = None,
        name: str = "SelectionInference",
        selection_mode: str = "score",
        max_steps: int = 5,
        temperature: float = 0.,
        max_facts: int = 3
    ):
        """
        Args:
            selection_examples: Few-shot examples for selection module
            inference_examples: Few-shot examples for inference module  
            name: Name of the reasoning module
            selection_mode: 'simple' or 'scoring' selection mode
            max_steps: Maximum number of reasoning steps
        """
        super().__init__(name)
        self.selection_mode = selection_mode
        self.max_steps = max_steps
        self.max_facts = max_facts
        
        self.selection_module = Selection(facts_from_context_examples, facts_from_facts_examples, temperature=temperature)
        self.inference_module = Inference(inference_examples, temperature=temperature)
    
    
    def extract_solution(self, response: str) -> str:
        marker = "The problem is solved:\n"
        idx = response.find(marker)
        if idx == -1:
            return "Solution not found"

        return response[idx + len(marker):]
    
    def _get_facts_list(self, context: str) -> List[str]:
        """Extract individual facts from context string."""
        sentences = re.split(r'\n', context)
        return [s.strip() for s in sentences if s.strip()]
    
    def run(self, question: str) -> str:
        facts = self.selection_module.extract_facts_from_context(question)
        facts_list = self._get_facts_list(facts)
        
        print(f"{facts}\n\n")
        
        for step in range(self.max_steps):
            if self.selection_mode == "score":
                selection = self.selection_module.select_scoring(facts_list, question, max_facts=self.max_facts)
            else:
                selection = self.selection_module.select_simple(facts_list, question)
            inference = self.inference_module.infer(selection)
            
            print(f"Step {step + 1}:\n\n")
            print(f"Selection:\n{selection}\n\n")
            print(f"Inference:\n{inference}\n\n")
            
            if inference.startswith("The problem is solved:\n"):
                return self.extract_solution(inference)
            
            facts_list.append(inference)
    
        return inference