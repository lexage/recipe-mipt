from typing import List, Dict, Any
import json
import re
from src.agent_constructor.agent import Agent


class Selection:
    """Selection module for selecting relevant facts from context."""
    
    def __init__(self, examples: str, mode: str = "simple"):
        """
        Args:
            examples: Few-shot examples for selection
            mode: 'simple' for direct generation or 'scoring' for scoring-based selection
        """
        self.examples = examples
        self.mode = mode
        
    def select_simple(self, context: str, question: str) -> str:
        """
        Simple selection using direct generation.
        
        Args:
            context: Current context with facts and rules
            question: Question to answer
            
        Returns:
            Selected facts string
        """
        prompt = self._build_selection_prompt(context, question)
        return self.llm(prompt)
    
    def select_scoring(self, context: str, question: str, max_facts: int = 2) -> str:
        """
        Scoring-based selection using log-likelihood scoring.
        
        Args:
            context: Current context with facts and rules  
            question: Question to answer
            max_facts: Maximum number of facts to select
            
        Returns:
            Selected facts string
        """
        facts = self._extract_facts_from_context(context)
        selected_facts = []
        
        for _ in range(max_facts):
            best_fact = None
            best_score = float('-inf')
            
            for fact in facts:
                if fact in selected_facts:
                    continue
                    
                current_selection = "We know that " + " and ".join(selected_facts + [fact])
                prompt = self._build_selection_prompt(context, question, current_selection)
                score = self._score_completion(prompt)
                
                if score > best_score:
                    best_score = score
                    best_fact = fact
            
            if best_fact:
                selected_facts.append(best_fact)
        
        return "We know that " + " and ".join(selected_facts)
    
    def _build_selection_prompt(self, context: str, question: str, current_selection: str = "") -> str:
        """Build selection prompt with few-shot examples."""
        prompt = f"{self.examples}\n"
        prompt += f"Context: {context}\n"
        prompt += f"Question: {question}\n"
        
        if current_selection:
            prompt += f"Reason: {current_selection}"
        else:
            prompt += "Reason:"

        return prompt
    
    def _extract_facts_from_context(self, context: str) -> List[str]:
        """Extract individual facts from context string."""
        # Simple extraction - split by sentences
        sentences = re.split(r'[.!?]+', context)
        return [s.strip() for s in sentences if s.strip()]
    
    def _score_completion(self, prompt: str) -> float:
        """Score completion using LLM."""
        # TODO: In real implementation, this would use LLM's log probability scoring
        return 0.0
    
    def llm(self, prompt: str) -> str:
        # TODO: Replace with actual LLM call
        return ""


class Inference:
    """Inference module for generating new facts from selections."""
    
    def __init__(self, examples: str):
        self.examples = examples
        
    def infer(self, selection: str) -> str:
        """
        Generate inference from selection.
        
        Args:
            selection: Selected facts string
            
        Returns:
            Newly inferred fact
        """
        prompt = self._build_inference_prompt(selection)
        completion = self.llm(prompt)
        return self._extract_first_sentence(completion)
    
    def _build_inference_prompt(self, selection: str) -> str:
        """Build inference prompt with few-shot examples."""
        prompt = f"{self.examples}\n\n"
        prompt += f"{selection}. Therefore,"
        return prompt
    
    def _extract_first_sentence(self, text: str) -> str:
        """Extract first sentence from generated text."""
        match = re.search(r'^[^.!?]*[.!?]', text)
        return match.group(0) if match else text.split('.')[0] + '.'
    
    def llm(self, prompt: str) -> str:
        # TODO: Replace with actual LLM call
        return ""


class SelectionInference(Agent):
    def __init__(
        self, 
        selection_examples: str,
        inference_examples: str,
        name: str = "SelectionInference",
        selection_mode: str = "simple",
        max_steps: int = 5
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
        self.selection_examples = self.load_examples(selection_examples)
        self.inference_examples = self.load_examples(inference_examples)
        self.selection_mode = selection_mode
        self.max_steps = max_steps
        
        self.selection_module = Selection(self.selection_examples, self.selection_mode)
        self.inference_module = Inference(self.inference_examples)
        
    def load_examples(self, examples: str, module: str = "selection") -> str:
        """
        Load examples.
        
        Args:
            examples: Examples as string or path to file
            
        Returns:
            Formatted examples string
        """
        if examples.endswith('.json'):
            with open(examples, 'r') as f:
                data = json.load(f)
                if module == "selection":
                    return self._format_selection_examples(data)
                else:
                    return self._format_inference_examples(data)
        else:
            return examples
    
    def _format_selection_examples(self, examples_data: Dict[str, Any]) -> str:
        """Format selection examples from structured data."""
        formatted = ""
        for example in examples_data.get("examples", []):
            formatted += f"Context: {example['context']}\n"
            formatted += f"Question: {example['question']}\n"
            formatted += f"Reason: {example['selection']}\n\n"
        return formatted
    
    def _format_inference_examples(self, examples_data: Dict[str, Any]) -> str:
        """Format inference examples from structured data."""
        formatted = ""
        for example in examples_data.get("examples", []):
            formatted += f"{example['selection']}. Therefore, {example['inference']}\n\n"
        return formatted
    
    def run(self, context: str, question: str) -> str:
        """
        Execute Selection-Inference reasoning.
        
        Args:
            context: Initial context with facts and rules
            question: Question to answer
            
        Returns:
            Answer to question
        """
        current_context = context
        final_inference = ""
        
        for _ in range(self.max_steps):
            if self.selection_mode == "simple":
                selection = self.selection_module.select_simple(current_context, question)
            else:
                selection = self.selection_module.select_scoring(current_context, question)
            
            inference = self.inference_module.infer(selection)
            current_context += f" {inference}"
            final_inference = inference
            
        return final_inference