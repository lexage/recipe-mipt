import openai
from typing import Optional, Dict, Any
from src.prompts import system_prompt, label_prompt


class EducationalEvaluator:
    """Evaluator for assessing educational content value using LLM."""
    
    def __init__(self, api_url: str, model_name: str, api_key: str = "vllm"):
        """
        Initialize the educational content evaluator.
        
        Args:
            api_url: Base URL for the OpenAI-compatible API endpoint
            model_name: Name of the model to use for evaluation
            api_key: API key for authentication (default: "vllm")
        """
        self.client = openai.OpenAI(
            base_url=api_url,
            api_key=api_key
        )
        self.model_name = model_name
        self.system_prompt = system_prompt
        self.label_template = label_prompt
        
    def _format_label_prompt(self, text: str) -> str:
        """
        Format the label prompt by replacing the example placeholder.
        
        Args:
            text: Text content to evaluate
            
        Returns:
            Formatted prompt with the text inserted
        """
        return self.label_template.replace("<example>", text)
        
    def evaluate_content(self, text: str) -> str:
        """
        Evaluate educational value of the provided text content.
        
        Args:
            text: Content to evaluate for educational value
            
        Returns:
            LLM response containing the evaluation result (0 or 1)
        """
        formatted_prompt = self._format_label_prompt(text)
        
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=[
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": formatted_prompt}
            ],
            temperature=0,
        )
        
        return response.choices[0].message.content
