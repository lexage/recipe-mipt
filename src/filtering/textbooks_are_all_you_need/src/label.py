import sys
import openai
from typing import List
from pathlib import Path

project_root = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(project_root))

from src.agent_constructor.core import Document
from src.filtering.textbooks_are_all_you_need.src.prompts import system_prompt, label_prompt


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
        
        print("formatted_prompt", formatted_prompt)
        
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=[
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": formatted_prompt}
            ],
            temperature=0,
        )
        
        return response.choices[0].message.content
    
    def get_annotations(self, subsample_documents: List[Document]) -> List[int]:
        """Generate educational-value annotations for a list of documents.
        
        This method iterates over a list of documents, evaluates each one using 
        `evaluate_content`, and collects the resulting binary labels as integers.
        
        Args:
            subsample_documents: A list of Document objects whose texts will be evaluated 
                                for educational value.
                                
        Returns:
            A list of integers (0 or 1), where each integer represents the educational 
            annotation for the corresponding document in the input list.
            
        Note:
            The method assumes that the output of `evaluate_content` is a string 
            containing either "0" or "1", which it converts to int.
        """
        annotations = []
        for doc in subsample_documents:
            label = self.evaluate_content(doc.text)
            annotations.append(label)
        
        print("annotations ", annotations)
            
        return annotations
            
