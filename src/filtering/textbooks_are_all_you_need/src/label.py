import sys
import openai
import json
from typing import List, Optional, Tuple
from pathlib import Path
from pydantic import BaseModel, Field, ValidationError

project_root = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(project_root))

from src.agent_constructor.core import Document
from src.filtering.textbooks_are_all_you_need.src.prompts import (
    system_prompt,
    label_prompt,
)


class EvaluationResult(BaseModel):
    reasoning: str = Field(description="Justification of the assessment")
    score: int = Field(ge=0, le=1, description="Score (0 or 1)")


class EducationalEvaluator:
    """Evaluator for assessing educational content value using LLM."""

    def __init__(
        self, api_url: str, model_name: str, limit: int = 20, api_key: str = "vllm"
    ):
        """
        Initialize the educational content evaluator.

        Args:
            api_url: Base URL for the OpenAI-compatible API endpoint
            model_name: Name of the model to use for evaluation
            api_key: API key for authentication (default: "vllm")
        """
        self.client = openai.OpenAI(base_url=api_url, api_key=api_key)
        self.model_name = model_name
        self.system_prompt = system_prompt
        self.label_template = label_prompt
        self.limit = limit

    def _format_label_prompt(self, text: str) -> str:
        """
        Format the label prompt by replacing the example placeholder.

        Args:
            text: Text content to evaluate

        Returns:
            Formatted prompt with the text inserted
        """
        return self.label_template.replace("{text}", text)

    def evaluate_content(
        self, text: str, max_retries: int = 3
    ) -> Optional[EvaluationResult]:
        """
        Evaluate educational value of the provided text content.

        Args:
            text: Content to evaluate for educational value

        Returns:
            LLM response containing the evaluation result (0 or 1)
        """

        for attempt in range(max_retries):
            try:
                prompt = self._format_label_prompt(text)
                response = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=[
                        {"role": "system", "content": self.system_prompt},
                        {"role": "user", "content": self._format_label_prompt(text)},
                    ],
                    temperature=0,
                    response_format={"type": "json_object"},
                )
                content = response.choices[0].message.content

                result = json.loads(content)
                return EvaluationResult(**result)
            except (json.JSONDecodeError, ValidationError) as e:
                if attempt == max_retries - 1:
                    raise
                continue
        return None

    def get_annotations(self, subsample_documents: List[Document]) -> List[int]:
        """Generate educational-value annotations for a list of documents.

        This method iterates over a list of documents, evaluates each one using
        `evaluate_content`, and collects the resulting binary labels as integers.
        Stops when minimum threshold of 10 examples for each class (0 and 1) is reached.

        Args:
            subsample_documents: A list of Document objects whose texts will be evaluated
                                for educational value.

        Returns:
            A list of integers (0 or 1) with length equal to the original documents list.
            Documents after reaching the threshold will have None values.
        """
        annotations = [None] * len(subsample_documents)
        count_0 = 0
        count_1 = 0
        limit_per_class = self.limit // 2

        for i, doc in enumerate(subsample_documents):
            result = self.evaluate_content(doc.text)
            if result is not None:
                score = result.score
                annotations[i] = score

                if score == 0:
                    count_0 += 1
                else:
                    count_1 += 1

                if count_0 >= limit_per_class and count_1 >= limit_per_class:
                    break

        return annotations
