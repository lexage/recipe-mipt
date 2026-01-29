import json
import sys
from pathlib import Path
from typing import List, Optional

import openai
from pydantic import BaseModel, Field, ValidationError

project_root = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(project_root))

from src.agent_constructor.agent import Agent
from src.agent_constructor.core import Chunk
from src.filtering.textbooks_are_all_you_need.src.prompts import (
    label_prompt,
    system_prompt,
)


class EvaluationResult(BaseModel):
    reasoning: str = Field(description="Justification of the assessment")
    score: int = Field(ge=0, le=1, description="Score (0 or 1)")


class EducationalEvaluator(Agent):
    """Evaluator for assessing educational content value using LLM."""

    def __init__(
        self,
        api_url: str,
        model_name: str,
        limit_per_class: int = 10,
        api_key: str = "vllm",
    ):
        """
        Initialize the educational content evaluator.

        Args:
            api_url: Base URL for the OpenAI-compatible API endpoint
            model_name: Name of the model to use for evaluation
            limit_per_class: Maximum number of annotated samples to consider during training for each class.
            api_key: API key for authentication (default: "vllm")
        """
        super().__init__("educational_evaluate_agent")
        self.client = openai.OpenAI(base_url=api_url, api_key=api_key)
        self.model_name = model_name
        self.system_prompt = system_prompt
        self.label_template = label_prompt
        self.limit_per_class = limit_per_class

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
        Evaluate the educational value of the provided text content using an LLM.

        Args:
            text (str): The textual content to be evaluated for educational value.
            max_retries (int, optional): Maximum number of retry attempts in case of 
                transient failures (e.g., network issues or LLM timeouts). Defaults to 3.

        Returns:
            Optional[EvaluationResult]: An object containing the evaluation result 
                (typically a binary score such as 0 or 1 indicating low or high educational value),
                or None if evaluation fails after all retries.
        """

        for attempt in range(max_retries):
            try:
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

    def run(self, subsample_chunks: List[Chunk]) -> List[int]:
        """Generate educational-value annotations for a list of chunks.

        This method iterates over a list of chunks, evaluates each one using
        `evaluate_content`, and collects the resulting binary labels as integers.
        Stops when minimum threshold of 10 examples for each class (0 and 1) is reached.

        Args:
            subsample_chunks: A list of Chunk objects whose texts will be evaluated
                                for educational value.

        Returns:
            A list of integers (0 or 1) with length equal to the original chunks list.
            Chunks after reaching the threshold will have None values.
        """
        annotations = [None] * len(subsample_chunks)
        count_0 = 0
        count_1 = 0

        for i, doc in enumerate(subsample_chunks):
            result = self.evaluate_content(doc.text)
            if result is not None:
                score = result.score
                annotations[i] = score

                if score == 0:
                    count_0 += 1
                else:
                    count_1 += 1

                if count_0 >= self.limit_per_class and count_1 >= self.limit_per_class:
                    break

        return annotations
