"""Shared schema for Self-Refine few-shot examples.

The dataset-specific example INSTANCES live in the ``prompts_<dataset>`` modules
(``FEWSHOT_EXAMPLES``); this file keeps only the ``Example`` model so both datasets share
one schema. For multiple-choice (CodeMMLU) ``answer`` is the chosen letter + reasoning and
``refined`` is the corrected letter; for code-generation (DS-1000) they are implementations.
"""

from pydantic import BaseModel


class Example(BaseModel):
    question: str
    answer: str
    feedback: str
    refined: str
