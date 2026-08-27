"""ReflectionOracleSolver — ORACLE answer-side experiment: Qwen2.5 answers a
DS1000 task, the answer is graded by RUNNING THE REAL HIDDEN TEST (the DS1000
oracle), and on failure the model reflects on the actual execution error and
retries, for up to ``max_reflections`` cycles.

Oracle because grading between cycles calls the DS1000 test harness directly
(src/benchmarks/ds1000/execution.py — the same one run_ds1000.py uses for the
final score) — a real deployment has no such ground-truth oracle available at
inference time. This is an upper-bound diagnostic for self-reflection
(Reflexion-style, see src/agents/critique/reflexion/), not a deployable
method. Answer-side (ТЗ 2.3 PRC), independent of the corpus filtration /
generation work.

Slots in as a normal ``agent:`` component of SimplePipeline — it sees the
same RAG context as any other solver — but needs the ground-truth
``code_context`` per task to run the oracle test, which the pipeline's
``run(task: str)`` doesn't carry. So this agent loads the DS1000 dataset
itself (``dataset_path``) and looks each task's ``code_context`` up by
matching on the exact prompt text it receives — the same identity
run_ds1000.py already relies on (``pipeline.run(task.prompt)``).

Every cycle (attempt code, pass/fail, raw execution error, reflection text)
is recorded in-memory (``get_log()``); run_ds1000.py drains it into
``reflection_cycles.jsonl`` next to ``runtime_stats.json``, mirroring the
existing ``--log-chunks`` retrieval log.

NB: no ``from __future__ import annotations`` — keep annotations concrete for
the registry's dependency introspection.
"""

import logging
import threading
from typing import Dict, List, Optional

from openai import OpenAI
from transformers import AutoTokenizer, PreTrainedTokenizerFast

from src.agent_constructor.agent import Agent
from src.agent_constructor.core import Text
from src.benchmarks.ds1000 import DatasetDS1000
from src.benchmarks.ds1000 import execution

logger = logging.getLogger(__name__)

DEFAULT_SYSTEM_PROMPT = (
    "Write a short code following the given format and indentation. Place "
    "the executable code between <code> and </code> tags, without any other "
    "non-executable things."
)


def _extract_code(text: Text) -> Text:
    """Strip the model's completion down to bare code.

    Mirrors ``DS1000._postprocess`` (src/benchmarks/ds1000/core.py) so a
    cycle graded here and the final answer graded by run_ds1000.py agree.
    Duplicated rather than imported: that method is post-processing for the
    FINAL answer only, not a shared utility, and the logic is 5 stable lines.
    """
    code = (text or "").split("</code>")[0]
    code = code.replace("```python", "")
    code = code.split("```")[0]
    code = code.split("\nEND SOLUTION")[0]
    code = code.replace("<code>", "")
    return code.strip()


class ReflectionOracleSolver(Agent):
    """DS1000 solver with an oracle-graded self-reflection loop.

    Args:
        url: base URL of the OpenAI-compatible LLM endpoint.
        model_name: model id to call. If None, uses whatever model the
            endpoint has loaded (queried via ``client.models.list()``).
        dataset_path: path to the DS1000 ``.jsonl.gz`` — source of the
            per-task ``code_context`` used to run the oracle test.
        system_prompt: instruction shown for every code-writing turn.
        max_reflections: maximum number of reflect-and-retry cycles AFTER
            the first attempt (0 = single attempt, no reflection).
        temperature, top_p, max_tokens, stop_tokens: sampling params for
            the code-writing turns (first attempt + every retry).
        reflection_max_tokens: token budget for the reflection (diagnosis)
            turn — smaller since it's prose, not code.
        max_context_length: token budget for retrieved RAG context, mirrors
            DS1000Solver's own truncation so the two are comparable.
        exec_timeout: seconds allowed for each oracle test execution.
    """

    def __init__(
        self,
        url: str,
        model_name: Optional[str] = None,
        dataset_path: str = "data/ds1000/ds1000.jsonl.gz",
        system_prompt: str = DEFAULT_SYSTEM_PROMPT,
        max_reflections: int = 2,
        temperature: float = 0.2,
        top_p: float = 0.95,
        max_tokens: int = 1024,
        reflection_max_tokens: int = 512,
        stop_tokens: Optional[List[str]] = None,
        max_context_length: int = 24000,
        exec_timeout: float = 120.0,
        name: str = "reflection_oracle_solver",
    ):
        super().__init__(name)
        self.client = OpenAI(base_url=url, api_key="vllm")
        self.model_name = model_name or self.client.models.list().data[0].id
        self.system_prompt = system_prompt
        self.max_reflections = max(0, int(max_reflections))
        self.temperature = float(temperature)
        self.top_p = float(top_p)
        self.max_tokens = int(max_tokens)
        self.reflection_max_tokens = int(reflection_max_tokens)
        self.stop_tokens = stop_tokens if stop_tokens is not None else ["</code>", "# SOLUTION END"]
        self.max_context_length = int(max_context_length)
        self.exec_timeout = float(exec_timeout)

        self.tokenizer: PreTrainedTokenizerFast = AutoTokenizer.from_pretrained(self.model_name)

        dataset = DatasetDS1000(dataset_path)
        self._task_by_prompt: Dict[Text, object] = {item.prompt: item for item in dataset}
        logger.info(
            "ReflectionOracleSolver: indexed %d tasks from %s (max_reflections=%d)",
            len(self._task_by_prompt), dataset_path, self.max_reflections,
        )

        self._log: List[dict] = []
        self._log_lock = threading.Lock()

    # ------------------------------------------------------------- helpers
    def _truncate(self, text: Text, budget: int) -> Text:
        if budget <= 0:
            return ""
        tokens = self.tokenizer.encode(text, add_special_tokens=False)
        if len(tokens) <= budget:
            return text
        return self.tokenizer.decode(tokens[:budget], skip_special_tokens=True)

    def _chat(self, messages: List[dict], max_tokens: int, stop: Optional[List[str]]) -> Text:
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            temperature=self.temperature,
            top_p=self.top_p,
            max_tokens=max_tokens,
            stop=stop,
        )
        return response.choices[0].message.content or ""

    def _grade(self, item, code: Text) -> Dict:
        """Run the real DS1000 hidden test on ``code`` — the oracle signal."""
        test_program = (
            item.code_context + "\n"
            + f"code = {repr(code)}\n"
            + "test_execution(code)\n"
            + ("test_string(code)\n" if "test_string(" in item.code_context else "\n")
        )
        result, _ = execution.check_correctness(
            test_program, timeout=self.exec_timeout, metadata=item.metadata,
        )
        return result  # {"score": 0|1, "result": "passed" | "timed out" | "failed: ..."}

    # -------------------------------------------------------------- interface
    def run(self, task: Text, context: Text) -> Text:
        item = self._task_by_prompt.get(task)
        if item is None:
            logger.warning(
                "ReflectionOracleSolver: no DS1000 task matches this prompt "
                "(dataset/prompt mismatch) -- answering once, without oracle "
                "grading or reflection."
            )

        if context:
            task_tokens = len(self.tokenizer.encode(task, add_special_tokens=False))
            budget = self.max_context_length - task_tokens
            context = self._truncate(context, budget)
            user_prompt = f"{context}\n\n# Task ({self.system_prompt})\n\n{task}"
        else:
            user_prompt = task

        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": user_prompt},
        ]

        problem_id = item.metadata.get("problem_id") if item else None
        cycles: List[dict] = []
        final_code = ""

        for depth in range(self.max_reflections + 1):
            raw = self._chat(messages, self.max_tokens, self.stop_tokens)
            code = _extract_code(raw)
            final_code = code
            messages.append({"role": "assistant", "content": raw})

            if item is None:
                cycles.append({"cycle": depth, "code": code, "graded": False})
                break

            grade = self._grade(item, code)
            passed = grade["score"] == 1
            cycle_record = {
                "cycle": depth,
                "code": code,
                "passed": passed,
                "exec_result": grade["result"],
                "reflection": None,
            }
            cycles.append(cycle_record)
            logger.info(
                "REFLECT\t%s\tcycle=%d\tpassed=%s\tresult=%s",
                problem_id, depth, passed, grade["result"],
            )

            if passed or depth == self.max_reflections:
                break

            reflect_prompt = (
                "Your implementation FAILED the hidden test with this error:\n"
                f"{grade['result']}\n\n"
                "In 2-4 sentences, diagnose exactly what is wrong with your "
                "implementation above and what needs to change. Do NOT write "
                "code, only the diagnosis."
            )
            messages.append({"role": "user", "content": reflect_prompt})
            reflection = self._chat(messages, self.reflection_max_tokens, None)
            messages.append({"role": "assistant", "content": reflection})
            cycle_record["reflection"] = reflection

            retry_prompt = (
                "Now write a corrected solution to the ORIGINAL task, applying "
                f"your diagnosis. {self.system_prompt}"
            )
            messages.append({"role": "user", "content": retry_prompt})

        record = {
            "problem_id": problem_id,
            "prompt": task,
            "n_cycles": len(cycles),
            "passed": cycles[-1]["passed"] if item is not None and cycles else None,
            "cycles": cycles,
        }
        with self._log_lock:
            self._log.append(record)

        return final_code

    def get_log(self) -> List[dict]:
        """Return a snapshot of every task's cycle log recorded so far."""
        with self._log_lock:
            return list(self._log)
