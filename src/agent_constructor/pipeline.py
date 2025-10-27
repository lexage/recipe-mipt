from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import (
    List,
    Optional,
    Tuple,
)

from src.agents.agent_constructor.context_engine import ContextAssembler
from src.agents.agent_constructor.core import Text
from src.agents.agent_constructor.db import IDB


class Agent(ABC):
    """Base class for all agents in the system."""

    def __init__(self, name: str):
        self.name = name

    @abstractmethod
    def run(self, *args, **kwargs):
        raise NotImplementedError

    def __str__(self):
        return f"{self.__class__.__name__}({self.name})"

# ---------- Agent primitives: Planner, Critic, Student ----------
class Planner(ABC):
    """Generate a plan (list of steps) given a task and context."""

    @abstractmethod
    def plan(self, task: Text, context: Text) -> List[Text]:
        raise NotImplementedError


class Critic(ABC):
    """Evaluate an intermediate/output and return feedback or an accept signal."""

    @abstractmethod
    def critique(self, candidate: Text, context: Text) -> Tuple[bool, Optional[Text]]:
        """Return (accepted, feedback). If accepted True, feedback can be None."""
        raise NotImplementedError


class Student(ABC):
    """Produce content (answer, code, explanation) given plan + context + feedback.

    The Student is the LLM-backed worker in most setups.
    """

    @abstractmethod
    def execute(self, prompt: Text, context: Text) -> Text:
        raise NotImplementedError


# ---------- Orchestration: Pipeline & Workflows ----------

@dataclass
class PipelineConfig:
    planner: Planner
    critic: Critic
    student: Student
    max_iterations: int = 3
    loop_until_accepted: bool = True


class AgentPipeline:
    """Orchestrates planner -> student -> critic cycles according to config."""

    def __init__(self, db: IDB, context_assembler: ContextAssembler, cfg: PipelineConfig):
        self.db = db
        self.context_assembler = context_assembler
        self.cfg = cfg

    def run(self, task: Text) -> Text:
        # 1) retrieve context
        retrieved = self.db.query(task, top_k=10)
        context = self.context_assembler.assemble(task, retrieved)

        # 2) planning
        steps = self.cfg.planner.plan(task, context)

        last_output: Optional[Text] = None
        for iteration in range(self.cfg.max_iterations):
            # either step-by-step or full plan
            if len(steps) > 1:
                plan_prompt = "\n".join(steps)
            else:
                plan_prompt = steps[0]

            # student produces
            output = self.cfg.student.execute(plan_prompt, context)

            # critic evaluates
            accepted, feedback = self.cfg.critic.critique(output, context)

            if accepted:
                return output

            # if not accepted, incorporate feedback into next iteration
            # simple strategy: append feedback to context and try again
            if feedback:
                context = context + "\n\nCRITIC_FEEDBACK:\n" + feedback

            last_output = output

            if not self.cfg.loop_until_accepted:
                break

        # return last produced output if nothing accepted
        return last_output or ""


# ---------- Example planner / critic / student (very small stubs) ----------
class SimplePlanner(Planner):
    def plan(self, task: Text, context: Text) -> List[Text]:
        # naive: split task by sentences as steps
        return [s.strip() for s in task.split(".") if s.strip()]


class SimpleCritic(Critic):
    def critique(self, candidate: Text, context: Text) -> Tuple[bool, Optional[Text]]:
        # accept if candidate is non-empty and longer than 10 chars
        if candidate and len(candidate) > 10:
            return True, None
        return False, "output too short"


class MockStudent(Student):
    def execute(self, prompt: Text, context: Text) -> Text:
        # In real system this wraps LLM calls. Here we echo prompt + short summary.
        return f"EXECUTION_RESULT:\nPrompt:\n{prompt}\n---\nContext-snippet: {context[:200]}"