from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import (
    Text,
    List,
    Optional,
    Tuple,
)
import random

from context_engine import ContextAssembler
# from core import Text
from db import IDB

# ---------- Agent primitives: Planner, Critic, Student ----------
class Agent(ABC):
    """Base class for all agents in the system."""

    def __init__(self, name: str):
        self.name = name
    
    @abstractmethod
    def run(self, *args, **kwargs):
        raise NotImplementedError
    
    def __str__(self):
        return f"{self.__class__.__name__}({self.name})"


class Planner(Agent):
    """Generate a plan (list of steps) given a task and context."""

    def __init__(self, name: str = "Planner"):
        super().__init__(name)
    
    @abstractmethod
    def run(self, task: Text, context: Text) -> List[Text]:
        raise NotImplementedError


class Critic(Agent):
    """Evaluate an intermediate/output and return feedback or an accept signal."""

    def __init__(self, name: str = "Critic"):
        super().__init__(name)
    
    @abstractmethod
    def run(self, candidate: Text, context: Text) -> Tuple[bool, Optional[Text]]:
        raise NotImplementedError


class Student(Agent):
    """Produce content (answer, code, explanation) given plan + context + feedback.

    The Student is the LLM-backed worker in most setups.
    """

    def __init__(self, name: str = "Student"):
        super().__init__(name)
    
    @abstractmethod
    def run(self, prompt: Text, context: Text) -> Text:
        raise NotImplementedError
        
# -------- Agents for MARS pipeline --------------
    
class UserProxy(Agent):
    def __init__(self, name: str = "UserProxy"):
        super().__init__(name)
    
    @abstractmethod
    def run(self, prompt: Text, context: Text) -> Text:
        raise NotImplementedError
    
    
class Teacher(Agent):
    def __init__(self, name: str = "Teacher"):
        super().__init__(name)
    
    @abstractmethod
    def run(self, prompt: Text, context: Text) -> Text:
        raise NotImplementedError
    
    
class Target(Agent):
    def __init__(self, name: str = "Target"):
        super().__init__(name)
    
    @abstractmethod
    def run(self, prompt: Text, context: Text) -> float:
        raise NotImplementedError
    
    @abstractmethod
    def check_stop_condition(self) -> bool:
        raise NotImplementedError


# ---------- Orchestration: Pipeline & Workflows ----------

@dataclass
class PipelineConfig:
    planner: Planner
    critic: Critic
    student: Student
    max_iterations: int = 3
    # loop_until_accepted: bool = True
    

class AgentPipeline1:
    """Orchestrates planner -> student -> critic cycles according to config."""

    def __init__(self, db: IDB, context_assembler: ContextAssembler, cfg: PipelineConfig):
        self.db = db
        self.context_assembler = context_assembler
        self.cfg = cfg

    def run(self, task: Text) -> Text:
        # 1) retrieve context
        retrieved = self.db.query(task, top_k=10)
        context = self.context_assembler.assemble(task, retrieved)

        last_output: Optional[Text] = None
        for iteration in range(self.cfg.max_iterations):
            # 2) planning
            steps = self.cfg.planner.run(task, context)
            # either step-by-step or full plan
            if len(steps) > 1:
                plan_prompt = "\n".join(steps)
            else:
                plan_prompt = steps[0]

            # student produces
            output = self.cfg.student.run(plan_prompt, context)

            # critic evaluates
            accepted, feedback = self.cfg.critic.run(output, context)

            if accepted:
                return output

            # if not accepted, incorporate feedback into next iteration
            # simple strategy: append feedback to context and try again
            if feedback:
                context = context + "\n\nCRITIC_FEEDBACK:\n" + feedback
            
            last_output = output

            # if not self.cfg.loop_until_accepted:
            #     break

        # return last produced output if nothing accepted
        return last_output or ""


class AgentPipeline2:
    """Orchestrates planner -> student -> critic cycles according to config."""

    def __init__(self, db: IDB, context_assembler: ContextAssembler, cfg: PipelineConfig):
        self.db = db
        self.context_assembler = context_assembler
        self.cfg = cfg

    def run(self, task: Text) -> Text:
        # 1) retrieve context
        retrieved = self.db.query(task, top_k=10)
        context = self.context_assembler.assemble(task, retrieved)

        output: Optional[Text] = None
        iteration = 0

        while True:
            # 2) planning
            steps = self.cfg.planner.run(task, context)
            # either step-by-step or full plan
            if len(steps) > 1:
                plan_prompt = "\n".join(steps)
            else:
                plan_prompt = steps[0]

            # student produces
            output = self.cfg.student.run(plan_prompt, context)

            if iteration == self.cfg.max_iterations:
                return output or ""

            # critic evaluates
            accepted, feedback = self.cfg.critic.run(output, context)

            if accepted:
                return output

            # if not accepted, incorporate feedback into next iteration
            # simple strategy: append feedback to context and try again
            if feedback:
                context = context + "\n\nCRITIC_FEEDBACK:\n" + feedback
            
            iteration += 1 
            
            
class AgentPipeline3:
    """Orchestrates planner -> student -> critic cycles according to config."""

    def __init__(self, db: IDB, context_assembler: ContextAssembler, cfg: PipelineConfig):
        self.db = db
        self.context_assembler = context_assembler
        self.cfg = cfg

    def run(self, task: Text) -> Text:
        # 1) retrieve context
        retrieved = self.db.query(task, top_k=10)
        context = self.context_assembler.assemble(task, retrieved)

        last_output: Optional[Text] = None
        for iteration in range(self.cfg.max_iterations):
            # 2) planning
            plan_step_prompt = self.cfg.planner.run(task, context)
            
            # student produces
            output = self.cfg.student.run(plan_step_prompt, context)

            # critic evaluates
            accepted, feedback = self.cfg.critic.run(output, context)

            if accepted:
                return output

            # if not accepted, incorporate feedback into next iteration
            # simple strategy: append feedback to context and try again
            if feedback:
                context = context + "\n\nCRITIC_FEEDBACK:\n" + feedback
            
            last_output = output

        # return last produced output if nothing accepted
        return last_output or ""


# ---------- MARS config and pipeline ----------------

@dataclass
class MARSConfig:
    user_proxy: UserProxy
    planner: Planner
    critic: Critic
    teacher: Teacher
    student: Student
    target: Target
    
    
class Manager(Agent):
    def __init__(self, db: IDB, context_assembler: ContextAssembler, cfg: MARSConfig, name: str = "Manager"):
        super().__init__(name)
        self.db = db
        self.context_assembler = context_assembler
        self.cfg = cfg
    
    def run(self, input: str):
        retrieved = self.db.query(task, top_k=10)
        context = self.context_assembler.assemble(task, retrieved)

        task = self.cfg.user_proxy.run(input, context)
        steps = self.cfg.planner.run(task, context)

        teacher_input = f"Task definition:\n{task}\n"
        teacher_init = self.cfg.teacher.run(teacher_input, context)

        critic_init = self.cfg.critic.run(teacher_init, context)

        student_input = f"Make the prompt better:\n{task}\n"
        student_response = self.cfg.student.run(student_input, context)

        target_response = self.cfg.target.run(student_response, context)

        while True:
            if self.cfg.target.check_stop_condition():
                print("Stop condition met!")
                break
            
            for i, step in enumerate(steps):
                teacher_input = f"\n{task}\n{student_response}\n{step}\n"
                teacher_response = self.cfg.teacher.run(teacher_input, context)

                accepted, feedback = self.cfg.critic.run(teacher_response, context)

                if accepted == False:
                    if feedback:
                        teacher_input = f"\n{feedback}\n{task}\n{student_response}\n{step}\n"
                    teacher_response = self.cfg.teacher.run(teacher_input, context)

                student_input = f"\n{task}\n{student_response}\n{teacher_response}\n"
                student_response = self.cfg.student.run(student_input, context)

            target_response = self.cfg.target.run(student_response, context)

        print("The end!")


# ---------- Example planner / critic / student (very small stubs) ----------
class SimplePlanner(Planner):
    def plan(self, task: Text, context: Text) -> List[Text]:
        # naive: split task by sentences as steps
        return [s.strip() for s in task.split(".") if s.strip()]
    
    
class SimplePlannerStepByStep(Planner):
    def __init__(self):
        super().__init__()
        self._call_count = 0
        self._cached_sentences = []
    
    def run(self, task: Text, context: Text) -> Text:
        if not self._cached_sentences:
            self._cached_sentences = [s.strip() for s in task.split(".") if s.strip()]
        
        if self._call_count >= len(self._cached_sentences):
            return ""
        
        result = self._cached_sentences[self._call_count]
        self._call_count += 1
        
        return result


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
    
    
class SimpleUserProxy(UserProxy):
    def run(self, prompt: Text, context: Text) -> Text:
        return prompt
    
    
class SimpleTeacher(Teacher):
    def run(self, prompt: Text, context: Text) -> Text:
        return prompt
    
    
class SimpleTarget(Target):
    def __init__(self):
        super().__init__()
        self.call_cnt = 0
    
    def run(self, prompt: Text, context: Text) -> float:
        accuracy = random.random()
        self.call_cnt += 1
        return accuracy

    def check_stop_condition(self) -> bool:
        if self.call_cnt >= 5:
            return True
        return False
