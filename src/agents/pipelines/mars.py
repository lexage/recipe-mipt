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

import os
import sys
sys.path.insert(0, os.getcwd())

from src.agent_constructor.agent import Agent

from context_engine import ContextAssembler
# from core import Text
from db import IDB

# ---------- Agent primitives: Planner, Critic, Student ----------

class Planner(Agent):
    """Generate a plan (list of steps) given a task and context."""

    def __init__(self, name: str = "Planner"):
        super().__init__(name)
    
    @abstractmethod
    def run(self, task: Text, context: Text = None) -> List[Text]:
        raise NotImplementedError


class Critic(Agent):
    """Evaluate an intermediate/output and return feedback or an accept signal."""

    def __init__(self, name: str = "Critic"):
        super().__init__(name)
    
    @abstractmethod
    def run(self, candidate: Text, context: Text = None) -> Tuple[bool, Optional[Text]]:
        raise NotImplementedError


class Student(Agent):
    """Produce content (answer, code, explanation) given plan + context + feedback.

    The Student is the LLM-backed worker in most setups.
    """

    def __init__(self, name: str = "Student"):
        super().__init__(name)
    
    @abstractmethod
    def run(self, prompt: Text, context: Text = None) -> Text:
        raise NotImplementedError

            
class UserProxy(Agent):
    def __init__(self, name: str = "UserProxy"):
        super().__init__(name)
    
    @abstractmethod
    def run(self, prompt: Text, context: Text = None) -> Text:
        raise NotImplementedError
    
    
class Teacher(Agent):
    def __init__(self, name: str = "Teacher"):
        super().__init__(name)
    
    @abstractmethod
    def run(self, prompt: Text, context: Text = None) -> Text:
        raise NotImplementedError
    
    
class Target(Agent):
    def __init__(self, name: str = "Target"):
        super().__init__(name)
    
    @abstractmethod
    def run(self, prompt: Text, context: Text = None) -> float:
        raise NotImplementedError
    
    @abstractmethod
    def check_stop_condition(self) -> bool:
        raise NotImplementedError


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
        steps = self.cfg.planner.run(task)

        teacher_input = f"Task definition:\n{task}\n"
        teacher_init = self.cfg.teacher.run(teacher_input)

        critic_init = self.cfg.critic.run(teacher_init)

        student_input = f"Make the prompt better:\n{task}\n"
        student_response = self.cfg.student.run(student_input)

        target_response = self.cfg.target.run(student_response)

        while True:
            if self.cfg.target.check_stop_condition():
                print("Stop condition met!")
                break
            
            for _, step in enumerate(steps):
                teacher_input = f"\n{task}\n{student_response}\n{step}\n"
                teacher_response = self.cfg.teacher.run(teacher_input)

                accepted, feedback = self.cfg.critic.run(teacher_response)

                if accepted == False:
                    if feedback:
                        teacher_input = f"\n{feedback}\n{task}\n{student_response}\n{step}\n"
                    teacher_response = self.cfg.teacher.run(teacher_input)

                student_input = f"\n{task}\n{student_response}\n{teacher_response}\n"
                student_response = self.cfg.student.run(student_input)

            target_response = self.cfg.target.run(student_response)
        
        print("The end!")
        return target_response


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


class SimpleStudent(Student):
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