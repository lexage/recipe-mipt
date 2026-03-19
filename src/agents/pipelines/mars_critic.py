import json
import re
import logging

from typing import List, Tuple
from openai import OpenAI

from src.agent_constructor.agent import Agent
from src.agent_constructor.core import Text


SYSTEM_PLANNER_PROMPT = """You are a task planning assistant. Your response must follow this exact template:

Total steps: [number]
Step 1: [description]
Step 2: [description]
...
Step N: [description]

For example:
Total steps: 3
Step 1: Analyze the task requirements.
Step 2: Identify the resources needed.
Step 3: Execute the plan.

Please follow this format strictly."""

SYSTEM_TEACHER_PROMPT =  """You are a teacher who asks questions in the Socratic manner based on objectives and student responses. 
Please ask a total of two questions:
The first one is for the problem that appeared in the prompt given by the students in the last round.
The second one is an optimization solution based on the current steps of the task.

IMPORTANT: 
- If the current solution is already correct and simple, ask questions that confirm correctness rather than suggesting changes.
- Do not suggest adding verification code, print statements, or complex logic if the simple solution works.
- Focus on matching the exact task requirements, not on unnecessary optimizations.

Please include only questions in your output and do not make answers for your students."""

SYSTEM_CRITIC_PROMPT = """You are an evaluator responsible for judging the quality of teacher's heuristic questions. Your output must strictly follow these rules:

1. If the questions are judged as HIGH-QUALITY (relevant, constructive, help improve the answer), output only:
   [True]

2. If the questions are judged as LOW-QUALITY (irrelevant, redundant, misleading, or already addressed), output:
   [False]
   [suggestion: <reason why the questions are inadequate>]

   Replace `<reason why the questions are inadequate>` with a clear and concise explanation of why the questions fail to improve the student's answer.

Do not include any additional text, comments, or explanations beyond the specified format."""


SYSTEM_STUDENT_PROMPT = """You are the generator of the improved final answer, please continue to iterate the final answers as needed.

IMPORTANT RULES:
- Output ONLY the final solution code, no explanations, no print statements, no verification code.
- Do not add unnecessary complexity if the current solution is already correct.
- Match the exact output format required by the task.
- Keep the solution as simple and direct as possible.

Note that you should output only the final response you generated."""


class PlannerMARS(Agent):
    """Agent writes steps to improve the final response."""

    def __init__(
        self,
        url: str = None,
        model_name: str = None,
        temperature: float = 0.0,
        name: str = "mars_planner_agent",
    ):
        super().__init__(name)
        self.client = OpenAI(base_url=url, api_key="vllm")
        self.model_name = model_name
        self.temperature = temperature
        self.system_prompt = SYSTEM_PLANNER_PROMPT

    def llm(self, prompt: str) -> str:
        """Сalling the llm to get a response."""
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=[
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": prompt}
            ],
            temperature=self.temperature,
        )
        return response.choices[0].message.content

    def extract_steps(self, planner_response: str) -> Tuple[int, List[str]]:
        """Extracts steps from the llm's response"""
        total_steps_match = re.search(r"Total steps: (\d+)", planner_response)
        if not total_steps_match:
            raise ValueError("Planner response does not contain 'Total steps'")
        total_steps = int(total_steps_match.group(1))

        # Extract the description of each step
        steps = []
        for i in range(1, total_steps + 1):
            step_match = re.search(fr"Step {i}: (.+)", planner_response)
            if step_match:
                steps.append(step_match.group(1).strip())
            else:
                raise ValueError(f"Step {i} is missing in the planner response")

        return total_steps, steps

    def run(self, task: Text, final_answer: str) -> Text:

        planner_prompt = f"""Analyze the task: {task}.
And the provided initial solution snippet: 
{final_answer}.

Your goal is to create a plan to COMPLETE the code snippet. 
The initial code structure is mostly correct. Do NOT rewrite the entire solution from scratch.
Focus ONLY on:
1. Completing the specific missing line (e.g., the 'result = ...' assignment).
2. Ensuring variable names match exactly what is defined in the initial snippet (e.g., 'df', 'List', 'result').
3. Keeping the solution minimal - do not add file reading, functions, or extra logic unless explicitly required.

For example, for a coding task with an incomplete solution, the plan should be:
Total steps: 2
Step 1: Identify the exact variable assignment needed to complete the code (e.g., result = df.iloc[List]).
Step 2: Verify the completed code matches the required output format and variable names exactly.
"""

        response = self.llm(planner_prompt).strip()
        total_steps, steps = self.extract_steps(response)
        
        return steps
    

class TeacherMARS(Agent):
    """Agent asks questions based on the steps."""

    def __init__(
        self,
        url: str = None,
        model_name: str = None,
        temperature: float = 0.0,
        name: str = "mars_teacher_agent",
    ):
        super().__init__(name)
        self.client = OpenAI(base_url=url, api_key="vllm")
        self.model_name = model_name
        self.temperature = temperature
        self.system_prompt = SYSTEM_TEACHER_PROMPT

    def llm(self, prompt: str) -> str:
        """Сalling the llm to get a response."""        
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=[
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": prompt}
            ],
            temperature=self.temperature,
        )
        return response.choices[0].message.content

    def run(self, task: Text, step: str, final_answer: str, feedback: str = None, mode: str = "ask") -> Text:
        
        if mode == "regenerate":
            teacher_prompt = f"""Here is feedback on whether your output matches the Socratic questioning, please refer to the suggestion to regenerate the questioning:
{feedback}
Here is the task definition:
{task}
Here is the final answer given by the student from the previous round:
{final_answer}
Ask heuristic questions based on the students' historical responses and the current step: {step}
"""
        else:
            teacher_prompt = f"""Here is the task definition:
{task}
Here is the final answer given by the student from the previous round:
{final_answer}
Ask heuristic questions based on the students' historical responses and the current step: {step}
"""
        response = self.llm(teacher_prompt).strip()
        return response
        

class CriticMARSUpd(Agent):
    """Agent evaluates the relevance of the questions."""

    def __init__(
        self,
        url: str = None,
        model_name: str = None,
        temperature: float = 0.0,
        name: str = "mars_critic_agent_upd",
    ):
        super().__init__(name)
        self.client = OpenAI(base_url=url, api_key="vllm")
        self.model_name = model_name
        self.temperature = temperature
        self.system_prompt = SYSTEM_CRITIC_PROMPT

    def llm(self, prompt: str) -> str:
        """Сalling the llm to get a response."""
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=[
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": prompt}
            ],
            temperature=self.temperature,
        )
        return response.choices[0].message.content

    def run(self, task: Text, final_answer: str, teacher_message: str = None) -> Text:

        critic_prompt = f"""
### Task Definition
{task}

### Student's Final Answer
{final_answer}

### Teacher's Questions to Evaluate
{teacher_message}

### Evaluation Criteria for HIGH-QUALITY Questions:
1. Relevance: Questions address actual gaps/errors in the Student's Answer.
2. Constructiveness: Questions guide toward concrete improvements, not just criticism.
3. Non-redundancy: Questions are not already answered in the Student's Answer.
4. Specificity: Questions are precise, not vague or overly broad.

### Output Requirement
Strictly follow your system instructions format:
- If questions are high-quality: [True]
- If questions are low-quality: [False] [suggestion: <concise reason>]
"""
        response = self.llm(critic_prompt).strip()
        
        return response


class StudentMARS(Agent):
    """Agent generates an improved final response."""

    def __init__(
        self,
        url: str = None,
        model_name: str = None,
        temperature: float = 0.0,
        name: str = "mars_student_agent",
    ):
        super().__init__(name)
        self.client = OpenAI(base_url=url, api_key="vllm")
        self.model_name = model_name
        self.temperature = temperature
        self.system_prompt = SYSTEM_STUDENT_PROMPT

    def llm(self, prompt: str) -> str:
        """Сalling the llm to get a response."""
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=[
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": prompt}
            ],
            temperature=self.temperature,
        )
        return response.choices[0].message.content

    def run(self, task: Text, final_answer: str = None, questions: str = None, mode: str = "update") -> Text:

        if mode == "init":
            student_prompt = f"""Here is the task definition:
{task}
Please generate a more appropriate final answer based on the following prompt and task definition: Think step by step and solve the question."""
        else:
            student_prompt = f"""Here is the task definition:
{task}
Here is your last final answer:
{final_answer}
Please base on the following question update your final answer, write ONLY FINAL CODE:
{questions}"""
        
        response = self.llm(student_prompt).strip()
        return response
