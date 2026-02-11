from typing import List
from pathlib import Path
import logging
import json
import re

from openai import OpenAI

from src.agent_constructor.agent import Agent
from src.agent_constructor.core import Text
from src.agent_constructor.context_engine import ContextAssembler
from src.agent_constructor.db import IDB


PLANNER_PROMPT = """For the following task, make plans that can solve the problem step by step. For each plan, indicate \
which external tool together with tool input to retrieve evidence. You can store the evidence into a \
variable #E that can be called by later tools. (Plan, #E1, Plan, #E2, Plan, ...)

Tools can be one of the following:
{tools}

For example,
Task: Thomas, Toby, and Rebecca worked a total of 157 hours in one week. Thomas worked x
hours. Toby worked 10 hours less than twice what Thomas worked, and Rebecca worked 8 hours
less than Toby. How many hours did Rebecca work?
Plan: Given Thomas worked x hours, translate the problem into algebraic expressions and solve
with Wolfram Alpha. #E1 = WolframAlpha[Solve x + (2x − 10) + ((2x − 10) − 8) = 157]
Plan: Find out the number of hours Thomas worked. #E2 = LLM[What is x, given #E1]
Plan: Calculate the number of hours Rebecca worked. #E3 = Calculator[(2 ∗ #E2 − 10) − 8]

Begin!
Describe your plans with rich details. Each Plan should be followed by only one #E.

Task: {task}"""


WORKER_PROMPT = """Respond in short directly with no extra words.\n\n{request}"""


SOLVER_PROMPT = """Solve the following task or problem. To solve the problem, we have made step-by-step Plan and \
retrieved corresponding Evidence to each Plan. Use them with caution since long evidence might \
contain irrelevant information.

{plan}

Now solve the question or task according to provided Evidence above. Respond with the answer
directly with no extra words.

Task: {task}
Response:"""


TOOLS_DESCRIPTION = """(1) retrieve_context[input]: Worker that retrieve relevant context or examples from the database. Useful when you need to find short
and succinct answers about a specific topic. The input should be a search query.
(2) LLM[input]: A pretrained LLM like yourself. Useful when you need to act with general
world knowledge and common sense. Prioritize it when you are confident in solving the problem
yourself. Input can be any instruction."""


class PlainFormatter(logging.Formatter):
    def format(self, record):
        return record.getMessage()


def setup_logger(log_file):
    logger = logging.getLogger("agent")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    logger.propagate = False
    Path(log_file).parent.mkdir(parents=True, exist_ok=True)
    file_handler = logging.FileHandler(log_file, encoding="utf-8")
    file_handler.setFormatter(PlainFormatter())
    logger.addHandler(file_handler)
    return logger


logger = setup_logger("grant/db/rewoo.log")


class PlannerREWOO(Agent):

    def __init__(
        self, 
        url: str = None,
        model_name: str = None,
        name: str = "rewoo_planner_agent",
        maximum_steps: int = 5
    ):
        super().__init__(name)
        self.client = OpenAI(base_url=url, api_key="vllm")
        self.model_name = model_name
        self.prompt = PLANNER_PROMPT.format(tools=TOOLS_DESCRIPTION, task="{task}")
        self.maximum_steps = maximum_steps

    def llm(self, prompt: str) -> str:
        """Сalling the llm to get a response."""
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=[
                {"role": "user", "content": prompt},
            ],
            temperature=0,
        )
        return response.choices[0].message.content

    def get_step_by_step_plan(self, response):
        regex_pattern = r"Plan:\s*(.+)\s*(#E\d+)\s*=\s*(\w+)\s*\[([^\]]+)\]"
        matches = re.findall(regex_pattern, response)
        return {"steps": matches, "plan_string": response}

    def run(self, task: Text) -> List[Text]:
        task_prompt = self.prompt.format(task=task)
        response = self.llm(task_prompt).strip()
        plan = self.get_step_by_step_plan(response)
        logger.info(f"PLANNER TASK PROMPT:\n\n{task_prompt}")
        logger.info("\n_____________________________\n")
        logger.info("STEP BY STEP PLAN:\n{plan}".format(plan=plan["plan_string"]))
        logger.info("\n_____________________________\n")
        return plan


class WorkerREWOO(Agent):

    def __init__(
        self, 
        url: str = None,
        model_name: str = None,
        name: str = "rewoo_worker_agent",
        db: IDB = None, 
        context_assembler: ContextAssembler = None
    ):
        super().__init__(name)
        self.client = OpenAI(base_url=url, api_key="vllm")
        self.model_name = model_name
        self.prompt = WORKER_PROMPT
        self.db = db
        self.context_assembler = context_assembler
        self.worker_evidences = dict()

    def retrieve_context(self, task: str, top_k: int = 3) -> str:
        """Retrieve context from the database."""
        if self.db and self.context_assembler:
            retrieved = self.db.query(task, top_k)
            retrieved_context = self.context_assembler.assemble(retrieved)
            return f"\n\nquery:\n{task} \n\nresponse:\n{retrieved_context}"
        return "No database or context assembler available."

    def llm(self, request: str) -> str:
        """Сalling the llm to get a response."""
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=[
                {
                    "role": "user",
                    "content": self.prompt.format(request=request)
                },
            ], temperature=0,
        )
        return response.choices[0].message.content

    def tool(self, tool, tool_input):
        """Сalling the tool to get a result."""
        if tool == "retrieve_context":
            tool_result = self.retrieve_context(tool_input)
        elif tool == "LLM":
            tool_result = self.llm(tool_input)
        else:
            tool_result = f"unknown tool: {tool}"
        return tool_result.strip()

    def run(self, plan: dict) -> dict:
        # plan format: {"steps": matches, "plan_string": response}
        for item in plan["steps"]:
            descr, step, tool, task = item
            tool_input = task.replace('"', '')
            for var in re.findall(r"#E\d+", task):
                if var in self.worker_evidences.keys():
                    tool_input = tool_input.replace(
                        var, "[" + self.worker_evidences[var]['tool_result'] + "]"
                    )
            tool_result = self.tool(tool, tool_input)
            self.worker_evidences[step] = {
                "tool": tool,
                "tool_input": tool_input,
                "tool_result": tool_result
            }
        logger.info("WORKER EVIDENCES:\n\n{evidences}".format(
            evidences=json.dumps(self.worker_evidences, indent=4)
        ))
        logger.info("\n_____________________________\n")
        return self.worker_evidences


class SolverREWOO(Agent):

    def __init__(
        self, 
        url: str = None,
        model_name: str = None,
        name: str = "rewoo_solver_agent"
    ):
        super().__init__(name)
        self.client = OpenAI(base_url=url, api_key="vllm")
        self.model_name = model_name
        self.prompt = SOLVER_PROMPT

    def llm(self, prompt: str) -> str:
        """Сalling the llm to get a response."""
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=[
                {"role": "user", "content": prompt}
            ],
            temperature=0,
        )
        return response.choices[0].message.content

    def run(self, task, plan, evidencies):
        # plan format: {"steps": matches, "plan_string": response}
        # evidence format: {"tool": tool, "tool_input": tool_input,"tool_result": tool_result}
        completed_plan = []
        for item in plan["steps"]:
            descr, step, _, _ = item
            evidence = evidencies[step]["tool_result"]
            completed_plan.append(f"\t- Plan: '{descr}'\n\t- Evidence: '{evidence}'")
        completed_plan = '\n'.join(completed_plan)
        solve_prompt = self.prompt.format(plan=completed_plan, task=task)
        final_answer = self.llm(solve_prompt).strip()
        logger.info(f"SOLVE PROMPT:\n\n{solve_prompt}")
        logger.info("\n_____________________________\n")
        logger.info(f"FINAL ANSWER:\n\n{final_answer}")
        logger.info("\n_____________________________\n")
        return final_answer
