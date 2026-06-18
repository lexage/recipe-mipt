from typing import List
import logging
import json
import re

from openai import OpenAI

from src.agent_constructor.agent import Agent
from src.agent_constructor.core import Text
from src.tools import BaseTool, LLMTool

from src.agents.pipelines.prompts.ds1000.prompts_rewoo import (
    PLANNER_PROMPT,
    WORKER_PROMPT,
    SOLVER_PROMPT,
)

_LOG_SEPARATOR = f"\n{'_' * 20}\n"


class PlannerREWOO(Agent):

    def __init__(
        self,
        url: str = None,
        model_name: str = None,
        temperature: float = 0.0,
        name: str = "rewoo_planner_agent",
        maximum_steps: int = 5,
        tools: List[BaseTool] = None,
    ):
        super().__init__(name)
        self.client = OpenAI(base_url=url, api_key="vllm")
        self.model_name = model_name
        self.temperature = temperature
        self.tools = tools

        if not self.tools:
            self.tools = [LLMTool(url=url, model_name=model_name)]

        self.tool_names = ", ".join(tool.name for tool in self.tools)
        self.tools_dict = {t.name: t for t in self.tools}
        self.tools_prompt = "\n\n".join(
            [t.get_prompt_description() for t in self.tools]
        )

        self.prompt = PLANNER_PROMPT.format(
            tool_names=self.tool_names, tools_formatted=self.tools_prompt, task="{task}"
        )
        self.maximum_steps = maximum_steps

    def llm(self, prompt: str) -> str:
        """Сalling the llm to get a response."""
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=[
                {"role": "user", "content": prompt},
            ],
            temperature=self.temperature,
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
        logging.info(f"PLANNER TASK PROMPT:\n\n{task_prompt}")
        logging.info(_LOG_SEPARATOR)
        logging.info("STEP BY STEP PLAN:\n{plan}".format(plan=plan["plan_string"]))
        logging.info(_LOG_SEPARATOR)
        return plan


class WorkerREWOO(Agent):

    def __init__(
        self,
        url: str = None,
        model_name: str = None,
        name: str = "rewoo_worker_agent",
        tools: List[BaseTool] = None,
    ):
        super().__init__(name)
        self.prompt = WORKER_PROMPT
        self.tools = tools

        if not self.tools:
            self.tools = [LLMTool(url=url, model_name=model_name)]

        self.tools_dict = {t.name: t for t in self.tools}
        self.worker_evidences = dict()

    def execute_tool(self, tool_name: str, argument: str) -> str:
        """Calling the tool."""

        if tool_name not in self.tools_dict:
            return f"unknown tool: {tool_name}"

        tool = self.tools_dict[tool_name]

        try:
            result = tool(argument)
            return result.strip()
        except Exception as e:
            return f"Error executing tool {tool_name}: {str(e)}"

    def run(self, plan: dict) -> dict:
        # plan format: {"steps": matches, "plan_string": response}
        for item in plan["steps"]:
            descr, step, tool, task = item
            tool_input = task.replace('"', "")
            for var in re.findall(r"#E\d+", task):
                if var in self.worker_evidences.keys():
                    tool_input = tool_input.replace(
                        var, "[" + self.worker_evidences[var]["tool_result"] + "]"
                    )

            tool_result = self.execute_tool(tool, tool_input)

            self.worker_evidences[step] = {
                "tool": tool,
                "tool_input": tool_input,
                "tool_result": tool_result,
            }
        logging.info(
            "WORKER EVIDENCES:\n\n{evidences}".format(
                evidences=json.dumps(self.worker_evidences, indent=4)
            )
        )
        logging.info(_LOG_SEPARATOR)
        return self.worker_evidences


class SolverREWOO(Agent):

    def __init__(
        self,
        url: str = None,
        model_name: str = None,
        name: str = "rewoo_solver_agent",
        temperature: float = 0.0,
    ):
        super().__init__(name)
        self.client = OpenAI(base_url=url, api_key="vllm")
        self.model_name = model_name
        self.temperature = temperature
        self.prompt = SOLVER_PROMPT

    def llm(self, prompt: str) -> str:
        """Сalling the llm to get a response."""
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=[{"role": "user", "content": prompt}],
            temperature=self.temperature,
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
        completed_plan = "\n".join(completed_plan)
        solve_prompt = self.prompt.format(plan=completed_plan, task=task)
        final_answer = self.llm(solve_prompt).strip()
        logging.info(f"SOLVE PROMPT:\n\n{solve_prompt}")
        logging.info(_LOG_SEPARATOR)
        logging.info(f"FINAL ANSWER:\n\n{final_answer}")
        logging.info(_LOG_SEPARATOR)
        return final_answer