from typing import List, Union, Dict, Any, Type
import logging
import json
import re

from pydantic import BaseModel, create_model
from openai import OpenAI

from src.agent_constructor.agent import Agent
from src.agent_constructor.core import Text
from src.tools import BaseTool, LLMTool


PLANNER_PROMPT = """For the following task, make plans that can solve the problem step by step. For each plan, indicate \
which external tool together with tool input to retrieve evidence. You can store the evidence into a \
variable #E that can be called by later tools.

Available tool names only: {tool_names}

{tools_formatted}

You must respond with a structured JSON format containing a list of steps. Each step should have:
- step_id: integer (starting from 1)
- plan: string description of what this step does
- tool: string name of the tool to use
- args: object with tool-specific arguments
- evidence_tag: string like "#E1", "#E2", etc.
- depends_on: list of evidence tags this step depends on (e.g., ["#E1"] or [])

For example:
Task: Thomas, Toby, and Rebecca worked a total of 157 hours in one week. Thomas worked x
hours. Toby worked 10 hours less than twice what Thomas worked, and Rebecca worked 8 hours
less than Toby. How many hours did Rebecca work?

{{
  "steps": [
    {{
      "step_id": 1,
      "plan": "Given Thomas worked x hours, translate the problem into algebraic expressions and solve with Wolfram Alpha",
      "tool": "WolframAlpha",
      "args": {{"query": "Solve x + (2x − 10) + ((2x − 10) − 8) = 157"}},
      "evidence_tag": "#E1",
      "depends_on": []
    }},
    {{
      "step_id": 2,
      "plan": "Find out the number of hours Thomas worked",
      "tool": "llm",
      "args": {{"query": "What is x, given #E1"}},
      "evidence_tag": "#E2",
      "depends_on": ["#E1"]
    }},
    {{
      "step_id": 3,
      "plan": "Calculate the number of hours Rebecca worked",
      "tool": "calculator",
      "args": {{"expression": "(2 * #E2 - 10) - 8"}},
      "evidence_tag": "#E3",
      "depends_on": ["#E2"]
    }}
  ]
}}

Begin!
Describe your plans with rich details.

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

_LOG_SEPARATOR = f"\n{'_' * 20}\n"


def _create_step_model(tools: List[BaseTool]) -> Type[BaseModel]:
    """Create dynamic Step model with proper tool argument types."""
    tool_args_types = [tool.args for tool in tools if hasattr(tool, 'args') and tool.args is not None]
    
    if tool_args_types:
        ToolArgs = tool_args_types[0] if len(tool_args_types) == 1 else Union[tuple(tool_args_types)]
    else:
        ToolArgs = Dict[str, Any]
    
    return create_model(
        'Step',
        step_id=(int, ...),
        plan=(str, ...),
        tool=(str, ...),
        args=(ToolArgs, ...),
        evidence_tag=(str, ...),
        depends_on=(List[str], ...),
        __base__=BaseModel
    )


def _create_plan_model(tools: List[BaseTool]) -> Type[BaseModel]:
    """Create Plan model that uses dynamic Step models."""
    StepModel = _create_step_model(tools)
    return create_model('Plan', steps=(List[StepModel], ...), __base__=BaseModel)


class PlannerREWOOSGR(Agent):

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

        self.prompt = PLANNER_PROMPT
        self.maximum_steps = maximum_steps

    def llm(self, prompt: str) -> BaseModel:
        """Call LLM to get a structured response."""
        Plan = _create_plan_model(self.tools)
        
        response = self.client.chat.completions.parse(
            model=self.model_name,
            messages=[{"role": "user", "content": prompt}],
            temperature=self.temperature,
            response_format=Plan
        )

        return response.choices[0].message.parsed

    def run(self, task: Text) -> Dict[str, Any]:
        task_prompt = self.prompt.format(
            task=task,
            tool_names=self.tool_names,
            tools_formatted=self.tools_prompt
        )
        parsed_response = self.llm(task_prompt)
        
        logging.info(f"PARSED RESPONSE:\n\n{parsed_response}")
        logging.info(f"PLANNER TASK PROMPT:\n\n{task_prompt}")
        logging.info(_LOG_SEPARATOR)
        
        return parsed_response


class WorkerREWOOSGR(Agent):

    def __init__(
        self,
        url: str = None,
        model_name: str = None,
        name: str = "rewoo_worker_agent",
        tools: List[BaseTool] = None,
    ):
        super().__init__(name)
        self.tools = tools or [LLMTool(url=url, model_name=model_name)]
        self.tools_dict = {t.name: t for t in self.tools}
        self.worker_evidences = {}

    def _get_expected_args(self, tool: BaseTool) -> List[str]:
        """Get expected argument names for a tool."""
        if hasattr(tool, 'args') and tool.args is not None and hasattr(tool.args, '__annotations__'):
            return list(tool.args.__annotations__.keys())
        return []

    #TODO нужно поменять тип arguments
    def execute_tool(self, tool_name: str, arguments: str) -> str:
        """Execute a tool with given arguments."""
        if tool_name not in self.tools_dict:
            return f"unknown tool: {tool_name}"

        tool = self.tools_dict[tool_name]
        try:
            if arguments.strip():
                parsed_args = self._parse_tool_arguments(arguments, tool)
                result = tool(**parsed_args)
            else:
                result = tool()
            return str(result) if result is not None else ""
        except Exception as e:
            return f"Error executing tool {tool_name}: {str(e)}"

    def _substitute_evidence_variables(self, tool_input: str) -> str:
        """Replace evidence variables (#E1, #E2, etc.) with actual results."""
        for var in re.findall(r"#E\d+", tool_input):
            if var in self.worker_evidences:
                tool_input = tool_input.replace(
                    var, f"[{self.worker_evidences[var]['tool_result']}]"
                )
        return tool_input

    def run(self, plan: Plan) -> Dict[str, Any]:
        """Execute the plan steps and collect evidence."""
        for descr, step, tool, task in plan["steps"]:
            tool_input = task.replace('"', "")
            tool_input = self._substitute_evidence_variables(tool_input)
            tool_result = self.execute_tool(tool, tool_input)

            self.worker_evidences[step] = {
                "tool": tool,
                "tool_input": tool_input,
                "tool_result": tool_result,
            }
            
        logging.info(f"WORKER EVIDENCES:\n\n{json.dumps(self.worker_evidences, indent=4)}")
        logging.info(_LOG_SEPARATOR)
        return self.worker_evidences


class SolverREWOOSGR(Agent):

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