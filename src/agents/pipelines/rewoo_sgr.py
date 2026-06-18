from typing import List, Union, Dict, Any, Type
import logging
import json
import re

from pydantic import BaseModel, create_model, Field
from openai import OpenAI

from src.agent_constructor.agent import Agent
from src.agent_constructor.core import Text
from src.tools import BaseTool, LLMTool


from src.agents.pipelines.prompts.prompts_rewoo_sgr import (
    PLANNER_PROMPT,
    SOLVER_PROMPT,
    FEW_SHOT_REGISTRY,
)

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
        few_shot_type: str = "zero_shot"
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

        if few_shot_type not in FEW_SHOT_REGISTRY:
            raise ValueError(f"Unknown few_shot_type: {few_shot_type}. Available: {list(FEW_SHOT_REGISTRY.keys())}")
            
        self.few_shot_examples = FEW_SHOT_REGISTRY[few_shot_type]

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

        # parsed_response = self.llm(REWOO_FEW_SHOT_COT_EXAMPLES + task_prompt)
        parsed_response = self.llm(self.few_shot_examples + task_prompt)
        # parsed_response = self.llm(task_prompt)

        logging.info(f"TASK:\n\n{task}")
        logging.info(_LOG_SEPARATOR)
        logging.info(f"PLANNER TASK PROMPT:\n\n{task_prompt}")
        logging.info(_LOG_SEPARATOR)
        logging.info(f"PLAN PARSED RESPONSE:\n\n{parsed_response}")
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

    def execute_tool(self, tool_name: str, arguments:  Dict[str, Any]) -> str:
        """Execute a tool with given arguments."""
        if tool_name not in self.tools_dict:
            return f"unknown tool: {tool_name}"

        tool = self.tools_dict[tool_name]
        try:
            result = tool(**arguments)
            return str(result) if result is not None else ""
        except Exception as e:
            return f"Error executing tool {tool_name}: {str(e)}"
    
    def _substitute_evidence_variables(self, tool_input: Union[Dict[str, Any], str], depends_on: List[str] = None) -> Union[Dict[str, Any], str]:
        """Replace evidence variables (#E1, #E2, etc.) with actual tool_result values.
        
        If depends_on is provided, only substitute variables that are in the depends_on list.
        This ensures that steps only access evidence they explicitly depend on.
        """
        if isinstance(tool_input, str):
            for var in re.findall(r"#E\d+", tool_input):
                if depends_on is None or var in depends_on:
                    if var in self.worker_evidences:
                        tool_input = tool_input.replace(
                            var, f"[{self.worker_evidences[var]['tool_result']}]"
                        )
            return tool_input
        elif isinstance(tool_input, dict):
            return {
                key: self._substitute_evidence_variables(value, depends_on) 
                for key, value in tool_input.items()
            }
        return tool_input

    def run(self, plan) -> Dict[str, Any]:
        """Execute the plan steps and collect evidence directly from Plan model."""
        for step in plan.steps:
            tool_name = step.tool
            tool_input = step.args.model_dump() if hasattr(step.args, "model_dump") else step.args
            
            logging.info(f"STEP ARGS:\n\n{step.args}")
            logging.info(_LOG_SEPARATOR)

            evidence_tag = step.evidence_tag

            tool_input = self._substitute_evidence_variables(tool_input, step.depends_on)
            
            logging.info(f"TOOL INPUT AFTER CALL:\n\n{tool_input}")
            logging.info(_LOG_SEPARATOR)

            tool_result = self.execute_tool(tool_name, tool_input)

            self.worker_evidences[evidence_tag] = {
                "tool": tool_name,
                "tool_input": tool_input,
                "tool_result": tool_result,
            }

        logging.info(f"WORKER EVIDENCES:\n\n{json.dumps(self.worker_evidences, indent=4)}")
        logging.info(_LOG_SEPARATOR)
        return self.worker_evidences

class SolverResponse(BaseModel):
    thought: str = Field(description="Internal chain of thought and logic analysis before writing the final solution.")
    response: str = Field(description="The final direct answer, code snippet, or solution with no extra conversational words.")

class SolverREWOOSGR(Agent):

    def __init__(
        self,
        url: str = None,
        model_name: str = None,
        name: str = "rewoo_solver_agent",
        temperature: float = 0.0,
        few_shot_type: str = "solver_cot"
    ):
        super().__init__(name)
        self.client = OpenAI(base_url=url, api_key="vllm")
        self.model_name = model_name
        self.temperature = temperature
        self.prompt = SOLVER_PROMPT

        if few_shot_type not in FEW_SHOT_REGISTRY:
            raise ValueError(f"Unknown few_shot_type: {few_shot_type}. Available: {list(FEW_SHOT_REGISTRY.keys())}")
            
        self.few_shot_examples = FEW_SHOT_REGISTRY[few_shot_type]

    # def llm(self, prompt: str) -> str:
    #     """Сalling the llm to get a response."""
    #     response = self.client.chat.completions.create(
    #         model=self.model_name,
    #         messages=[{"role": "user", "content": prompt}],
    #         temperature=self.temperature,
    #     )
    #     return response.choices[0].message.content

    def llm(self, prompt: str) -> SolverResponse:
        """LLM calling"""
        response = self.client.chat.completions.parse(
            model=self.model_name,
            messages=[{"role": "user", "content": prompt}],
            temperature=self.temperature,
            response_format=SolverResponse
        )
        return response.choices[0].message.parsed

    def _build_completed_plan_str(self, plan, evidencies) -> str:
        completed_plan = []
        for step in plan.steps:
            step_descr = step.plan
            evidence_tag = step.evidence_tag
            evidence_result = evidencies.get(evidence_tag, {}).get("tool_result", "No evidence available")
            completed_plan.append(f"\t- Plan: '{step_descr}'\n\t- Evidence: '{evidence_result}'")
        return "\n".join(completed_plan)
      
    # def run(self, task, plan, evidencies):
    #     # plan format: {"steps": matches, "plan_string": response}
    #     # evidence format: {"tool": tool, "tool_input": tool_input,"tool_result": tool_result}
    #     completed_plan_str = self._build_completed_plan_str(plan, evidencies)
    #     solve_prompt = self.prompt.format(plan=completed_plan_str, task=task)
        
    #     final_answer = self.llm(solve_prompt).strip()
        
    #     logging.info(f"SOLVE PROMPT:\n\n{solve_prompt}")
    #     logging.info(_LOG_SEPARATOR)
    #     logging.info(f"FINAL ANSWER:\n\n{final_answer}")
    #     logging.info(_LOG_SEPARATOR)
    #     return final_answer
      
    def run(self, task, plan, evidencies):
        completed_plan_str = self._build_completed_plan_str(plan, evidencies)
        solve_prompt = self.prompt.format(plan=completed_plan_str, task=task)
        
        full_prompt = self.few_shot_examples + "\n\n" + solve_prompt
        
        parsed_result = self.llm(full_prompt)
        
        logging.info(f"SOLVE PROMPT:\n\n{full_prompt}")
        logging.info(_LOG_SEPARATOR)
        logging.info(f"SOLVER INTERNAL THOUGHT:\n\n{parsed_result.thought}")
        logging.info(_LOG_SEPARATOR)
        logging.info(f"FINAL ANSWER (RESPONSE ONLY):\n\n{parsed_result.response}")
        logging.info(_LOG_SEPARATOR)
        
        return parsed_result.response

    def run_after_critic(self, task, plan, evidencies, critic):
        # plan format: {"steps": matches, "plan_string": response}
        # evidence format: {"tool": tool, "tool_input": tool_input,"tool_result": tool_result}
        completed_plan = []
        for step in plan.steps:
            step_descr = step.plan
            evidence_tag = step.evidence_tag
            evidence_result = evidencies.get(evidence_tag, {}).get("tool_result", "No evidence available")
            completed_plan.append(f"\t- Plan: '{step_descr}'\n\t- Evidence: '{evidence_result}'")
        
        completed_plan_str = "\n".join(completed_plan) + "\nCritic: " + critic
        solve_prompt = self.prompt.format(plan=completed_plan_str, task=task)
        
        final_answer = self.llm(solve_prompt).strip()
        
        logging.info(f"SOLVE PROMPT:\n\n{solve_prompt}")
        logging.info(_LOG_SEPARATOR)
        logging.info(f"FINAL ANSWER AFTER CRITIC:\n\n{final_answer}")
        logging.info(_LOG_SEPARATOR)
        return final_answer
