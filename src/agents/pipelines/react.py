import logging
import re
import json
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple

from openai import OpenAI

from src.agent_constructor.agent import Agent
from src.tools import BaseTool, LLMTool


REACT_SYSTEM_PROMPT = """You are an autonomous AI agent using the ReAct (Reasoning + Acting) framework.
        
        ### CRITICAL RULES
        1. You may ONLY use tools listed under "### AVAILABLE TOOLS" below.
        2. Tool names are CASE-SENSITIVE and must match EXACTLY.
        3. NEVER invent, guess, or modify tool names.
        4. If no tool fits → explain in Thought, then Action: Finish.
        
        Available tool names only: {tool_names}

        ### STRICT OUTPUT FORMAT
        Answer the following questions by reasoning step-by-step. Use the following format:

        Thought: <your reasoning about what to do next>
        Action: <AVAILABLE tool name>[<arguments>] or <Finish>
        
        If Action is Finish, then you don't need to write arguments!
        
        You may repeat the Thought/Action cycle multiple times.
        When you have the final answer, use: Action: Finish
        
        ### EXAMPLES
        
        Question: What is 2 + 2 * 3?
        Thought: I need to evaluate this mathematical expression.
        Action: calculator[2 + 2 * 3]
        Observation: 8
        Thought: The calculation is complete.
        Action: Finish

        ### AVAILABLE TOOLS
        {tools_formatted}

        ### RULES
        1. Always start with "Thought:".
        2. Use only exact tool names from the list above, or "Finish" when done.
        3. Arguments go inside square brackets immediately after the action name: action_name[arguments].
        4. When you have the final answer, output: Action: Finish.
        5. One action per response. Stop after outputting Action.
    """

REACT_FINISH_PROMPT = """Based on the information provided, generate only the final answer for the task without Thought and Action, only final text: {task}"""


_LOG_SEPARATOR = f"\n{'_' * 20}\n"


class PlainFormatter(logging.Formatter):
    def format(self, record):
        return record.getMessage()


def setup_logger(log_file) -> logging.Logger:
    logger = logging.getLogger("agent")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    logger.propagate = False

    Path(log_file).parent.mkdir(parents=True, exist_ok=True)

    file_handler = logging.FileHandler(log_file, encoding="utf-8")
    file_handler.setFormatter(PlainFormatter())
    logger.addHandler(file_handler)

    return logger


logger = setup_logger("/workspace/data/react.log")


class ReActAgent(Agent):
    """ReAct agent for automatic programming tasks."""

    def __init__(
        self,
        url: str = None,
        model_name: str = None,
        temperature: float = 0.0,
        name: str = "ReActAgent",
        instruction: str = None,
        examples: list = None,
        max_iterations: int = 10,
        tools: List[BaseTool] = None,
    ):
        super().__init__(name)
        self.examples = examples or []
        self.max_iterations = max_iterations
        self.tools = tools
        
        if not self.tools:
            self.tools = [LLMTool(url=url, model_name=model_name)]
        
        self.tool_names = ", ".join(tool.name for tool in self.tools)
        self.tools_dict = {t.name: t for t in self.tools}
        self.tools_prompt = "\n\n".join([t.get_prompt_description() for t in self.tools])
        
        self.memory = []

        self.instruction = instruction or REACT_SYSTEM_PROMPT.format(
            tool_names=self.tool_names, tools_formatted=self.tools_prompt
        )

        self.client = OpenAI(base_url=url, api_key="vllm")
        self.model_name = model_name
        self.temperature = temperature

        logger.info(f"SYSTEM PROMPT: {self.instruction}")
        logger.info(_LOG_SEPARATOR)

    def llm(self, messages: List[Dict[str, Any]]) -> Any:
        """Calling the llm to get a response."""
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            temperature=self.temperature,
        )
        return response.choices[0].message

    def _finish(self, messages: List[Dict[str, Any]]) -> str:
        return self.llm(messages).content

    def _parse_action_info(self, content: str) -> Tuple[str, Optional[str]]:
        """Parsing the final response."""

        action_prefix = "Action: "

        lines = content.strip().split("\n")
        action_block = None
        for line in reversed(lines):
            if line.strip().startswith(action_prefix):
                action_block = line.strip()
                break

        if not action_block:
            raise ValueError(f"Could not parse LLM Output: {content}")

        action_str = action_block[len(action_prefix) :]

        re_matches = re.search(r"^(.*?)(?:\[(.*)\])?$", action_str)

        if re_matches is None:
            raise ValueError(f"Could not parse action directive: {action_str}")
        action = re_matches.group(1)

        if "Finish" not in action:
            action_input = re_matches.group(2)
        else:
            action = "Finish"
            action_input = None

        return action, action_input

    def _parse_thought(self, content: str) -> str | None:
        """Parsing the 'Thought' field."""
        if not content:
            return None

        thought_match = re.search(r"Thought:\s*(.*?)(?=\s*Action:)", content, re.DOTALL)

        if not thought_match:
            return None
        return thought_match.group(1).strip()

    def _parse(self, content: str) -> Tuple[Optional[str], str, Optional[str]]:
        """Parsing the information."""
        if not content:
            return None

        thought = self._parse_thought(content)

        action, action_input = self._parse_action_info(content)

        return thought, action, action_input

    def execute_tool(self, tool_name: str, argument: str) -> str:
        """Calling the tool."""
        if tool_name not in self.tools_dict:
            available = ", ".join(self.tools_dict.keys())
            raise ValueError(
                f"Unknown tool '{tool_name}'. Available tools: {available}"
            )

        tool = self.tools_dict[tool_name]

        try:
            result = tool(argument)
            return result
        except Exception as e:
            return f"Error executing tool {tool_name}: {str(e)}"

    def run(self, task: str) -> str:
        """Run the ReAct agent to solve the programming task."""

        logger.info(f"TASK: {task}")
        logger.info(_LOG_SEPARATOR)

        self.memory = [
            {"role": "system", "content": self.instruction},
            {"role": "user", "content": task},
        ]

        for idx in range(self.max_iterations):

            message = self.llm(self.memory)
            self.memory.append({"role": "assistant", "content": message.content})
            thought, action, action_input = self._parse(message.content)

            logger.info(f"STEP {idx+1}:")
            logger.info(_LOG_SEPARATOR)
            logger.info(f"THOUGHT: {thought}")
            logger.info(_LOG_SEPARATOR)
            logger.info(f"ACTION: {action}")
            logger.info(_LOG_SEPARATOR)
            logger.info(f"ACTION INPUT: {action_input}")
            logger.info(_LOG_SEPARATOR)

            if action == "Finish":
                self.memory.append(
                    {"role": "user", "content": REACT_FINISH_PROMPT.format(task=task)}
                )
                logger.info(f"FINISH PROMPT: {REACT_FINISH_PROMPT.format(task=task)}")
                logger.info(_LOG_SEPARATOR)
                answer = self._finish(self.memory)
                logger.info(f"FINAL ANSWER: {answer}")
                logger.info(_LOG_SEPARATOR)
                return answer

            else:

                observation = self.execute_tool(action, action_input)

                self.memory.append(
                    {
                        "role": "user",
                        "content": f"Observation from {action}: {observation}",
                    }
                )

                logger.info(f"OBSERVATION: {observation}")
                logger.info(_LOG_SEPARATOR)

                continue

        last_message = self.memory[-1].get("content", "") if self.memory else ""
        return last_message or "Maximum iterations reached without solution."
