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
        Action: <AVAILABLE tool name> or <Finish>
        Action Input: <arguments>  # if Action is Finish then write None
        
        If Action is Finish, then write in Action Input: None
        
        When you have the final answer, use: Action: Finish
        
        ### EXAMPLES
        
        Question: What is 2 + 2 * 3?
        Thought: I need to evaluate this mathematical expression.
        Action: calculator
        Action Input: 2 + 2 * 3
        Observation: 8
        Thought: The calculation is complete.
        Action: Finish
        Action Input: None

        ### AVAILABLE TOOLS
        {tools_formatted}

        ### RULES
        1. Always start with "Thought:".
        2. Use only exact tool names from the list above, or "Finish" when done.
        3. Arguments go on a separate line after Action: Action Input: <arguments>.
        4. When you have the final answer, output: Action: Finish
        5. One action per response. Stop after outputting Action.
    """

REACT_FINISH_PROMPT = """Based on the information provided, generate only the final answer for the task without Thought and Action, only final text: {task}"""

_LOG_SEPARATOR = f"\n{'_' * 20}\n"


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
        self.tools_prompt = "\n\n".join(
            [t.get_prompt_description() for t in self.tools]
        )

        self.memory = []

        self.instruction = instruction or REACT_SYSTEM_PROMPT.format(
            tool_names=self.tool_names, tools_formatted=self.tools_prompt
        )

        self.client = OpenAI(base_url=url, api_key="vllm")
        self.model_name = model_name
        self.temperature = temperature

        logging.info(f"SYSTEM PROMPT: {self.instruction}")
        logging.info(_LOG_SEPARATOR)

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

        action_match = re.search(r"Action:\s*(.*?)(?=\s*Action Input:)", content, re.DOTALL)

        if not action_match:
            raise ValueError(f"Could not parse LLM Output: {content}")

        action = action_match.group(1).strip()

        action_input = content.split("Action Input:")[1]

        stop_words = ["Thought:", "Observation:", "Action:"]
        for word in stop_words:
            if word in action_input:
                action_input = action_input.split(word)[0]
        
        if "Finish" in action:
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

        logging.info(f"TASK: {task}")
        logging.info(_LOG_SEPARATOR)

        self.memory = [
            {"role": "system", "content": self.instruction},
            {"role": "user", "content": task},
        ]

        for idx in range(self.max_iterations):

            message = self.llm(self.memory)
            self.memory.append({"role": "assistant", "content": message.content})
            thought, action, action_input = self._parse(message.content)

            logging.info(f"STEP {idx+1}:")
            logging.info(_LOG_SEPARATOR)
            logging.info(f"THOUGHT: {thought}")
            logging.info(_LOG_SEPARATOR)
            logging.info(f"ACTION: {action}")
            logging.info(_LOG_SEPARATOR)
            logging.info(f"ACTION INPUT: {action_input}")
            logging.info(_LOG_SEPARATOR)

            if action == "Finish":
                self.memory.append(
                    {"role": "user", "content": REACT_FINISH_PROMPT.format(task=task)}
                )
                logging.info(f"FINISH PROMPT: {REACT_FINISH_PROMPT.format(task=task)}")
                logging.info(_LOG_SEPARATOR)
                answer = self._finish(self.memory)
                logging.info(f"FINAL ANSWER: {answer}")
                logging.info(_LOG_SEPARATOR)
                return answer

            else:

                observation = self.execute_tool(action, action_input)

                self.memory.append(
                    {
                        "role": "user",
                        "content": f"Observation from {action}: {observation}",
                    }
                )

                logging.info(f"OBSERVATION: {observation}")
                logging.info(_LOG_SEPARATOR)

                continue

        last_message = self.memory[-1].get("content", "") if self.memory else ""
        return last_message or "Maximum iterations reached without solution."


