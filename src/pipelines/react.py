import re
import logging
import json
from openai import OpenAI
from pathlib import Path
from typing import List, Dict, Any
from src.agent_constructor.agent import Agent
from src.tools import BaseTool


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


logger = setup_logger("/workspace/data/react.log")


class ReActAgent(Agent):
    """ReAct agent for automatic programming tasks."""
    
    def __init__(
        self, 
        url: str = None,
        model_name: str = None,
        name: str = "ReActAgent", 
        instruction: str = None, 
        examples: list = None,
        max_iterations: int = 10, 
        tools: List[BaseTool] = None,
    ):
        super().__init__(name)
        self.examples = examples or []
        self.max_iterations = max_iterations
        
        tools = tools or []
        self.tools = tools
        self.tools_dict = {t.name: t for t in tools}
        self.tools_schema = [t.get_schema() for t in tools]
        
        logger.info(f"TOOLS_SCHEMA: {self.tools_schema}")
        logger.info(f"\n{'_'*20}\n")
        
        self.memory = []
        
        self.instruction = instruction or self._build_system_prompt()
        self.client = OpenAI(base_url=url, api_key="vllm")
        self.model_name = model_name
        
        logger.info(f"SYSTEM_PROMPT: {self.instruction}")
        logger.info(f"\n{'_'*20}\n")
        
    def _format_tools_for_prompt(self) -> str:
        """Formatting information about tools in a readable format for industrial purposes."""
        formatted_tools = []
        
        for tool in self.tools_schema:
            func = tool['function']
            name = func['name']
            description = func['description']
            params = func.get('parameters', {})
            properties = params.get('properties', {})
            required = params.get('required', [])
            
            params_block = []
            for param_name, param_info in properties.items():
                param_type = param_info.get('type', 'string')
                param_desc = param_info.get('description', '')
                is_required = param_name in required
                req_mark = "REQUIRED" if is_required else "optional"
                
                params_block.append(
                    f"      - {param_name} ({param_type}, {req_mark}): {param_desc}"
                )
            
            params_str = "\n".join(params_block) if params_block else "      No parameters"
            
            tool_str = f"""  - {name}: {description}
                Parameters:
            {params_str}"""
            formatted_tools.append(tool_str)
        
        return "\n\n".join(formatted_tools)

    def _build_system_prompt(self) -> str:
        """Formation of a system prompt."""
        
        tools_formatted = self._format_tools_for_prompt()
        tool_names = ", ".join([t['function']['name'] for t in self.tools_schema])
        
        base_prompt = f"""You are an autonomous AI agent using the ReAct (Reasoning + Acting) framework.
        You have access to the following tools: {tool_names}.

        ### STRICT OUTPUT FORMAT
        You must output your response in exactly one of the following two formats. Do not add any extra text, markdown, or explanations outside this format.

        FORMAT 1 (If you need to use a tool):
        Think: <your reasoning about what to do next>
        Action: <exact tool name from the list above>
        Action Input: <valid JSON object with arguments>

        FORMAT 2 (If you have the final answer):
        Think: <reasoning that you are done>
        Final Answer: <your direct response to the user>

        ### AVAILABLE TOOLS
        {tools_formatted}

        ### EXAMPLES

        User: Find the user with ID 123.
        Assistant:
        Think: I need to search the database for a user with ID 123.
        Action: db_search
        Action Input: {{"user_id": 123}}

        User: What is the result?
        Assistant:
        Think: The tool returned the user data, so I can now answer.
        Final Answer: The user with ID 123 is John Doe.

        ### RULES
        1. ALWAYS start with "Think:".
        2. If you don't know the answer, you MUST use an Action. Do not guess.
        3. "Action Input" must be raw JSON. Do NOT wrap it in markdown code blocks (no ```json).
        4. Use only the exact tool names listed above.
        5. Stop immediately after generating "Final Answer".
        """
        return base_prompt
    
    def llm(self, messages: List[Dict[str, Any]]) -> Any:
        """Calling the llm to get a response."""
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            temperature=0,
        )
        return response.choices[0].message
    
    def _parse_final_answer(self, content: str) -> str | None:
        """Parsing the final response."""
        if not content:
            return None
        
        final_match = re.search(r'Final Answer:\s*(.*)', content, re.DOTALL)
        if not final_match:
            return None
        return final_match.group(1).strip()
    
    def _parse_think(self, content: str) -> str | None:
        """Parsing the 'Think' field."""
        if not content:
            return None
        
        think_match = re.search(r'Think:\s*(.*)', content)
        if not think_match:
            return None
        return think_match.group(1).strip()

    def _parse_tool_call_from_content(self, content: str) -> Dict[str, Any] | None:
        """Parsing the selected tool and the arguments for them."""
        if not content:
            return None
        
        action_match = re.search(r'Action:\s*([a-zA-Z_][a-zA-Z0-9_]*)', content)
        if not action_match:
            return None
        
        json_str = None
        input_match = re.search(r'Action Input:\s*(\{.*\})', content, re.DOTALL)
        if input_match:
            json_str = input_match.group(1)
        else:
            json_matches = re.findall(r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}', content)
            json_str = json_matches[-1] if json_matches else None
        
        if not json_str:
            return None
            
        try:
            return {"name": action_match.group(1), "arguments": json.loads(json_str)}
        except json.JSONDecodeError:
            return None
        
    def execute_tool(self, tool_name: str, arguments: Dict[str, Any]) -> str:
        """Calling the tool."""
        if tool_name not in self.tools_dict:
            available = ", ".join(self.tools_dict.keys())
            raise ValueError(f"Unknown tool '{tool_name}'. Available tools: {available}")

        tool_instance = self.tools_dict[tool_name]
        
        try:
            result = tool_instance.run(**arguments)
            return result
        except Exception as e:
            return f"Error executing tool {tool_name}: {str(e)}"
    
    def run(self, task: str) -> str:
        """Run the ReAct agent to solve the programming task."""
        
        logger.info(f"TASK: {task}")
        logger.info(f"\n{'_'*20}\n")
        
        self.memory = [
            {"role": "system", "content": self.instruction},
            {"role": "user", "content": task}
        ]
        
        for idx in range(self.max_iterations):
            logger.info(f"STEP {idx+1}:")
            logger.info(f"\n{'_'*20}\n")
            
            logger.info(f"INSTRUCTION: {task}")
            logger.info(f"\n{'_'*20}\n")


            message = self.llm(self.memory)
            
            self.memory.append({"role": "assistant", "content": message.content})
            
            tool_call_data = None
            
            think = self._parse_think(message.content)
            parsed = self._parse_tool_call_from_content(message.content)
            
            logger.info(f"THINK: {think}")
            logger.info(f"\n{'_'*20}\n")

            
            if parsed:
                tool_call_data = {
                    "name": parsed["name"],
                    "arguments": parsed["arguments"],
                    "id": None
                }
                
            if tool_call_data:
                tool_name = tool_call_data["name"]
                tool_args = tool_call_data["arguments"]
                
                logger.info(f"DECISION: Calling tool '{tool_name}' with args {tool_args}")
                logger.info(f"\n{'_'*20}\n")

                observation = self.execute_tool(tool_name, tool_args)
                
                logger.info(f"OBSERVATION: {observation}")
                logger.info(f"\n{'_'*20}\n")
                
                self.memory.append({
                    "role": "user",
                    "content": f"Observation from {tool_name}: {observation}"
                })
                
                continue
            
            else:
                final_answer = self._parse_final_answer(message.content)
                logger.info(f"FINAL_ANSWER: {final_answer}")
                logger.info(f"\n{'_'*20}\n")

                return final_answer or message.content
            
        last_message = self.memory[-1].get("content", "") if self.memory else ""
        return last_message or "Maximum iterations reached without solution."