import logging
import re
import json
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple

from openai import OpenAI

from src.agent_constructor.agent import Agent
from src.tools import BaseTool, LLMTool

# REACT_SYSTEM_PROMPT = """You are an autonomous AI agent using the ReAct (Reasoning + Acting) framework.
            
# ### CRITICAL RULES
# 1. You may ONLY use tools listed under "### AVAILABLE TOOLS" below.
# 2. Tool names are CASE-SENSITIVE and must match EXACTLY.
# 3. NEVER invent, guess, or modify tool names.
# 4. Each action accepts EXACTLY ONE argument: a plain text string. NO JSON, NO dictionaries, NO multiple parameters.

# ### AVAILABLE TOOLS (REMINDER - CHECK BEFORE EACH ACTION)
# {tools_formatted}

# ### ⚠️ TOOLS POLICY - CRITICAL
# ✅ You may ONLY use these exact tool names: {tool_names}
# ❌ NEVER use any other name, even if it seems logical
# ❌ NEVER invent, abbreviate, or modify tool names
# ✅ If unsure, use 'llm' to ask for clarification

# ### STRICT OUTPUT FORMAT
# Answer the following questions by reasoning step-by-step. Use the following format:

# Thought: <your reasoning about what to do next>
# Action: <EXACT tool name from the available list ONLY>
# Action Input: <a single plain text string argument - MUST NOT BE EMPTY>

# When you have the final answer, use:

# Thought: I now know the final answer
# Action: Finish

# ### EXAMPLES
# Question: What is 2 + 2 * 3?
# Thought: I need to evaluate this mathematical expression following order of operations.
# Action: calculator
# Action Input: 2 + 2 * 3

# ### RULES
# 1. Always start your response with "Thought:".
# 2. Use only exact tool names from the available list above, or "Finish" when done.
# 3. After "Action:", write ONLY the tool name — no arguments, no brackets, no parentheses.
# 4. Write the action argument on the next line as "Action Input:" — it must be a SINGLE plain text string.
# 5. NEVER use JSON, curly braces, quotes, or key-value pairs in Action Input. Just raw text.
# 6. When you have the final answer, output exactly:
# Thought: I now know the final answer
# Action: Finish
# 7. One action per response. Stop output immediately after "Action: Finish" or after "Action Input:".
# 8. Never output "Observation:" yourself — it will be provided by the system after tool execution.
# 9. If you make 2 consecutive tool errors, re-read this prompt and choose ONLY from: {tool_names}
# """

REACT_SYSTEM_PROMPT = """You are an autonomous AI agent using the ReAct (Reasoning + Acting) framework.

### 🎯 CORE PRINCIPLES
1. Think step-by-step before acting
2. Use ONLY the tools explicitly listed below
3. When done, output the final answer in the exact Finish format

### 🔧 AVAILABLE TOOLS (CASE-SENSITIVE, EXACT NAMES ONLY)
{tools_formatted}

### ⚠️ CRITICAL TOOL RULES
✅ Tool names: {tool_names}
✅ Each action takes EXACTLY ONE argument: a plain text string
❌ NEVER use JSON, dictionaries, or key-value pairs in Action Input
❌ NEVER invent, abbreviate, or modify tool names
❌ NEVER put arguments on the same line as Action:
✅ If unsure which tool to use → Action: llm with your question

### 📋 STRICT OUTPUT FORMAT (NON-NEGOTIABLE)
Every response MUST follow this exact structure:

Thought: <your reasoning about what to do next>
Action: <EXACT tool name from available list>
Action Input: <single plain text string - NO JSON, NO formatting>

### 🔄 BEFORE YOU RESPOND — QUICK CHECKLIST
✅ Did I start with "Thought:"?
✅ Is my Action one of: {tool_names} (EXACT spelling)?
✅ Is Action Input a NON-EMPTY plain text string (no JSON, no brackets)?
✅ Have I asked this SAME question in the last 2 turns? → If YES, rephrase or try code directly.

### ✅ CORRECT EXAMPLES

Example 1 - Using a tool:
Question: What users have type=1?
Thought: I need to search the database for users with type=1
Action: db_search
Action Input: select * from users where type = 1

Example 2 - Asking for clarification:
Question: How do I filter this data?
Thought: I'm not sure which tool handles filtering
Action: llm
Action Input: Which tool should I use to filter DataFrame rows by column value?

Example 3 - Finishing (MOST IMPORTANT):
Question: Calculate the sum of [1,2,3]
Thought: I can compute this directly: 1+2+3=6
Action: Finish

❌ NEVER add anything after "Action: Finish"
❌ NEVER write "Action Input:" after "Action: Finish"
❌ NEVER include code, explanations, or markers in the Finish step

### 🚫 COMMON MISTAKES TO AVOID
❌ Wrong: Action: db_search {{"query": "..."}}  ← NO JSON
❌ Wrong: Action: DB_Search  ← case-sensitive!
❌ Wrong: Action: Finish
   Action Input: result = 6  ← NOTHING after Finish!
❌ Wrong: Thought: done
   Action: Finish
   Here is my answer: ...  ← NO extra text!

### 🔄 ERROR RECOVERY
- If you get a FORMAT ERROR: re-read the STRICT OUTPUT FORMAT section
- If you get a TOOL ERROR: check available tools and try again
- After 2 consecutive errors: pause, re-read this prompt, then proceed

### ⚠️ ITERATION RULE
- NEVER ask the same question twice
- If stuck after 2 similar attempts, try a different approach
- Track your last 3 queries to avoid repetition

### 🏁 FINAL ANSWER PROTOCOL
When you output "Action: Finish":
1. Stop generating immediately
2. Do NOT add any text, code, or explanations
3. The system will request your final answer separately
4. At that point, provide ONLY the answer — no markers, no reasoning

### RULES SUMMARY
1. Start every response with "Thought:"
2. One action per response
3. Stop output immediately after "Action Input:" or "Action: Finish"
4. Never output "Observation:" — the system provides this
5. Tool names: {tool_names} (EXACT, case-sensitive)
"""

REACT_SYSTEM_FINISH_PROMPT = """You are in REACT FINAL ANSWER MODE.
Output ONLY the final answer text.
"""

REACT_FINISH_PROMPT = """
### TASK
{task}

### HISTORY
{history}

### OUTPUT REQUIREMENTS
- Provide ONLY the final answer
- No "Thought:", "Action:", "Observation:" markers
"""



_LOG_SEPARATOR = f"\n{'_' * 20}\n"


class ReActAgentObs(Agent):
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
        history_context: int = 5
    ):
        super().__init__(name)
        self.examples = examples or []
        self.max_iterations = max_iterations
        self.tools = tools
        self.history_context = history_context
        self._tool_call_retry_count = 0
        self.error_history = [] 

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

    def llm(self, messages: List[Dict[str, Any]], stop: List[str] = None) -> Any:
        """Calling the llm to get a response."""
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            temperature=self.temperature,
            stop=stop or ["Observation:"]
        )
        return response.choices[0].message

    def _finish(self, messages: List[Dict[str, Any]]) -> str:
        return self.llm(messages).content

    def _parse_action_info(self, content: str) -> Tuple[str, Optional[str], bool]:
        """Returns (action, action_input, has_multiple_actions)"""
        if "Action:" not in content:
            return "", None, False
        
        # Проверяем на наличие нескольких действий
        has_multiple = self._detect_multiple_actions(content)
        
        # Извлекаем Action (всегда берем первый)
        action_part = content.split("Action:")[1].split("\n")[0].strip()
        action = action_part.split()[0] if action_part else ""
        
        if "Finish" in action:
            return "Finish", None, has_multiple

        if "Action Input:" not in content:
            return action, None, has_multiple

        # Извлекаем все, что после Action Input:
        action_input = content.split("Action Input:")[1]

        # Ищем ближайшее стоп-слово и обрезаем по нему
        stop_words = ["Thought:", "Observation:", "Action:"]
        for word in stop_words:
            if word in action_input:
                action_input = action_input.split(word)[0]
                
        return action, action_input.strip(), has_multiple

    def _parse_thought(self, content: str) -> str | None:
        """Parsing the 'Thought' field."""
        if not content:
            return None

        thought_match = re.search(r"Thought:\s*(.*?)(?=\s*Action:)", content, re.DOTALL)

        if not thought_match:
            return None
        return thought_match.group(1).strip()

    def _parse(self, content: str) -> Tuple[Optional[str], Optional[str], Optional[str], bool]:
        """Parsing the information. Returns (thought, action, action_input, has_multiple_actions)"""
        if not content:
            return None, None, None, False

        thought = self._parse_thought(content)
        action, action_input, has_multiple = self._parse_action_info(content)

        return thought, action, action_input, has_multiple
    
    # def _parse_action_info(self, content: str) -> Tuple[str, Optional[str]]:
    #     if "Action:" not in content:
    #         return "", None
        
    #     # Извлекаем Action
    #     action_part = content.split("Action:")[1].split("\n")[0].strip()
    #     action = action_part.split()[0] if action_part else ""
        
    #     if "Finish" in action:
    #         return "Finish", None

    #     if "Action Input:" not in content:
    #         return action, None

    #     # Извлекаем все, что после Action Input:
    #     action_input = content.split("Action Input:")[1]

    #     # Ищем ближайшее стоп-слово и обрезаем по нему
    #     stop_words = ["Thought:", "Observation:", "Action:"]
    #     for word in stop_words:
    #         if word in action_input:
    #             action_input = action_input.split(word)[0]
                
    #     return action, action_input.strip()

    # def _parse_thought(self, content: str) -> str | None:
    #     """Parsing the 'Thought' field."""
    #     if not content:
    #         return None

    #     thought_match = re.search(r"Thought:\s*(.*?)(?=\s*Action:)", content, re.DOTALL)

    #     if not thought_match:
    #         return None
    #     return thought_match.group(1).strip()

    # def _parse(self, content: str) -> Tuple[Optional[str], Optional[str], Optional[str]]:
    #     """Parsing the information."""
    #     if not content:
    #         return None, None, None 

    #     thought = self._parse_thought(content)

    #     action, action_input = self._parse_action_info(content)

    #     return thought, action, action_input
        
    # def _format_history(self) -> str:
    #     """Format history"""
    #     lines = []
    #     history = self.memory if len(self.memory) < self.history_context + 1 else self.memory[-self.history_context:]
        
    #     for m in history:
    #         if m["role"] == "system" or not m.get("content"):
    #             continue
            
    #         if "ERROR" in m["content"]:
    #             continue
            
    #         role_label = {
    #             "user": "[User]",
    #             "assistant": "[Agent]",
    #         }.get(m["role"], f"[{m['role']}]")
            
    #         content = m["content"]
    #         lines.append(f"{role_label}: {content}")
        
    #     return '\n'.join(lines)

    def _format_history(self, include_errors: bool = False) -> str:
        """Format history for final answer"""
        lines = []
        history = self.memory if len(self.memory) < self.history_context + 1 else self.memory[-self.history_context:]
        
        for m in history:
            if m["role"] == "system" or not m.get("content"):
                continue
            
            # Пропускаем сообщения об ошибках, если не указано include_errors=True
            if not include_errors and ("ERROR" in m["content"] or "❌" in m["content"]):
                continue
            
            role_label = {
                "user": "[User]",
                "assistant": "[Agent]",
            }.get(m["role"], f"[{m['role']}]")
            
            content = m["content"]
            lines.append(f"{role_label}: {content}")
        
        return '\n'.join(lines)


    def _create_error_feedback(self, error_type: str, action: str, action_input: str, original_response: str, available_tools: str = None) -> str:
        """Создает структурированную обратную связь для LLM"""
        
        available_tools = available_tools or self.tool_names
        
        feedback_templates = {
            "empty_input": f"""❌ FORMAT ERROR: Your Action Input was empty.

YOUR RESPONSE:
{original_response}

PROBLEM: You provided an empty Action Input for tool '{action}'.

SOLUTION: 
1. Action Input MUST contain a non-empty string
2. DO NOT use JSON, dictionaries, or key-value pairs
3. Example: 
Action: {action}
Action Input: your query here

PLEASE PROVIDE A CORRECTED RESPONSE with a non-empty Action Input.""",

            "unknown_tool": f"""❌ TOOL ERROR: Unknown tool '{action}'.

YOUR RESPONSE:
{original_response}

PROBLEM: Tool '{action}' does not exist.

AVAILABLE TOOLS: {available_tools}

SOLUTION: Use ONLY one of the available tools listed above.
- Tool names are CASE-SENSITIVE
- 'LLM' ≠ 'llm', 'DB_Search' ≠ 'db_search'

PLEASE PROVIDE A CORRECTED RESPONSE with a valid tool name.""",

            "missing_action": f"""❌ FORMAT ERROR: Could not extract 'Action:' from your response.

YOUR RESPONSE:
{original_response[:300]}...

PROBLEM: Your response doesn't contain a properly formatted Action.

REQUIRED FORMAT:
Thought: <your reasoning>
Action: <tool_name>
Action Input: <single plain text string>

AVAILABLE TOOLS: {available_tools}

PLEASE PROVIDE A CORRECTED RESPONSE following the exact format.""",

            "multiple_actions": f"""❌ FORMAT ERROR: Multiple actions detected.

YOUR RESPONSE:
{original_response}

PROBLEM: You generated multiple Thought/Action sequences. 
You must generate ONLY ONE Thought and ONE Action per response.

SOLUTION: 
1. Generate exactly ONE Thought
2. Generate exactly ONE Action
3. Stop after Action Input
4. Wait for the Observation before your next thought

PLEASE PROVIDE A CORRECTED RESPONSE with a single Thought/Action pair.""",

            "repetition_error": f"""❌ REPETITION ERROR: You keep making the same mistake.

YOUR LAST RESPONSE:
{original_response}

PROBLEM: You have made the same type of error multiple times in a row.

SOLUTION: 
1. STOP and carefully review the required format
2. Available tools are: {available_tools}
3. Each Action Input must be a non-empty plain text string
4. DO NOT invent tool names

PLEASE TAKE A MOMENT TO READ THE INSTRUCTIONS AND PROVIDE A CORRECT RESPONSE."""
        }
        
        return feedback_templates.get(error_type, f"Unknown error. Please correct your response.\n\nYour response: {original_response}")
    
    def _detect_multiple_actions(self, content: str) -> bool:
        """Detects if there are multiple Action: tags in the response"""
        action_count = len(re.findall(r'\nAction:', content))
        return action_count > 1
    
    def execute_tool(self, tool_name: str, argument: str) -> str:
        """Calling the tool."""
        tool = self.tools_dict[tool_name]

        try:
            result = tool(argument)
            return result
        except Exception as e:
            return f"Error executing tool {tool_name}: {str(e)}"

    def run(self, task: str) -> str:
        """Run the ReAct agent to solve the programming task."""

        self._tool_call_retry_count = 0
        self.error_history = []
        available_tools = ", ".join(self.tools_dict.keys())

        logging.info(f"TASK:\n{task}")
        logging.info(_LOG_SEPARATOR)

        self.memory = [
            {"role": "system", "content": self.instruction},
            {"role": "user", "content": task},
        ]

        action = None

        for idx in range(self.max_iterations):
            
            if self._tool_call_retry_count >= 5:
                logging.info(f"AGENT FAILED: Retry limit exceeded after {idx+1} iterations. Last action: {action}")
                raise ValueError("The agent could not call the available tool after 5 attempts")

            # Добавляем превентивные подсказки при повторяющихся ошибках
            messages = [self.memory[0]] + self.memory[1:][-self.history_context:]
            
            # Анализируем паттерны ошибок
            if len(self.error_history) >= 2:
                last_errors = self.error_history[-2:]
                if all(e["type"] == "empty_input" for e in last_errors):
                    # Добавляем напоминание
                    hint = {
                        "role": "system",
                        "content": "REMINDER: Remember to provide a non-empty Action Input. DO NOT leave it blank."
                    }
                    messages.insert(1, hint)

            message = self.llm(messages, stop=["Observation:", "Observation from"])

            thought, action, action_input, has_multiple = self._parse(message.content)
            clean_content = f"Thought: {thought}\nAction: {action}\nAction Input: {action_input or ''}"
            self.memory.append({"role": "assistant", "content": clean_content.strip()})

            logging.info(f"STEP {idx+1}:")
            logging.info(_LOG_SEPARATOR)
            logging.info(f"MESSAGE CONTENT: {message.content}")
            logging.info(_LOG_SEPARATOR)
            logging.info(f"THOUGHT: {thought}")
            logging.info(_LOG_SEPARATOR)
            logging.info(f"ACTION: {action}")
            logging.info(_LOG_SEPARATOR)
            logging.info(f"ACTION INPUT: {action_input}")
            logging.info(_LOG_SEPARATOR)
            logging.info(f"HAS MULTIPLE ACTIONS: {has_multiple}")
            logging.info(_LOG_SEPARATOR)

            # Проверка на множественные действия
            if has_multiple:
                observation = self._create_error_feedback(
                    "multiple_actions",
                    action,
                    action_input,
                    message.content,
                    available_tools
                )
                self.memory.append({"role": "user", "content": f"Observation: {observation}"})
                self.error_history.append({"type": "multiple_actions", "step": idx})
                logging.info(f"MULTIPLE ACTIONS DETECTED: {observation}")
                logging.info(_LOG_SEPARATOR)
                continue

            if action == "Finish":
                # Для финального ответа используем историю без ошибок
                clean_history = self._format_history(include_errors=False)
                # Но логируем полную историю для отладки
                full_history = self._format_history(include_errors=True)
                logging.debug(f"FULL HISTORY:\n{full_history}")
                
                messages = (
                    {"role": "system", "content": REACT_SYSTEM_FINISH_PROMPT},
                    {"role": "user", "content": REACT_FINISH_PROMPT.format(task=task, history=clean_history)}
                )
                logging.info(f"FINISH PROMPT: {REACT_FINISH_PROMPT.format(task=task, history=clean_history)}")
                logging.info(_LOG_SEPARATOR)
                answer = self._finish(messages)
                logging.info(f"FINAL ANSWER: {answer}")
                logging.info(_LOG_SEPARATOR)
                return answer

            if action is None:
                self._tool_call_retry_count += 1
                self.error_history.append({"type": "missing_action", "step": idx})
                
                # Проверяем на повторяющиеся ошибки
                recent_errors = [e["type"] for e in self.error_history[-3:]]
                if recent_errors.count("missing_action") >= 2:
                    error_type = "repetition_error"
                else:
                    error_type = "missing_action"
                
                observation = self._create_error_feedback(
                    error_type,
                    "",
                    "",
                    message.content,
                    available_tools
                )
                
                self.memory.append({"role": "user", "content": f"Observation: {observation}"})
                logging.info(f"PARSE ERROR - Action missing: {observation}")
                logging.info(_LOG_SEPARATOR)
                continue
            
            if action_input is None or not action_input.strip():
                self._tool_call_retry_count += 1
                self.error_history.append({"type": "empty_input", "step": idx})
                
                # Проверяем на повторяющиеся ошибки
                recent_errors = [e["type"] for e in self.error_history[-3:]]
                if recent_errors.count("empty_input") >= 2:
                    error_type = "repetition_error"
                else:
                    error_type = "empty_input"
                
                observation = self._create_error_feedback(
                    error_type,
                    action,
                    action_input,
                    message.content,
                    available_tools
                )
                
                self.memory.append({"role": "user", "content": f"Observation: {observation}"})
                logging.info(f"PARSE ERROR - Empty input for tool: {observation}")
                logging.info(_LOG_SEPARATOR)
                continue
            
            if action not in self.tools_dict:
                self._tool_call_retry_count += 1
                self.error_history.append({"type": "unknown_tool", "step": idx, "tool": action})
                
                # Проверяем на повторяющиеся ошибки
                recent_errors = [e["type"] for e in self.error_history[-3:]]
                if recent_errors.count("unknown_tool") >= 2:
                    error_type = "repetition_error"
                else:
                    error_type = "unknown_tool"
                
                observation = self._create_error_feedback(
                    error_type,
                    action,
                    action_input,
                    message.content,
                    available_tools
                )
                
                logging.info(f"Incorrect tool '{action}'. Attempt {self._tool_call_retry_count}/5")
            else:
                self._tool_call_retry_count = 0 
                observation = self.execute_tool(action, action_input)

            self.memory.append(
                {
                    "role": "user",
                    "content": f"Observation from {action}: {observation}",
                }
            )
            
            logging.info(f"OBSERVATION: {observation}")
            logging.info(_LOG_SEPARATOR)

        logging.info(f"MAX ITERATIONS ({self.max_iterations}) REACHED without solution")
        last_message = self.memory[-1].get("content", "") if self.memory else ""
        return last_message or "Maximum iterations reached without solution."
        
    # def run(self, task: str) -> str:
    #     """Run the ReAct agent to solve the programming task."""

    #     self._tool_call_retry_count = 0
    #     available_tools = ", ".join(self.tools_dict.keys())

    #     logging.info(f"TASK:\n{task}")
    #     logging.info(_LOG_SEPARATOR)

    #     self.memory = [
    #         {"role": "system", "content": self.instruction},
    #         {"role": "user", "content": task},
    #     ]

    #     action = None

    #     for idx in range(self.max_iterations):
            
    #         if self._tool_call_retry_count >= 5:
    #             logging.info(f"AGENT FAILED: Retry limit exceeded after {idx+1} iterations. Last action: {action}")
    #             raise ValueError("The agent could not call the available tool after 5 attempts")


    #         messages = [self.memory[0]] + self.memory[1:][-self.history_context:]
    #         message = self.llm(messages, stop=["Observation:", "Observation from"])

    #         thought, action, action_input = self._parse(message.content)
    #         clean_content = f"Thought: {thought}\nAction: {action}\nAction Input: {action_input or ''}"
    #         self.memory.append({"role": "assistant", "content": clean_content.strip()})

    #         logging.info(f"STEP {idx+1}:")
    #         logging.info(_LOG_SEPARATOR)
    #         logging.info(f"MESSAGE CONTENT: {message.content}")
    #         logging.info(_LOG_SEPARATOR)
    #         logging.info(f"THOUGHT: {thought}")
    #         logging.info(_LOG_SEPARATOR)
    #         logging.info(f"ACTION: {action}")
    #         logging.info(_LOG_SEPARATOR)
    #         logging.info(f"ACTION INPUT: {action_input}")
    #         logging.info(_LOG_SEPARATOR)


    #         if action == "Finish":
    #             history = self._format_history()
    #             messages = (
    #                 {"role": "system", "content": REACT_SYSTEM_FINISH_PROMPT},
    #                 {"role": "user", "content": REACT_FINISH_PROMPT.format(task=task, history=history)}
    #             )
    #             logging.info(f"FINISH PROMPT: {REACT_FINISH_PROMPT.format(task=task, history=history)}")
    #             logging.info(_LOG_SEPARATOR)
    #             answer = self._finish(messages)
    #             logging.info(f"FINAL ANSWER: {answer}")
    #             logging.info(_LOG_SEPARATOR)
    #             return answer

    #         if action is None:
    #             self._tool_call_retry_count += 1
    #             observation = (
    #                 f"❌ FORMAT ERROR: Could not extract 'Action:' from your response.\n"
    #                 f"Your raw response excerpt: \"{message.content[:200]}...\"\n\n"
    #                 f"✅ REQUIRED FORMAT (copy exactly):\n"
    #                 f"Thought: <your reasoning>\n"
    #                 f"Action: <tool_name>\n"
    #                 f"Action Input: <single plain text string>\n\n"
    #                 f"⚠️ Available tools: {available_tools}\n"
    #                 f"💡 Tip: Make sure 'Action:' is on its own line, followed by 'Action Input:' on the next line."
    #             )
                
    #             self.memory.append({"role": "user", "content": f"Observation from PARSER: {observation}"})
    #             logging.info(f"PARSE ERROR - Action missing: {observation}")
    #             logging.info(_LOG_SEPARATOR)
    #             continue
            
    #         if action_input is None or not action_input.strip():
    #             self._tool_call_retry_count += 1               
    #             observation = (
    #                 f"❌ ARGUMENT ERROR: Tool '{action}' requires a non-empty string argument.\n"
    #                 f"You provided: '{action_input}' (empty or whitespace)\n\n"
    #                 f"✅ CORRECT EXAMPLE:\n"
    #                 f"Action: {action}\n"
    #                 f"Action Input: <your query>\n\n"
    #                 f"❌ WRONG EXAMPLES:\n"
    #                 f"Action Input: {{\"query\": \"...\"}}  ← NO JSON\n"
    #                 f"Action Input: param1=value1&param2=value2  ← NO key-value pairs\n"
    #                 f"Action Input:   ← NO empty input"
    #             )
    #             self.memory.append({"role": "user", "content": f"Observation from {action}: {observation}"})
    #             logging.info(f"PARSE ERROR - Empty input for tool : {observation}")
    #             logging.info(_LOG_SEPARATOR)
    #             continue
            
    #         if action not in self.tools_dict:
    #             self._tool_call_retry_count += 1
    #             observation = (
    #                 f"❌ TOOL ERROR: Unknown tool name '{action}'.\n"
    #                 f"✅ Available tools (EXACT names, case-sensitive): {available_tools}\n\n"
    #                 f"💡 Reminder: Tool names are CASE-SENSITIVE. 'LLM' ≠ 'llm', 'DB_Search' ≠ 'db_search'\n"
    #                 f"✅ If unsure which tool to use, try: Action: llm\n"
    #                 f"Action Input: <your question about which tool to use>"
    #             )
    #             logging.info(f"Incorrect tool '{action}'. Attempt {self._tool_call_retry_count}/5")
    #         else: 
    #             self._tool_call_retry_count = 0 
    #             observation = self.execute_tool(action, action_input)

    #         self.memory.append(
    #             {
    #                 "role": "user",
    #                 "content": f"Observation from {action}: {observation}",
    #             }
    #         )
            
    #         logging.info(f"OBSERVATION: {observation}")
    #         logging.info(_LOG_SEPARATOR)

    #     logging.info(f"MAX ITERATIONS ({self.max_iterations}) REACHED without solution")
    #     last_message = self.memory[-1].get("content", "") if self.memory else ""
    #     return last_message or "Maximum iterations reached without solution."
