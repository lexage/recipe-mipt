import importlib
import logging
from abc import ABC, abstractmethod
from src.agent_constructor.agent import Agent
import re
from langchain_core.tools import BaseTool
from typing import List
from langchain_core.prompts import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)
from langchain_core.messages import HumanMessage, ToolMessage
from langchain_experimental.utilities import PythonREPL
from typing import Annotated
from langchain_classic.agents import AgentExecutor
from langchain_classic.agents import create_openai_functions_agent
from langchain_experimental.utilities import PythonREPL
from typing import Annotated
from langchain_core.tools import tool
from langchain_core.tools import tool
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
import requests
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder


from .examples import Example, fewshot_examples


def get_examples() -> list[Example]:
    return fewshot_examples


def _load_critic_prompts(dataset: str):
    """Load the CRITIC prompt module for the given dataset (ds1000, codemmlu, ...).

    Each ``prompts_<dataset>`` module exposes the same public names, so switching
    benchmarks changes only which module is imported — selected via the ``dataset`` param.
    """
    return importlib.import_module(f"src.agents.critique.critic.prompts_{dataset}")


class Critic(Agent):
    """
    Agent that implements CRITIC: Large Language Models Can Self-Correct with Tool-Interactive Critiquing.
    Agent finds faults in model response(problem's implementation) using tools.
    Agent has got the following tools:
    1. python_repl_tool - tool for Python code compilation.
    2. compare_tool - tool for comparing result of compilation of current problem's implementation and the expected results of solving the problem
    At the first stage agent uses python_repl_tool to get result of compilation of the current implementation. Then it uses compare_tool to compare the expected results of the solution with the actual compilation output.
    Then agent generates criticism of the implementation using these results.

    """

    def __init__(
        self,
        name: str = "CRITIC",
        model_name: str = "Qwen/Qwen1.5-32B-Chat-AWQ",
        openai_api_base_url="http://localhost:7215/v1",
        dataset: str = "ds1000",
    ):
        super().__init__(name)
        self.dataset = dataset
        self.prompts = _load_critic_prompts(dataset)
        self.refrain = """You are a Python programming assistant.
        You will be given a problem and implementation in Python and critique of implementation.
        Your goal is to make the correct implementation based on the critique
        Return only correct implementation"""
        self.tools = self.get_tools()
        self.llm_model = ChatOpenAI(
            model=model_name,
            openai_api_base=openai_api_base_url,
            openai_api_key="fake-key",
            temperature=0.7,
            # Cap generation so a degenerate no-EOS run can't grow to the full
            # context window (the main "hang" source on vLLM). A 120s timeout with a
            # single retry fails fast instead of blocking ~30 min (600s x 2 retries).
            max_tokens=1024,
            timeout=120.0,
            max_retries=1,
        )

    def llm(self, message) -> str:

        messages = [HumanMessage(content=message)]

        response = self.llm_model.invoke(messages)
        return response.content

    # Wall-clock cap (seconds) for executing LLM-generated option code. Without it,
    # PythonREPL.run() execs in-process with no timeout, so a generated `while True:`,
    # an `input()` waiting on stdin, or a blocking call hangs the whole run forever.
    # Passing a timeout makes PythonREPL run the code in a child process it can kill.
    REPL_TIMEOUT = 10

    def get_python_repl_tool(self) -> BaseTool:
        repl = PythonREPL()
        repl_timeout = self.REPL_TIMEOUT

        @tool(description="Use it to compile Python code")
        def python_repl_tool(
            code: Annotated[str, "The python code to execute."],
        ) -> str:
            """Use this to execute python code. If you want to see the output of a value,
            you should print it out with print(...). This is visible to the user."""
            try:
                result = repl.run(code, timeout=repl_timeout)
            except BaseException as e:
                return f"Failed to execute. Error: {repr(e)}"
            return f"Successfully executed:\n : {result}"

        return python_repl_tool

    def get_compare_tool(self) -> BaseTool:

        @tool(
            description="Use it make compare between the expected result of problem and compilation's result"
        )
        def compare_tool(
            problem: Annotated[str, "Problem."],
            implementation: Annotated[str, "Implementation of problem."],
            compilation_result: Annotated[str, "Results of compilation"],
        ) -> str:

            prompt = self.prompts.COMPARE_PROMPT.format(
                problem=problem,
                implementation=implementation,
                compilation_result=compilation_result,
            )
            return self.llm(prompt)

        return compare_tool

    def get_code(self, answer: str) -> str:
        promt = self.prompts.GET_CODE_PROMPT.format(answer=answer)
        return self.llm(promt)

    def get_tools(self) -> List[BaseTool]:
        # NOTE: the web-search (DuckDuckGo) tool was removed. It had NO enforceable
        # timeout: after a couple hours of queries the IP gets rate-limited and the
        # HTTP call blocks indefinitely. The AgentExecutor's max_execution_time only
        # fires BETWEEN steps, so it cannot interrupt an in-flight tool call — that
        # was the run-freeze source. The MC critic already gathers all its evidence
        # in _gather_evidence_mc (each option executed), so web search is redundant.
        return [
            self.get_python_repl_tool(),
            self.get_compare_tool(),
        ]

    def _gather_evidence(self, problem: str, implementation: str) -> str:
        """Run tools to collect behavioural evidence for the critique step.

        Dispatches on the dataset: code-generation (DS-1000) executes the single
        implementation; multiple-choice (CodeMMLU) executes each option (A/B/C/D) and
        compares them. The DS-1000 path is unchanged from the original ``call_tools``.
        """
        if self.dataset == "codemmlu":
            return self._gather_evidence_mc(problem)
        return self._gather_evidence_code(problem, implementation)

    # Backwards-compatible alias for the original public name.
    def call_tools(self, problem: str, implementation: str) -> str:
        return self._gather_evidence(problem, implementation)

    def _gather_evidence_code(self, problem: str, implementation: str) -> str:
        tool_map = {tool.name: tool for tool in self.tools}
        tool_criticisms = []
        code = self.get_code(implementation)
        compilation_result = tool_map["python_repl_tool"].invoke(code)
        compare_result = tool_map["compare_tool"].invoke(
            {
                "problem": problem,
                "implementation": code,
                "compilation_result": compilation_result,
            }
        )
        tool_criticisms = [f"Result of the implementation: {compilation_result} "]
        tool_criticisms += [
            f"Comparing the expected result of problem and the compilation result of implementation: {compare_result}"
        ]

        return "\n".join(tool_criticisms)

    @staticmethod
    def _parse_options_response(raw: str) -> dict:
        """Parse the GET_OPTIONS_PROMPT output into a ``{letter: code}`` map.

        The prompt emits each option as a ``[[A]]`` / ``[[B]]`` / ... marker followed by its
        runnable code. Options with an empty body (conceptual, no runnable code) map to "".
        """
        options: dict = {}
        # Split keeps the captured letters: ['', 'A', codeA, 'B', codeB, ...]
        parts = re.split(r"\[\[\s*([ABCD])\s*\]\]", raw or "")
        for i in range(1, len(parts) - 1, 2):
            letter = parts[i].strip()
            code = parts[i + 1].strip()
            options[letter] = code
        return options

    def _parse_options(self, problem: str) -> dict:
        raw = self.llm(self.prompts.GET_OPTIONS_PROMPT.format(problem=problem))
        return self._parse_options_response(raw)

    def _gather_evidence_mc(self, problem: str) -> str:
        """CodeMMLU: execute every option, then compare outputs vs expected behaviour.

        Degrades gracefully — options without runnable code, or whose execution errors, are
        recorded as evidence rather than aborting, so a verdict can still be reasoned out.
        """
        tool_map = {tool.name: tool for tool in self.tools}
        repl = tool_map["python_repl_tool"]

        options = self._parse_options(problem)
        exec_results = {}
        for letter in ("A", "B", "C", "D"):
            code = options.get(letter, "")
            if not code:
                exec_results[letter] = "No runnable code (conceptual option or not extracted)."
                continue
            try:
                exec_results[letter] = repl.invoke(code)
            except BaseException as e:  # repl tool already traps most errors; belt-and-braces
                exec_results[letter] = f"Failed to execute. Error: {repr(e)}"

        execution_summary = "\n\n".join(
            f"Option {letter}:\n{result}" for letter, result in exec_results.items()
        )
        compare_result = self.llm(
            self.prompts.MC_COMPARE_PROMPT.format(
                problem=problem, options_execution=execution_summary
            )
        )

        return (
            f"Per-option execution results:\n{execution_summary}\n\n"
            f"Comparison against the expected behaviour:\n{compare_result}"
        )

    def llm_critique(self, problem: str, implementation: str) -> str:
        tools_list = self.get_tools()
        tools_result = self._gather_evidence(problem, implementation)
        prompt = ChatPromptTemplate.from_messages(
            [
                ("system", self.prompts.CRITIQUE_SYSTEM_PROMPT),
                ("human", self.prompts.CRITIQUE_HUMAN_PROMPT),
                ("placeholder", "{agent_scratchpad}"),
            ]
        )

        agent = create_openai_functions_agent(self.llm_model, tools_list, prompt=prompt)
        agent_executor = AgentExecutor(
            agent=agent,
            tools=tools_list,
            verbose=True,
            return_intermediate_steps=True,
            # Bound the tool-calling loop in wall-clock too (not just max_iterations):
            # a slow/blocking tool (e.g. DuckDuckGo rate-limiting) can't stall forever.
            max_execution_time=180,
        )

        # AgentExecutor prints its trace to stdout (verbose=True), NOT to logging, so
        # the per-config log goes silent during the critic stage. Bracket the call
        # with log lines so a stall here is visible in the log instead of looking
        # like a total freeze right after the ReAct FINAL ANSWER.
        logging.info("CRITIC: invoking AgentExecutor")
        response = agent_executor.invoke(
            {
                "problem": problem,
                "tools_result": tools_result,
                "implementation": implementation,
            }
        )
        logging.info(f"CRITIC VERDICT:\n{response['output']}")
        return response["output"]

    def run(self, question: str, response: str) -> str:
        return self.llm_critique(question, response)
