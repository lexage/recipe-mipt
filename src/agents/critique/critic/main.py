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
from langchain_community.tools import DuckDuckGoSearchResults
from langchain_community.utilities import DuckDuckGoSearchAPIWrapper
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


class Critic(Agent):
    """
    Agent that implements CRITIC: Large Language Models Can Self-Correct with Tool-Interactive Critiquing.
    Agent finds faults in model response(problem's implementation) using tools.
    Agent has got the following tools:
    1. python_repl_tool - tool for Python code compilation.
    2. compare_tool - tool for comparing result of compilation of current problem's implementation and the expected results of solving the problem
    3. web_search_tool - tool for searching information using Internet
    At the first stage agent uses python_repl_tool to get result of compilation of the current implementation. Then it uses compare_tool to compare the expected results of the solution with the actual compilation output.
    Then agent generates criticism of the implementation using these results.

    """

    def __init__(
        self,
        name: str = "CRITIC",
        model_name: str = "Qwen/Qwen1.5-32B-Chat-AWQ",
        openai_api_base_url="http://localhost:7215/v1",
    ):
        super().__init__(name)
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
        )

    def llm(self, message) -> str:

        messages = [HumanMessage(content=message)]

        response = self.llm_model.invoke(messages)
        return response.content

    def get_python_repl_tool(self) -> BaseTool:
        repl = PythonREPL()

        @tool(description="Use it to compile Python code")
        def python_repl_tool(
            code: Annotated[str, "The python code to execute."],
        ) -> str:
            """Use this to execute python code. If you want to see the output of a value,
            you should print it out with print(...). This is visible to the user."""
            try:
                result = repl.run(code)
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

            prompt = f"You will get a problem, implementation and compilation result of this implementation. You should compare the expected result of problem and compilation's result. Return only result of comparison. Problem: {problem} \n Implementation: {implementation} \n compilation_result: {compilation_result}  "
            return self.llm(prompt)

        return compare_tool

    def get_web_search_tool(self, max_results: int = 1) -> BaseTool:
        wrapper = DuckDuckGoSearchAPIWrapper(
            max_results=max_results,
        )
        tool = DuckDuckGoSearchResults(api_wrapper=wrapper)
        tool.description = (
            """Search information using Internet. Use English language for searching."""
        )
        return tool

    def get_code(self, answer: str) -> str:
        promt = f"You only need to extract the Python code from the following text. I want to compile this code. And You can't fix it. If you can't find python code, return nothing. Text: {answer} "
        return self.llm(promt)

    def get_tools(self) -> List[BaseTool]:
        return [
            self.get_web_search_tool(),
            self.get_python_repl_tool(),
            self.get_compare_tool(),
        ]

    def call_tools(self, problem: str, implementation: str):
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

    def llm_critique(self, problem: str, implementation: str) -> str:
        tools_list = self.get_tools()
        tools_result = self.call_tools(problem, implementation)
        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    """
            You will be given a problem and Implementation in Python and information about Implementation. 
            You should generate criticism of this Implementation using the information:
            It is important for you to use following criteria:
            1. Evaluation of implementation
            2. Logical errors in reasoning 
            3. Syntax and semantic correctness 
            4. Conceptual misunderstandings 
            5. Potential bugs or edge cases 
            6. Alignment with problem requirements You will need this as a hint when you 
             Return only Criticism, not implementation""",
                ),
                (
                    "human",
                    "Problem: {problem} \n Implementation: {implementation} \n Information: {tools_result}  . ",
                ),
                ("placeholder", "{agent_scratchpad}"),
            ]
        )

        agent = create_openai_functions_agent(self.llm_model, tools_list, prompt=prompt)
        agent_executor = AgentExecutor(
            agent=agent, tools=tools_list, verbose=True, return_intermediate_steps=True
        )

        response = agent_executor.invoke(
            {
                "problem": problem,
                "tools_result": tools_result,
                "implementation": implementation,
            }
        )
        return response["output"]

    def run(self, question: str, response: str) -> str:
        return self.llm_critique(question, response)
