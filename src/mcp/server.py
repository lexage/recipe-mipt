import os
import argparse
from typing import Annotated

from dotenv import load_dotenv
from pydantic import Field
from fastmcp import FastMCP


load_dotenv()
server = FastMCP("agent_tools")


@server.tool(description="""Простой калькулятор для выполнения
             математических вычислений. Принимает строку с выражением
             (например, '5 * (10 + 2)') и возвращает результат.
             Поддерживаемые операции: +, -, *, /, //, math.sqrt""")
def calculator(
    expression: Annotated[str, Field(description="Математическое выражение для вычисления")]
) -> str:
    try:
        # Using a safer eval
        allowed_chars = "0123456789+-*/(). "
        if all(char in allowed_chars for char in expression):
            return str(eval(expression))
        else:
            return "Error: Invalid characters in expression."
    except Exception as e:
        return f"Error: {e}"


@server.tool(description="""Данный инструмент позволяет узнать
             актуальную погоду в Москве.""")
def weather(
    expression: Annotated[str, Field(description="время суток")]
) -> str:
    try:
        return "The weather is GOOD!" 
    except Exception as e:
        return f"Error: {e}"


parser = argparse.ArgumentParser()
parser.add_argument("--transport", type=str, required=False, default="sse")


if __name__ == "__main__":
    args = parser.parse_args()
    transport = args.transport

    if transport == "stdio":
        server.run()
    else:
        server.run(
            transport="sse",
            host=os.getenv("MCP_HOST"),
            port=int(os.getenv("MCP_PORT")),
            path=os.getenv("MCP_PATH"),
            log_level="debug",
        )
