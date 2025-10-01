import os
import asyncio
import json

from autogen_ext.tools.mcp import SseServerParams, mcp_server_tools
from autogen_agentchat.agents import AssistantAgent
from autogen_ext.models.openai import OpenAIChatCompletionClient


async def main() -> None:
    llm_host = 'http://0.0.0.0:11455/v1'
    mcp_server_url = 'http://127.0.0.1:8002/mcp'
    model_name = "Qwen/Qwen3-30B-A3B-FP8"
    server_params = SseServerParams(url=mcp_server_url)
    tools = await mcp_server_tools(server_params)

    print("Все доступные инструменты:")
    print(tools)

    agent = AssistantAgent(
        name="tool_user",
        model_client=OpenAIChatCompletionClient(
            model=model_name,
            api_key="-",
            base_url=llm_host,
            model_info={
                "json_output": False,
                "function_calling": True,
                "vision": False,
                "family": "unknown",
                "structured_output": False
            }
        ),
        tools=tools
    )

    while True:
        result = await agent.run(task=input('user: '))
        print("Вызванный инструмент:")
        print(result.messages[-1])
        print()
        print("После парсинга:")
        print(result.messages[-1].content)
        print(result.messages[-1].type)


if __name__ == "__main__":
    asyncio.run(main())
