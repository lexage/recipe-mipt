import os
import json
import asyncio

from dotenv import load_dotenv
from autogen_ext.tools.mcp import SseServerParams, mcp_server_tools
from autogen_agentchat.agents import AssistantAgent
from autogen_ext.models.openai import OpenAIChatCompletionClient


load_dotenv('./recipe-mipt/.env')


async def main() -> None:

    llm_name = os.getenv("LLM")
    llm_url = f'{os.getenv("LLM_URL")}:{os.getenv("LLM_PORT")}/v1'

    vllm_host = os.getenv("VLLM_HOST")
    vllm_port = os.getenv("VLLM_PORT")
    vllm_url = f'http://{vllm_host}:{vllm_port}/v1'

    mcp_path = os.getenv("MCP_PATH")
    mcp_host = os.getenv("MCP_HOST")
    mcp_port = os.getenv("MCP_PORT")
    mcp_server_url = f'http://{mcp_host}:{mcp_port}{mcp_path}'

    server_params = SseServerParams(url=mcp_server_url)
    tools = await mcp_server_tools(server_params)
    print(f"Все доступные инструменты: {tools}")

    model_client = OpenAIChatCompletionClient(
        model=llm_name,
        base_url=llm_url,
        api_key="-",
        model_info={
            "max_tokens": 32768,
            "json_output": False,
            "function_calling": True,
            "vision": False,
            "family": "unknown",
            "structured_output": False,
        },
    )

    agent = AssistantAgent(
        name="tool_user",
        model_client=model_client,
        tools=tools,
        reflect_on_tool_use=False,
        model_client_stream=True,
    )

    # agent = AssistantAgent(
    #     name="tool_user",
    #     model_client=OpenAIChatCompletionClient(
    #         model=model_name,
    #         api_key="-",
    #         base_url=llm_host,
    #         model_info={
    #             "json_output": False,
    #             "function_calling": True,
    #             "vision": False,
    #             "family": "unknown",
    #             "structured_output": False
    #         }
    #     ),
    #     tools=tools
    # )

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
