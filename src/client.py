import os
import json
import asyncio

from dotenv import load_dotenv
from autogen_ext.tools.mcp import SseServerParams, mcp_server_tools
from autogen_agentchat.agents import AssistantAgent
from autogen_ext.models.openai import OpenAIChatCompletionClient

from utils.openai_client import OpenAIClientWrapper


load_dotenv()


async def main() -> None:

    connection_type = os.getenv("CONNECTION")
    llm_api_key = os.getenv("API_KEY")
    llm_name = os.getenv("LLM")

    llm_host = os.getenv("LLM_HOST")
    llm_port = os.getenv("LLM_PORT")
    llm_path = os.getenv("LLM_PATH")
    llm_url = f'http://{llm_host}:{llm_port}{llm_path}'

    vllm_host = os.getenv("VLLM_HOST")
    vllm_port = os.getenv("VLLM_PORT")
    vllm_path = os.getenv("VLLM_PATH")
    vllm_url = f'http://{vllm_host}:{vllm_port}{vllm_path}'

    mcp_host = os.getenv("MCP_HOST")
    mcp_port = os.getenv("MCP_PORT")
    mcp_path = os.getenv("MCP_PATH")
    mcp_server_url = f'http://{mcp_host}:{mcp_port}{mcp_path}'

    server_params = SseServerParams(url=mcp_server_url)
    tools = await mcp_server_tools(server_params)
    print(f"Все доступные инструменты: {tools}")

    if connection_type == 'remote':
        llm_url = llm_url
    elif connection_type == 'local':
        llm_url = vllm_url
    else:
        raise ValueError(f'not implemented connection: {connection_type}')

    model_client = OpenAIChatCompletionClient(
        model=llm_name,
        base_url=llm_url,
        api_key=llm_api_key,
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
