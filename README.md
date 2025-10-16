## Структура проекта

```
recipe-mipt/
├── data/ # for small data and tests
├── dtools/
│ ├── Dockerfile
│ ├── docker_compose.yaml
│ ├── build.sh # build agents and services
│ ├── run.sh # run agents and services
│ └── README.md # description
├── services/... # vLLM, DB, ..., etc.
├── src/
│ ├── agents/
│ │ ├── pipelines
│ │ │ ├── maps.py
│ │ │ ├── mars.py
│ │ │ ├── custom.py
│ │ │ ├── ...
│ │ ├── react.py
│ │ ├── plan_and_execute.py
│ │ ├── ...
│ │ ...
│ ├── agent_constructor/
│ │ ├── ...
│ │ ...
│ ├── icl/ # iCL modules
│ │ ├── ...
│ │ ...
│ ├── rag/ # RAG modules
│ │ ├── ...
│ │ ...
│ ├── mcp/ # MCP servers
│ │ ├── server.py
│ │ ├── server_v2.py
│ │ ├── ...
│ │ ...
│ ├── tools/
│ │ ├── search_tool.py
│ │ ...
│ ├── utils/ # dataclasses, helper functions
│ │ ├── ...
│ │ ...
│ ├── client.py # MCP client
├── .env
├── .gitignore
├── pyproject.toml
├── README.md
├── requirements.txt
```
