## Структура проекта
```
agent-constructor/
├── README.md # overview + usage
├── pyproject.toml # project config
├── src/
│ ├── agent_constructor/
│ │ ├── __init__.py
│ │ ├── core.py # base classes, interfaces (this file)
│ │ ├── db.py # Database implementations + storage adapters
│ │ ├── chunkers.py # chunking strategies
│ │ ├── filters.py # filters (source-specific and generic)
│ │ ├── augmenters.py # augmentation strategies
│ │ ├── context_engine.py # RAG / iCL / reasoning context building
│ │ ├── pipeline.py # Planner / Critic / Student and orchestration
│ │ └── examples.py # small usage examples and recipes
├── tests/
│ └── test_core.py
└── docs/
└── design.md
```