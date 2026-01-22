# CoRAG Retriever

- **Модуль**: [`CoRAGRetriever`](../../src/rag/corag/retriever.py)
- **Оригинальная статья**: [Chain-of-Retrieval Augmented Generation](https://arxiv.org/pdf/2501.14342)
- **Оригинальная реализация**: [GitHub](https://github.com/microsoft/LMOps/tree/main/corag)
- **Конспект**: [Google Docs](https://docs.google.com/document/d/1mRlQQphQpDIEkH37_yggol5F05q-n-plUkS4zGvQges/edit?tab=t.0#heading=h.9deid3lqlwzx)

---

## Принцип работы

Итеративно разбивает запрос на подзапросы, создавая цепочку "подзапрос — ответ". Каждый подзапрос генерируется на основе ответов на предыдущие подзапросы.

---

## Параметры инициализации

- `name: str` — название блока
- `data_base: IDB` — база данных для поиска
- `generator: Agent` — агент, генерирующий подзапросы
- `sub_solver: Agent` — агент, отвечающий на подзапросы
- `max_sub_queries: int` — максимальное количество подзапросов

---

## Алгоритм

Для поиска вызывается метод `CoRAGRetriever.retrieve(query: str, k: int = 1)`, который выполняет следующие шаги:

1. С помощью `generator: Agent` генерируется подзапрос на основе оригинального запроса и ответов на предыдущие подзапросы (на первой итерации — только на основе запроса).
2. Получается контекст из `data_base: IDB` для подзапроса.
3. Используя контекст, `sub_solver: Agent` отвечает на подзапрос. Результаты сохраняются.
4. Шаги 1–3 повторяются `max_sub_queries: int` раз.
5. Ретривер возвращает список пар "подзапрос-ответ" и список использованных чанков.

---

## Дополнительные модули

Для сборки контекста можно использовать [`CoRAGContextAssembler`](../../src/agent_constructor/context_engine.py).

Для генерации подзапросов можно использовать [`CoRAGSubQueryGeneratorAgent`](../../src/agents/rag/corag_agents.py).

Для генерации ответов на подзапросы можно использовать [`CoRAGSubSolver`](../../src/agents/rag/corag_agents.py).

Для генерации финального ответа можно использовать [`CoRAGFinalSolver`](../../src/agents/rag/corag_agents.py).
