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

- `name: str, ` — название блока
- `url: str` — url для обращения к модели генерации промежуточных шагов
- `data_base: IDB, ` — база данных для поиска
- `search_type: CoRAGSearchTypes` — тип поиска
    - `CoRAGSearchTypes.SAMPLE_SEARCH` — простой линейный поиск
    - `CoRAGSearchTypes.TREE_SEARCH` — поиск по дереву
    - `CoRAGSearchTypes.BEST_OF_N_SEARCH` — поиск наилучшего простого пути из N
- `return_docs_as_chunks: bool` — если `True`, то в результате работы будут возвращаться чанки с полным текстом используемых документов. 
- `max_path_length: int` — максимальная длина цепочки подзапросов
- `max_message_length: int` — максимальная длина сообщений
- `temperature: float` — температура при генерации подзапросов и ответов
- `task_description: str` — описание задачи для генерации подзапросов (не путать с query)
- `expand_size: int` — кол-во расширений состояние при `TREE_SEARCH`
- `num_rollouts: int` — кол-во независимых продолжений цепочки для каждого кандидата при `TREE_SEARCH`
- `beam_size: int` — размер луча при `BEST_OF_N_SEARCH`
- `n: int` — число кандидатов для `BEST_OF_N_SEARCH`

---

## Алгоритм

Для поиска вызывается метод `CoRAGRetriever.retrieve(query: str, k: int = 1)`, который выполняет следующие шаги:

1. Для поиска используется указанная при инициализации стратегия. В самом простом случае. 
2. На основе запроса генерируется подзапрос. 
3. Получается контекст из `data_base: IDB` для подзапроса.
4. LLM отвечает на подзапрос. Результаты сохраняются.
5. Шаги 2–4 повторяются `max_path_length: int` раз.
5. Ретривер возвращает список чанков. Один чанк, у которого `id="corag_intermediate_steps"` содержит в себе подзапросы и ответы на них. Остальные чанки - получены из БД. 

---

## Дополнительные модули

Для сборки контекста можно использовать [`CoRAGContextAssembler`](../../src/agent_constructor/context_engine.py).

Для генерации финального ответа можно использовать [`CoRAGFinalSolver`](../../src/agents/rag/corag_agents.py).
