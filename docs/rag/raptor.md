# RAPTOR Retriever

- **Модуль**: [`RaptorRetriever`](../../src/rag/raptor/retriever.py)
- **Оригинальная статья**: [RAPTOR: Recursive Abstractive Processing for Tree-Organized Retrieval](https://arxiv.org/pdf/2401.18059)
- **Оригинальная реализация**: [GitHub](https://github.com/parthsarthi03/raptor)
- **Конспект**: [Google Docs](https://docs.google.com/document/d/1mRlQQphQpDIEkH37_yggol5F05q-n-plUkS4zGvQges/edit?tab=t.0#heading=h.uge8kpe48cbz)

---

## Принцип работы

Строит иерархическое дерево из документов, используя кластеризацию и суммаризацию. Документы группируются в кластеры, которые затем суммируются, создавая многоуровневую структуру. При поиске релевантная информация извлекается из дерева, что позволяет находить информацию на разных уровнях абстракции.

---

## Параметры инициализации

- `data_base: IDB` — база данных для поиска
- `path_to_raptor_db: str` — путь для сохранения/загрузки дерева
- `embedding_agent: Agent` — агент для создания эмбеддингов
- `qa_agent: Agent` — агент для генерации ответов на вопросы
- `summarization_agent: Agent` — агент для суммаризации текста

---

## Алгоритм

При инициализации `RaptorRetriever` выполняются следующие шаги:

1. Создается конфигурация `RetrievalAugmentationConfig` с обертками для агентов (эмбеддинги, QA, суммаризация).
2. Пытается загрузить существующее дерево из `path_to_raptor_db`.
3. Если дерево не найдено, создается новое:
   - Все чанки из `data_base` объединяются в один текст.
   - Строится дерево с помощью кластеризации и суммаризации.
   - Дерево сохраняется в `path_to_raptor_db`.

Для поиска вызывается метод `RaptorRetriever.retrieve(query: str, top_k: int = 5, collapsed: bool = True)`, который:

1. Использует внутренний `TreeRetriever` для поиска релевантной информации в дереве.
2. Возвращает контекст, извлеченный из дерева на основе запроса.

---

## Дополнительные модули

Для генерации ответов на вопросы можно использовать [`RaptorQAAgent`](../../src/agents/rag/raptor_agents.py).

Для суммаризации текста можно использовать [`RaptorSummarizationAgent`](../../src/agents/rag/raptor_agents.py).
