# InstructRAG Retriever

- **Модуль**: [`InstructRAGRetriever`](../../src/rag/instructrag/retriever.py)
- **Оригинальная статья**: [InstructRAG: Instructing Retrieval-Augmented Generation via Self-Synthesized Rationales](https://arxiv.org/pdf/2406.13629)
- **Оригинальная реализация**: [GitHub](https://github.com/weizhepei/InstructRAG)
- **Конспект**: [Google Docs](https://docs.google.com/document/d/1mRlQQphQpDIEkH37_yggol5F05q-n-plUkS4zGvQges/edit?tab=t.0#heading=h.mpbmr0bdy1rg)

---

## Принцип работы

Для каждого полученного чанка генерируются рационали, т.е. оценка релевантонсти/полезности или критика содержимого чанка. 

---

## Параметры инициализации

- `name: str` — название блока
- `data_base: IDB` — база данных для поиска
- `rationality_agent: Agent` — агент, генерирующий рационали

---

## Алгоритм

Для поиска вызывается метод `InstructRAGRetriever.retrieve(query: str, k: int = 1)`, который выполняет следующие шаги:

1. Ищет релевантные чанки в `data_base: IDB`.
2. `rationality_agent: Agent` генерирует рационалии для найденых чанков.
3. Ретривер возвращает кортеж изи списка рационалий и списка чанков.

---

## Дополнительные модули

Для сборки контекста можно использовать [`InstructRAGContextAssembler`](../../src/agent_constructor/context_engine.py).
