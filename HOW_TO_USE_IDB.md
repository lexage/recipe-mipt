# HOW TO USE IDB - Интерфейс базы данных

## Что такое IDB?

**IDB** — это интерфейс для работы с SQLite базой данных, который сохраняет все примеры, извлеченные парсером, и позволяет их искать, фильтровать и анализировать.

---

## Быстрый старт

### 1. Импорт
```python
from src.utils.github_parser.idb import IDB, DS1000Wrapper

# После того как собрали dataset через pipeline.run()
db = IDB("github_examples.db")
uploaded = db.add_extracted_examples_batch(dataset, "Lightning-AI/torchmetrics")
print(f"Сохранено {uploaded} примеров")

# Поиск по тексту
results = db.search("accuracy", limit=10)
for r in results:
    print(f"{r['source_object_name']}: {r['task_description'][:50]}")

# Поиск по фрагменту кода
code_results = db.search_by_code("return (y_true == y_pred).mean()")

stats = db.get_stats()
print(f"Всего примеров: {stats['total_examples']}")
print(f"По типам: {stats['by_type']}")
print(f".rst файлов: {stats['rst_documents']}")
print(f"Ссылок: {stats['references']}")
print(f"Размер БД: {stats['database_size_mb']:.2f} MB")

# Сохранить .rst файл
db.save_rst_document("docs/source/quickstart.rst", content)

# Получить необработанные .rst
rst_files = db.get_unprocessed_rst_documents()
for rst in rst_files:
    # Отправить в LLM
    llm_result = call_llm(rst['content'])
    # Сохранить результат
    db.update_rst_document_with_llm_result(rst['id'], llm_result)