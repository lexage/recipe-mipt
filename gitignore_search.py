from typing import List

from src.agent_constructor.db import IDB
from src.agent_constructor.core import Document, Chunk

from src.agents.general.embedding_agents import EmbeddingAgent
from src.agent_constructor.context_engine import SimpleContextAssembler
from src.agent_constructor.chunkers import DummyChunker

# Импорт класса GitHubDocsDB: 
from src.utils.github_parser.github_parser_db.github_docs_db import GitHubDocsDB

# Что бы был коннект к url нужно настроить ssh конфиг и пробросить порты
# Смотри: docs/services/remote_vllm.md пунткы 1 и 4. 
embedder=EmbeddingAgent(
        url = "http://localhost:7216/v1",
        model_name = "Qwen/Qwen3-Embedding-4B",
    ),


# Твоя реализация класса БД
db : IDB = GitHubDocsDB(
    embedder=embedder,
    db_path="data/github_example.db", # путь к уже созданой SQLite БД-шке
    vector_db_path="data/github_vector_database", 
    collection_name="docs",
)

# Доп компоненты, которые используются в пайплайне
chuker = DummyChunker()
assembler = SimpleContextAssembler(name="sad")

# Получаем документы из БД
documents : List[Document] = db.get_documents()

# Применяем чанкинг
chunks : List[Chunk] = []
for doc in documents:
    chunks.extend(chuker.chunk(doc))

# Загружаем чанки в IDB, векторизуем и создаем векторную БД
db.add_chunks(chunks)

# Пробуем найти чанки по запросу
result : List[Chunk] = db.query(query_text="How to merge numpy array?", top_k=5)

# Собираем контекст
context = assembler.assemble(result)

print(context)
