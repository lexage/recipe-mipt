-- 1. Таблица для хранения чанков (для поиска)
CREATE TABLE IF NOT EXISTS chunks (
    id TEXT PRIMARY KEY,
    text TEXT NOT NULL,
    metadata TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- 2. Таблица для хранения сырых примеров
CREATE TABLE IF NOT EXISTS raw_examples (
    id TEXT PRIMARY KEY,
    source_object_type TEXT NOT NULL,
    source_object_name TEXT NOT NULL,
    source_object_path TEXT NOT NULL,
    task_description TEXT,
    solution_code TEXT NOT NULL,
    metadata_source_code TEXT,
    references TEXT,
    repository_name TEXT NOT NULL,
    chunk_index INTEGER DEFAULT 0,
    total_chunks INTEGER DEFAULT 1,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Индексы для ускорения поиска
CREATE INDEX IF NOT EXISTS idx_chunks_text ON chunks(text);
CREATE INDEX IF NOT EXISTS idx_raw_type ON raw_examples(source_object_type);
CREATE INDEX IF NOT EXISTS idx_raw_repo ON raw_examples(repository_name);
CREATE INDEX IF NOT EXISTS idx_raw_path ON raw_examples(source_object_path);