CREATE TABLE IF NOT EXISTS code_examples (
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
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

--2. ТАБЛИЦА ДЛЯ ССЫЛОК
CREATE TABLE IF NOT EXISTS refs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    example_id TEXT,
    reference_url TEXT NOT NULL,
    reference_title TEXT,
    reference_type TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (example_id) REFERENCES code_examples(id) ON DELETE CASCADE
);

-- 3. ТАБЛИЦА ДЛЯ RST ФАЙЛОВ
CREATE TABLE IF NOT EXISTS rst_docs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    file_path TEXT UNIQUE NOT NULL,
    content TEXT NOT NULL,
    parsed_by_llm BOOLEAN DEFAULT FALSE,
    llm_result TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- 4. ТАБЛИЦА ТЕГОВ
CREATE TABLE IF NOT EXISTS tags (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT UNIQUE NOT NULL,
    description TEXT
);

-- 5. СВЯЗЬ ПРИМЕРОВ С ТЕГАМИ
CREATE TABLE IF NOT EXISTS example_tags (
    example_id TEXT,
    tag_id INTEGER,
    FOREIGN KEY (example_id) REFERENCES code_examples(id) ON DELETE CASCADE,
    FOREIGN KEY (tag_id) REFERENCES tags(id) ON DELETE CASCADE,
    PRIMARY KEY (example_id, tag_id)
);

-- 6. ТАБЛИЦА РЕПОЗИТОРИЕВ
CREATE TABLE IF NOT EXISTS repos (
    name TEXT PRIMARY KEY,
    total_examples INTEGER DEFAULT 0,
    last_processed TIMESTAMP,
    readme_content TEXT
);

-- Индексы
CREATE INDEX IF NOT EXISTS idx_type ON code_examples(source_object_type);
CREATE INDEX IF NOT EXISTS idx_repo ON code_examples(repository_name);
CREATE INDEX IF NOT EXISTS idx_path ON code_examples(source_object_path);
CREATE INDEX IF NOT EXISTS idx_refs_url ON refs(reference_url);