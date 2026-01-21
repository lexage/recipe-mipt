import sqlite3
import argparse
from pathlib import Path


class DocumentationDB:
    def __init__(self, db_path='documentation.db'):
        self.db_path = db_path
        self.conn = None
        
    def connect(self):
        self.conn = sqlite3.connect(self.db_path)
        self.conn.row_factory = sqlite3.Row
        return self.conn
    
    def setup_database(self):
        conn = self.connect()
        cursor = conn.cursor()
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS libraries (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL UNIQUE,
                description TEXT
            )
        ''')
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS sections (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                library_id INTEGER REFERENCES libraries(id),
                name TEXT NOT NULL,
                path TEXT NOT NULL,
                FOREIGN KEY (library_id) REFERENCES libraries(id)
            )
        ''')
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS documents (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                section_id INTEGER REFERENCES sections(id),
                name TEXT NOT NULL,
                content TEXT NOT NULL,
                length INTEGER,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (section_id) REFERENCES sections(id)
            )
        ''')

        conn.execute('''
            CREATE TABLE IF NOT EXISTS examples (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                doc_id INTEGER REFERENCES documents(id),
                order_id INTEGER,
                content TEXT NOT NULL
            )
        ''')
        
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_examples_doc ON examples(doc_id)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_documents_content ON documents(content)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_documents_section ON documents(section_id)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_sections_library ON sections(library_id)')
        
        conn.commit()
        print("База данных успешно создана")
    
    def parse_folder_name(self, folder_name):
        parts = folder_name.split('_')
        if len(parts) >= 2:
            library = parts[0]
            section = '_'.join(parts[1:])
            return library, section
        return None, None
    
    def import_documents(self, results_dir='results'):
        conn = self.connect()
        cursor = conn.cursor()
        
        base_path = Path(results_dir)
        if not base_path.exists():
            print(f"Папка {results_dir} не найдена")
            return
        
        for folder_path in base_path.iterdir():
            if folder_path.is_dir():
                folder_name = folder_path.name
                library_name, section_name = self.parse_folder_name(folder_name)
                
                if not library_name or not section_name:
                    print(f"Не удалось распарсить имя папки: {folder_name}")
                    continue
                
                print(f"Обработка: {library_name} -> {section_name}")
                
                cursor.execute(
                    'INSERT OR IGNORE INTO libraries (name) VALUES (?)',
                    (library_name,)
                )
                
                cursor.execute('SELECT id FROM libraries WHERE name = ?', (library_name,))
                library_id = cursor.fetchone()[0]
                
                cursor.execute(
                    '''INSERT OR IGNORE INTO sections 
                    (library_id, name, path) VALUES (?, ?, ?)''',
                    (library_id, section_name, str(folder_path))
                )
                
                cursor.execute(
                    'SELECT id FROM sections WHERE library_id = ? AND name = ?',
                    (library_id, section_name)
                )
                section_id = cursor.fetchone()[0]
                
                self._import_files_from_folder(cursor, section_id, folder_path)
        
        conn.commit()
        print("Импорт документов завершен")
    
    def _import_files_from_folder(self, cursor, section_id, folder_path):
        for doc_folder in folder_path.iterdir():
            if not doc_folder.is_dir():
                continue
            
            try:
                doc_name = doc_folder.name
                main_file_path = doc_folder / f"{doc_name}.txt"
                
                if not main_file_path.exists():
                    print(f"  Основной файл не найден: {main_file_path}")
                    continue
                
                with open(main_file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                cursor.execute('''
                    INSERT OR REPLACE INTO documents 
                    (section_id, name, content, length)
                    VALUES (?, ?, ?, ?)
                ''', (
                    section_id,
                    doc_name,
                    content,
                    len(content)
                ))
                
                doc_id = cursor.lastrowid
                if doc_id == 0:
                    cursor.execute('SELECT id FROM documents WHERE file_path = ?', (str(main_file_path),))
                    doc_id = cursor.fetchone()[0]
                
                print(f"  Добавлен документ: {main_file_path.name}")
                
                example_count = self._import_examples(cursor, doc_id, doc_folder, doc_name)
                if example_count > 0:
                    print(f"    Добавлено примеров: {example_count}")
                
            except Exception as e:
                print(f"  Ошибка при обработке {doc_folder}: {e}")
    
    def _import_examples(self, cursor, doc_id, doc_folder, document_name):
        example_count = 0
        
        for example_file in doc_folder.glob('example_*.txt'):
            try:
                with open(example_file, 'r', encoding='utf-8') as f:
                    example_content = f.read()
                order_id = int(example_file.stem.split('_')[1])
                cursor.execute('''
                    INSERT INTO examples (doc_id, content, order_id)
                    VALUES (?, ?, ?)
                ''', (doc_id, example_content, order_id))
                
                example_count += 1
                
            except Exception as e:
                print(f"    Ошибка при обработке примера {example_file}: {e}")
        
        return example_count
    
    def get_stats(self):
        conn = self.connect()
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT 
                l.name as library, 
                s.name as section, 
                COUNT(DISTINCT d.id) as doc_count,
                COUNT(e.id) as example_count
            FROM libraries l
            LEFT JOIN sections s ON l.id = s.library_id
            LEFT JOIN documents d ON s.id = d.section_id
            LEFT JOIN examples e ON d.id = e.doc_id
            GROUP BY l.name, s.name
            ORDER BY l.name, s.name
        ''')
        
        stats = cursor.fetchall()
        print("\nСтатистика базы данных:")
        print("-" * 70)
        print(f"{'Библиотека':<15} {'Раздел':<25} {'Документы':<12} {'Примеры':<10}")
        print("-" * 70)
        for row in stats:
            print(f"{row['library']:<15} {row['section']:<25} {row['doc_count']:<12} {row['example_count']:<10}")
        
        cursor.execute('SELECT COUNT(*) as total FROM documents')
        total_docs = cursor.fetchone()['total']
        
        cursor.execute('SELECT COUNT(*) as total FROM examples')
        total_examples = cursor.fetchone()['total']
        
        print("-" * 70)
        print(f"\nВсего документов: {total_docs}")
        print(f"Всего примеров: {total_examples}")
    
    def close(self):
        if self.conn:
            self.conn.close()

def main():
    parser = argparse.ArgumentParser(
        description='Создание базы данных из спаршенных документов'
    )
    parser.add_argument(
        '--db',
        type=str,
        default='docs_database.db',
        help='Путь к файлу базы данных (по умолчанию: docs_database.db)'
    )
    parser.add_argument(
        '--docs',
        type=str,
        default='results',
        help='Путь к папке со спаршенными документами (по умолчанию: results)'
    )
    
    args = parser.parse_args()
    
    db = DocumentationDB(args.db)
    
    try:
        db.setup_database()
        
        db.import_documents(args.docs)
        
        db.get_stats()
        
    finally:
        db.close()

if __name__ == "__main__":
    main()