import sqlite3
import json

# Подключаемся к БД
conn = sqlite3.connect('data/github_example.db')
cursor = conn.cursor()

# Смотрим все таблицы
cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
tables = cursor.fetchall()
print("📊 ТАБЛИЦЫ В БД:")
for table in tables:
    table_name = table[0]
    cursor.execute(f"SELECT COUNT(*) FROM {table_name};")
    count = cursor.fetchone()[0]
    print(f"   - {table_name}: {count} записей")

# Смотрим содержимое таблицы raw_examples (если есть)
if ('raw_examples',) in tables:
    print("\n📝 ПЕРВЫЕ 3 ЗАПИСИ ИЗ raw_examples:")
    cursor.execute("SELECT source_object_name, source_object_path, solution_code FROM raw_examples LIMIT 3;")
    for row in cursor.fetchall():
        print(f"   Имя: {row[0]}")
        print(f"   Путь: {row[1]}")
        print(f"   Код: {row[2][:100]}...")
        print("   ---")

# Смотрим содержимое таблицы chunks (если есть)
if ('chunks',) in tables:
    print("\n📝 ПЕРВЫЕ 3 ЗАПИСИ ИЗ chunks:")
    cursor.execute("SELECT id, text, metadata FROM chunks LIMIT 3;")
    for row in cursor.fetchall():
        print(f"   ID: {row[0]}")
        print(f"   Текст: {row[1][:100]}...")
        if row[2]:
            metadata = json.loads(row[2])
            print(f"   Метаданные: {metadata}")
        print("   ---")

conn.close()