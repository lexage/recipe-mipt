import sqlite3

# Подключаемся к БД
conn = sqlite3.connect('data/github_example.db')
cursor = conn.cursor()

# Получаем список всех таблиц
cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
tables = cursor.fetchall()
print("📊 ТАБЛИЦЫ В БД:")
for table in tables:
    table_name = table[0]
    print(f"   - {table_name}")

print("\n" + "="*60)

# Для каждой таблицы показываем количество записей и пример данных
for table in tables:
    table_name = table[0]
    cursor.execute(f"SELECT COUNT(*) FROM {table_name};")
    count = cursor.fetchone()[0]
    print(f"\n📋 Таблица: {table_name} — {count} записей")
    
    if count > 0:
        # Показываем первые 3 записи
        cursor.execute(f"SELECT * FROM {table_name} LIMIT 3;")
        rows = cursor.fetchall()
        
        # Получаем названия колонок
        cursor.execute(f"PRAGMA table_info({table_name});")
        columns = [col[1] for col in cursor.fetchall()]
        print(f"   Колонки: {columns}")
        
        for i, row in enumerate(rows):
            print(f"   Запись {i+1}:")
            for j, col in enumerate(columns):
                value = row[j]
                if isinstance(value, str) and len(value) > 100:
                    value = value[:100] + "..."
                print(f"      {col}: {value}")
    else:
        print("   (пусто)")

conn.close()