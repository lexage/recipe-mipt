#!/usr/bin/env python3
"""
Извлечение примеров из вашей БД с torchmetrics
Читает данные из таблицы chunks
"""

import sqlite3
import json
from pathlib import Path
from collections import Counter
import re

def extract_from_db(db_path="data/github_example.db"):
    
    
    # Подключаемся к БД
    
    if not Path(db_path).exists():
        
        return []
    
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # Смотрим, какие таблицы есть
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
    tables = cursor.fetchall()
    print(f"   Таблицы в БД: {[t[0] for t in tables]}")
    
    # Читаем из таблицы chunks
    if ('chunks',) not in tables:
                return []
    
    cursor.execute("""
    SELECT 
        id,
        text,
        metadata
    FROM chunks
    WHERE text IS NOT NULL 
    AND LENGTH(text) > 30
    """)
    
    rows = cursor.fetchall()
    
    
    if len(rows) == 0:
       
        return []
    
    # Преобразуем в список словарей
    examples = []
    categories = Counter()
    
    for row in rows:
        chunk_id = row[0]
        code = row[1]
        metadata_str = row[2]
        
        # Парсим metadata (это JSON строка)
        try:
            metadata = json.loads(metadata_str) if metadata_str else {}
        except:
            metadata = {}
        
        example = {
            'id': chunk_id,
            'name': metadata.get('name', 'unknown'),
            'path': metadata.get('path', 'unknown'),
            'code': code,
            'description': metadata.get('description', ''),
            'type': metadata.get('type', 'unknown'),
            'repo': metadata.get('repo', 'torchmetrics')
        }
        
        # Определяем категорию
        category = detect_category(example['path'])
        example['category'] = category
        categories[category] += 1
        
        # Определяем сложность
        code_lines = example['code'].count('\n')
        if code_lines < 5:
            example['difficulty'] = 'easy'
        elif code_lines < 15:
            example['difficulty'] = 'medium'
        else:
            example['difficulty'] = 'hard'
        
        # Извлекаем параметры
        example['parameters'] = extract_parameters(example['code'])
        
        # Извлекаем импорты
        example['imports'] = extract_imports(example['code'])
        
        examples.append(example)
    
    conn.close()
    
    # Выводим статистику
    
    print("\n   Категории:")
    for cat, count in sorted(categories.items(), key=lambda x: -x[1]):
        print(f"     {cat:15s}: {count:4d}")
    
    # Сохраняем
    output_file = Path("data/torchmetrics_tasks/raw_examples.json")
    output_file.parent.mkdir(exist_ok=True)
    
    with open(output_file, "w", encoding='utf-8') as f:
        json.dump(examples, f, indent=2, ensure_ascii=False)
    
    
    
    return examples

def detect_category(path):
    """Определяет категорию метрики по пути"""
    path_lower = path.lower()
    
    categories = {
        'classification': ['classification', 'accuracy', 'precision', 'recall', 'f1', 'auroc', 'roc'],
        'regression': ['regression', 'mse', 'mae', 'r2', 'rmse', 'mape'],
        'text': ['text', 'bleu', 'rouge', 'cer', 'wer', 'chrf', 'bert'],
        'image': ['image', 'psnr', 'ssim', 'lpips'],
        'video': ['video', 'vmaf'],
        'audio': ['audio', 'pesq', 'stoi', 'sdr'],
        'pairwise': ['pairwise', 'cosine', 'euclidean', 'manhattan'],
        'aggregation': ['aggregation', 'mean', 'sum', 'cat']
    }
    
    for category, keywords in categories.items():
        if any(keyword in path_lower for keyword in keywords):
            return category
    
    return 'other'

def extract_parameters(code):
    """Извлекает параметры из кода"""
    params = {}
    
    # Ищем сигнатуру функции
    func_match = re.search(r'def\s+\w+\s*\((.*?)\)', code)
    if func_match:
        param_str = func_match.group(1)
        for param in param_str.split(','):
            param = param.strip()
            if '=' in param:
                name, default = param.split('=', 1)
                params[name.strip()] = default.strip()
            elif param and param not in ['self', 'cls']:
                params[param] = None
    
    return params

def extract_imports(code):
    """Извлекает импорты из кода"""
    imports = []
    import_lines = re.findall(r'^import .+$|^from .+ import .+$', code, re.MULTILINE)
    
    for line in import_lines:
        imports.append(line.strip())
    
    return imports[:5]

if __name__ == "__main__":
    examples = extract_from_db()