#!/usr/bin/env python3
"""
Валидация сгенерированных задач
"""

import json
from pathlib import Path

def validate_tasks():
    
    
    tasks_file = Path("data/torchmetrics_tasks/tasks_all.json")
    if not tasks_file.exists():
        
        return
    
    with open(tasks_file, 'r', encoding='utf-8') as f:
        tasks = json.load(f)
    
    
    
    errors = []
    required_fields = ['id', 'prompt', 'reference_code', 'test_code', 'code_context', 'metadata']
    
    for i, task in enumerate(tasks):
        # Проверка обязательных полей
        for field in required_fields:
            if field not in task:
                errors.append(f"Задача {task['id']}: отсутствует поле {field}")
        
        # Проверка метаданных
        if 'metadata' in task:
            for meta_field in ['library', 'category', 'difficulty']:
                if meta_field not in task['metadata']:
                    errors.append(f"Задача {task['id']}: отсутствует metadata.{meta_field}")
        
        # Проверка длины
        if len(task['prompt']) < 50:
            errors.append(f"Задача {task['id']}: слишком короткий промпт ({len(task['prompt'])} символов)")
        
        if len(task['reference_code']) < 10:
            errors.append(f"Задача {task['id']}: слишком короткий код")
        
        if (i + 1) % 500 == 0:
            print(f"   Проверено {i + 1} задач")
    
    # Вывод результатов
   
    if errors:
        
        for err in errors[:10]:  # показываем первые 10
            print(f"      • {err}")
        if len(errors) > 10:
            print(f"и ещё {len(errors) - 10} ошибок")
    else:
        print(f"Все задачи валидны!")
    
    # Статистика
    
    print(f"   Всего задач: {len(tasks)}")
    
    libs = set(t['metadata']['library'] for t in tasks if 'metadata' in t)
    print(f"   Библиотеки: {', '.join(libs)}")
    
    cats = set(t['metadata']['category'] for t in tasks if 'metadata' in t)
    print(f"   Категории: {', '.join(cats)}")
    
    return len(errors) == 0

if __name__ == "__main__":
    validate_tasks()