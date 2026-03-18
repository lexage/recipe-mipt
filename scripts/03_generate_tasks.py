#!/usr/bin/env python3
"""
Генерация задач для torchmetrics в формате DS-1000
"""

import json
from pathlib import Path
import random
from typing import Dict, List
import re

def load_examples():
    """Загружает примеры из предыдущего шага"""
    examples_file = Path("data/torchmetrics_tasks/raw_examples.json")
    if not examples_file.exists():
        raise FileNotFoundError(f"Сначала запустите 02_extract_from_db.py")
    
    with open(examples_file, 'r', encoding='utf-8') as f:
        return json.load(f)

def create_prompt(example: Dict) -> str:
    """Создаёт промпт для задачи"""
    
    # Определяем тип задачи
    if example['type'] == 'function':
        task_type = "function"
    else:
        task_type = "class"
    
    prompt = f"""# Task: Implement {example['name']} for torchmetrics

## Description
{example['description'] or f"Implement the {example['name']} {task_type} for computing metrics."}

## Категория: {example['category']}
Сложность: {example['difficulty']}

## Сигнатура функции
"""
    
    # Извлекаем сигнатуру
    if 'def' in example['code']:
        sig = example['code'].split('def')[1].split('):')[0] + '):'
        prompt += f"def {example['name']}{sig}\n\n"
    else:
        prompt += f"def {example['name']}(preds, target):\n    pass\n\n"
    
    # Добавляем информацию о параметрах
    if example['parameters']:
        prompt += "## Parameters\n"
        for param, default in example['parameters'].items():
            if default:
                prompt += f"- {param}: default = {default}\n"
            else:
                prompt += f"- {param}: required\n"
        prompt += "\n"
    
    # Добавляем пример использования (если есть)
    if '>>>' in example['code']:
        prompt += "## Example\n"
        example_lines = example['code'].split('\n')
        for line in example_lines:
            if '>>>' in line:
                prompt += line.strip() + "\n"
        prompt += "\n"
    
    prompt += "## Наша реализация:\n```python\n"
    
    return prompt

def create_test_code(example: Dict) -> str:
    """Создаёт тесты для проверки решения"""
    
    func_name = example['name']
    category = example['category']
    
    test_code = []
    
    # Импорты
    test_code.append("import torch")
    test_code.append("import torchmetrics")
    test_code.append("import pytest")
    test_code.append("from solution import " + func_name)
    test_code.append("")
    
    # Тест 1: Базовый случай
    test_code.append(f"def test_{func_name}_basic():")
    
    if category == 'classification':
        test_code.append("    # Binary classification case")
        test_code.append("    preds = torch.tensor([0.9, 0.1, 0.8, 0.2])")
        test_code.append("    target = torch.tensor([1, 0, 1, 0])")
        test_code.append(f"    result = {func_name}(preds, target)")
        test_code.append("    assert isinstance(result, torch.Tensor)")
        test_code.append("    assert 0 <= result <= 1")
        test_code.append("")
        
        test_code.append(f"def test_{func_name}_multiclass():")
        test_code.append("    # Multiclass case")
        test_code.append("    preds = torch.tensor([[0.1, 0.9], [0.8, 0.2], [0.3, 0.7]])")
        test_code.append("    target = torch.tensor([1, 0, 1])")
        if 'num_classes' in str(example['parameters']):
            test_code.append(f"    result = {func_name}(preds, target, num_classes=2)")
        else:
            test_code.append(f"    result = {func_name}(preds, target)")
        test_code.append("    assert isinstance(result, torch.Tensor)")
    
    elif category == 'regression':
        test_code.append("    preds = torch.tensor([3.0, -0.5, 2.0, 7.0])")
        test_code.append("    target = torch.tensor([2.5, 0.0, 2.0, 8.0])")
        test_code.append(f"    result = {func_name}(preds, target)")
        test_code.append("    assert isinstance(result, torch.Tensor)")
    
    else:
        test_code.append("    # Generic test with random data")
        test_code.append("    preds = torch.randn(10, 5)")
        test_code.append("    target = torch.randint(0, 5, (10,))")
        test_code.append(f"    result = {func_name}(preds, target)")
        test_code.append("    assert isinstance(result, torch.Tensor)")
    
    test_code.append("")
    
    # Тест 2: Edge cases
    test_code.append(f"def test_{func_name}_edge_cases():")
    test_code.append("    # Test with identical inputs")
    test_code.append("    preds = torch.ones(10)")
    test_code.append("    target = torch.ones(10)")
    test_code.append(f"    assert {func_name}(preds, target) == 1.0")
    test_code.append("")
    test_code.append("    # Test with completely different inputs")
    test_code.append("    preds = torch.ones(10)")
    test_code.append("    target = torch.zeros(10)")
    test_code.append(f"    assert {func_name}(preds, target) == 0.0")
    test_code.append("")
    
    return '\n'.join(test_code)

def generate_tasks():
   
    
    # Загружаем примеры
    examples = load_examples()
    
    
    # Генерируем задачи
    tasks = []
    for i, ex in enumerate(examples):
        task = {
            "id": f"torchmetrics_{i:05d}",
            "prompt": create_prompt(ex),
            "reference_code": ex['code'],
            "test_code": create_test_code(ex),
            "code_context": {
                "test_string": "assert 'torch.' in solution",  # проверка использования torch
                "exec_context": "torch"
            },
            "metadata": {
                "library": "torchmetrics",
                "category": ex['category'],
                "difficulty": ex['difficulty'],
                "type": ex['type'],
                "name": ex['name'],
                "path": ex['path']
            }
        }
        tasks.append(task)
        
        if (i + 1) % 100 == 0:
            print(f"   Сгенерировано {i + 1} задач")
    
    
    
    # Разделяем на train/test (80/20)
    random.shuffle(tasks)
    split_idx = int(len(tasks) * 0.8)
    train_tasks = tasks[:split_idx]
    test_tasks = tasks[split_idx:]
    
    
    print(f"   Train: {len(train_tasks)} задач (80%)")
    print(f"   Test:  {len(test_tasks)} задач (20%)")
    
    # Сохраняем
    output_dir = Path("data/torchmetrics_tasks")
    
    with open(output_dir / "tasks_all.json", "w", encoding='utf-8') as f:
        json.dump(tasks, f, indent=2, ensure_ascii=False)
    
    with open(output_dir / "tasks_train.json", "w", encoding='utf-8') as f:
        json.dump(train_tasks, f, indent=2, ensure_ascii=False)
    
    with open(output_dir / "tasks_test.json", "w", encoding='utf-8') as f:
        json.dump(test_tasks, f, indent=2, ensure_ascii=False)
    
    
    
    # Статистика по категориям
    from collections import Counter
    categories = Counter(t['metadata']['category'] for t in tasks)
    
    for cat, count in sorted(categories.items(), key=lambda x: -x[1]):
        print(f"   {cat:15s}: {count:4d} задач ({count/len(tasks)*100:.1f}%)")
    
    return tasks

if __name__ == "__main__":
    tasks = generate_tasks()