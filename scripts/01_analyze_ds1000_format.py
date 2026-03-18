#!/usr/bin/env python3
"""
Анализ формата DS-1000 датасета
Загрузка и изучение структуры задач
"""

from datasets import load_dataset
import json
from pathlib import Path
from pprint import pprint

def analyze_ds1000():
    
    
    # Загружаем DS-1000 с Hugging Face
    
    ds1000 = list(load_dataset("xlangai/DS-1000")["test"])
    print(f"Загружено {len(ds1000)} задач")
    
    # Анализируем структуру первой задачи
    
    sample = ds1000[0]
    print(f"\nКлючи задачи: {list(sample.keys())}")
    
    
    
    # prompt
    print("\n--- prompt ---")
    print(f"Тип: {type(sample['prompt'])}")
    print(f"Длина: {len(sample['prompt'])} символов")
    print(f"Первые 300 символов:\n{sample['prompt'][:300]}...")
    
    # reference_code
    print("\n--- reference_code ---")
    print(f"Тип: {type(sample['reference_code'])}")
    print(f"Длина: {len(sample['reference_code'])} символов")
    print(f"Содержимое:\n{sample['reference_code'][:300]}...")
    
    # metadata
    print("\n--- metadata ---")
    print(f"Тип: {type(sample['metadata'])}")
    if isinstance(sample['metadata'], dict):
        pprint(sample['metadata'])
    else:
        print(sample['metadata'])
    
    # code_context
    print("\n--- code_context ---")
    print(f"Тип: {type(sample['code_context'])}")
    if isinstance(sample['code_context'], dict):
        print(f"Ключи: {list(sample['code_context'].keys())}")
    else:
        print(f"Значение (первые 200 символов):\n{sample['code_context'][:200]}...")
    
    # Сохраняем пример для дальнейшего использования
    
    output_dir = Path("data/ds1000_analysis")
    output_dir.mkdir(exist_ok=True)
    
    with open(output_dir / "sample_task.json", "w", encoding='utf-8') as f:
        json.dump(sample, f, indent=2, ensure_ascii=False)
    
    
    
    # Статистика по библиотекам
    
    lib_counts = {}
    for task in ds1000:
        if isinstance(task['metadata'], dict):
            lib = task['metadata'].get('library', 'unknown')
        else:
            lib = 'unknown'
        lib_counts[lib] = lib_counts.get(lib, 0) + 1
    
    for lib, count in sorted(lib_counts.items()):
        print(f"   {lib:15s}: {count:4d} задач ({count/len(ds1000)*100:.1f}%)")
    
    return ds1000

if __name__ == "__main__":
    ds1000 = analyze_ds1000()