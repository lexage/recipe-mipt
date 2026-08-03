import json
import requests
import time
from pathlib import Path

# Используем нашу модель через SSH туннель
MODEL_URL = "http://localhost:7215/v1/chat/completions"

def call_llm(prompt):
    """Вызов модели через локальный туннель"""
    headers = {"Content-Type": "application/json"}
    data = {
        "model": "Qwen/Qwen1.5-32B-Chat-AWQ",
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.7
    }
    
    try:
        response = requests.post(MODEL_URL, json=data, headers=headers)
        return response.json()["choices"][0]["message"]["content"]
    except Exception as e:
        print(f"Ошибка вызова модели: {e}")
        return None

# Загружаем базовые задачи
with open("data/torchmetrics_benchmark/core_tasks.json") as f:
    core_tasks = json.load(f)

print(f"Загружено {len(core_tasks)} базовых задач")

def expand_task(task, variation_type):
    prompts = {
        "surface": f"""Create a surface perturbation of this coding task.
Change only the function name, variable names, and wording.
Keep the exact same logic and difficulty.

Original code:
{task['code']}

Return ONLY the new code (without explanation):""",

        "semantic": f"""Create a semantic perturbation of this coding task.
Change the meaning while keeping similar difficulty.
For example: accuracy → error rate, MSE → RMSE.

Original code:
{task['code']}

Return ONLY the new code:""",

        "difficult": f"""Create a more difficult version of this coding task.
Add complexity like class weights or multi-class support.

Original code:
{task['code']}

Return ONLY the new code:"""
    }
    
    return call_llm(prompts[variation_type])

# Генерируем вариации
surface_tasks = []
semantic_tasks = []
difficult_tasks = []

for i, task in enumerate(core_tasks):
    print(f"Обработка {i+1}/{len(core_tasks)}: {task['name']}")
    
    surface_code = expand_task(task, "surface")
    if surface_code:
        surface_tasks.append({**task, "id": task["id"].replace("core", "surface"), "type": "surface", "code": surface_code})
        print(f"Surface")
    
    time.sleep(1)
    
    semantic_code = expand_task(task, "semantic")
    if semantic_code:
        semantic_tasks.append({**task, "id": task["id"].replace("core", "semantic"), "type": "semantic", "code": semantic_code})
        print(f"Semantic")
    
    time.sleep(1)
    
    difficult_code = expand_task(task, "difficult")
    if difficult_code:
        difficult_tasks.append({**task, "id": task["id"].replace("core", "difficult"), "type": "difficult", "code": difficult_code})
        print(f"Difficult")
    
    time.sleep(1)

# Сохраняем
output_dir = Path("data/torchmetrics_benchmark")
with open(output_dir / "surface_tasks.json", "w") as f:
    json.dump(surface_tasks, f, indent=2)

with open(output_dir / "semantic_tasks.json", "w") as f:
    json.dump(semantic_tasks, f, indent=2)

with open(output_dir / "difficult_tasks.json", "w") as f:
    json.dump(difficult_tasks, f, indent=2)

print(f"\nСоздано: Surface: {len(surface_tasks)}, Semantic: {len(semantic_tasks)}, Difficult: {len(difficult_tasks)}")