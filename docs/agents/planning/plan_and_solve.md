# Plan-and-Solve Prompting

- **Модуль**: [`PlanAndSolve`](../../../src/agents/planning/plan_and_solve/main.py)
- **Cтатья**: [Plan-and-Solve Prompting: Improving Zero-Shot Chain-of-Thought Reasoning by Large Language Models](https://arxiv.org/pdf/2305.04091)
- **Официальный репозиторий**: [GitHub](https://github.com/AGI-Edgerunners/Plan-and-Solve-Prompting)

---

## Принцип работы

Языковая модель составляет план, разделяя сложную задачу на более мелкие подзадачи.

---

## Параметры инициализации

- `examples: Optional[Union[str, List[Dict[str, str]]]]` - путь до файла с few-shot примерами в формате `jsonl` или список словарей с ключом `"example"`;
- `name: str = "PlanAndSolve"` - имя агента; 
- `model_url: str = "http://localhost:7215/v1"`;
- `model_name: str = "Qwen/Qwen1.5-32B-Chat-AWQ"`;
- `temperature: float = 0` - температура для генерации плана.

---

## Работа со скриптами

Является наследником класса Agent. Пример использования:

```
from src.agents.planning.plan_and_solve.main import PlanAndSolve

planner = PlanAndSolve("src/agents/planning/plan_and_solve/examples.jsonl")
plan = planner.run(task) # task: str - текстовое описание задачи 
```

Основной метод - `run` - возвращает план решения задачи (`str`).

---

## Особенности работы с методом

Решение генерируется за один запрос, план целиком передается в промпт. 
