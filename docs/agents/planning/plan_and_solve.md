# Plan-and-Solve Prompting

- **Модуль**: [`PlanAndSolveAgent`](../../../src/agents/planning/plan_and_solve/main.py)
- **Cтатья**: [Plan-and-Solve Prompting: Improving Zero-Shot Chain-of-Thought Reasoning by Large Language Models](https://arxiv.org/pdf/2305.04091)
- **Официальный репозиторий**: [GitHub](https://github.com/AGI-Edgerunners/Plan-and-Solve-Prompting)

---

## Принцип работы

Языковая модель составляет план, разделяя сложную задачу на более мелкие подзадачи.

---

## Работа со скриптами

Является наследником класса Agent. Пример использования:

```
from src.agents.planning.plan_and_solve.main import PlanAndSolveAgent

planner = PlanAndSolveAgent() # передавать параметры инициализации не нужно
plan = planner.run(task) # task: str - текстовое описание задачи 
```

Основной метод - `run` - возвращает план решения задачи.

---

## Особенности работы с методом

Решение генерируется за один запрос, план целиком передается в промпт. 
