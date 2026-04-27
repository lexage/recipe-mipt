# Least-to-Most Prompting

- **Модуль**: [`LeastToMost`](../../../src/agents/planning/least_to_most/main.py)
- **Cтатья**: [LEAST-TO-MOST PROMPTING ENABLES COMPLEX REASONING IN LARGE LANGUAGE MODELS](https://arxiv.org/pdf/2205.10625)

---

## Принцип работы

Метод заключается в разбиении сложной задачи на последовательность более простых подзадач, которые решаются строго по порядку. 

---

## Параметры инициализации

- `examples: Optional[Union[str, List[Dict[str, str]]]]` - путь до файла с few-shot примерами в формате `jsonl` или список словарей с ключом `"example"`;
- `name: str = "L2M_Planner"` - имя агента; 
- `model_url: str = "http://localhost:7215/v1"`;
- `model_name: str = "Qwen/Qwen1.5-32B-Chat-AWQ"`;
- `temperature: float = 0` - температура для генерации плана.

---

## Работа со скриптами

Является наследником класса Agent. Пример использования:
 
```
from src.agents.planning.least_to_most.main import LeastToMost

planner = LeastToMost("src/agents/planning/least_to_most/examples.jsonl")
plan = planner.run(task) # task: str - текстовое описание задачи 
```

Основной метод - `run` - возвращает список подзадач (`List[str]`), на которые разбилась исходная задача, от простой до сложной. Исходная задача будет в этом списке последней. 

---

## Особенности работы с методом

Подзадачи должны решаться последовательно, что подразумевает несколько обращений к модели. Ответ на предыдущую подзадачу включается в промпт для решения следующей подзадачи. 
