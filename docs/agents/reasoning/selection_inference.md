# Selection-Inference

- **Модуль**: [`SelectionInference`](../../../src/agents/reasoning/selection_inference/main.py)
- **Cтатья**: [Selection-Inference: Exploiting Large Language Models for Interpretable Logical Reasoning](https://arxiv.org/pdf/2205.09712)

---

## Принцип работы

Разбивает процесс рассуждения на два этапа: 

* **Selection**: модель выбирает из контекста ограниченный набор фактов, достаточный для совершения ровно одного шага рассуждения.

* **Inference**: модель, опираясь исключительно на эти выбранные факты и не имея прямого доступа к исходному вопросу, генерирует новое промежуточное утверждение. 

Процесс повторяется циклично: каждый новый факт добавляется обратно в контекст, становясь доступным для последующих итераций, пока не будет достигнуто максимальное число шагов алгоритма. 

---

## Параметры инициализации

- `facts_from_context_examples: Optional[str] = None` - путь до файла с примерами извлечения фактов из описания задачи. Формат `jsonl`. Имеет ключи `"task"` (описание задачи) и `"facts"` (извлеченные факты); 
- `facts_from_facts_examples: Optional[str] = None` - путь до файла с примерами извлечения из набора фактов наиболее релевантных задаче. Формат `jsonl`. Имеет ключи `"task"` (описание задачи), `"facts"` (текущий набор фактов), `"selection"` (несколько наиболее релевантных фактов); 
- `inference_examples: Optional[str] = None` - путь до файла с примерами вывода из нескольких фактов, полученных на этапе Selection, нового факта. Формат `jsonl`. Имеет ключи `"selection"` (набор фактов) и `"inference"` (утверждение, выведенное на основе этих фактов); 
- `name: str = "SelectionInference"` - имя агента; 
- `selection_mode: str = "simple"` - метод выбора фактов на этапе Selection. Есть две опции: `"simple"` и `"scoring"`.   
    * Simple подход: показать модели несколько примеров и явно попросить ее сгенерировать необходимые факты;
    * Scoring подход: для каждого факта в текущем наборе получить оценку его логарифмического правдоподобия, добавить факт с максимальной оценкой к выбранным фактам и в текущий набор, повторять процесс, пока не наберется нужное количество фактов;
- `max_steps: int = 5` - число итераций алгоритма;
- `temperature: float = 0.`;
- `max_facts: int = 3` - число фактов, которое будет отобрано на этапе Selection методом scoring.

---

## Работа со скриптами

Является наследником класса Agent. Пример использования:

```
from src.agents.reasoning.selection_inference.main import SelectionInference

reasoner = SelectionInference(
    facts_from_context_examples="src/agents/reasoning/selection_inference/facts_from_context_examples.jsonl",
    facts_from_facts_examples="src/agents/reasoning/selection_inference/facts_from_facts_examples.jsonl",
    inference_examples="src/agents/reasoning/selection_inference/inference_examples_v2.jsonl",
    selection_mode="score",
    max_steps=5
)
solution = reasoner.run(question)
```

Основной метод - `run` - возвращает решение задачи или последний выведенный факт на этапе Inference (`str`).