# Chain-of-Thought

- **Модуль**: [`CoT`](../../../src/agents/reasoning/cot/main.py)
- **Модуль**: [`CoT`](../../../src/agents/reasoning/cot/main.py)
- **Cтатья**: 
    * CoT Few-Shot: [Chain-of-Thought Prompting Elicits Reasoning in Large Language Models](https://proceedings.neurips.cc/paper_files/paper/2022/file/9d5609613524ecf4f15af0f7b31abca4-Paper-Conference.pdf)
    * CoT Zero-Shot: [Large Language Models are Zero-Shot Reasoners](https://proceedings.neurips.cc/paper_files/paper/2022/file/8bb0d291acd4acf06ef112099c16f326-Paper-Conference.pdf)

---

## Принцип работы

Есть два режима работы: Few-Shot и Zero-Shot. 
* Few-Shot: модели предоставляется несколько демонстрационных примеров, каждый из которых содержит входной вопрос, последовательность промежуточных шагов рассуждения и итоговый ответ. 
* Zero-Shot: к запросу добавляется фраза-инструкция, такая как «Let’s think step by step», которая побуждает модель генерировать цепочку рассуждений. 
* Few-Shot: модели предоставляется несколько демонстрационных примеров, каждый из которых содержит входной вопрос, последовательность промежуточных шагов рассуждения и итоговый ответ. 
* Zero-Shot: к запросу добавляется фраза-инструкция, такая как «Let’s think step by step», которая побуждает модель генерировать цепочку рассуждений. 

---

## Параметры инициализации

- `name: str = "CoT"` - имя агента;
- `mode: str = "zero-shot"` - режим: `"zero-shot"` или `"few-shot"`;
- `few_shot_examples: Optional[Union[str, List[Dict[str, str]]]] = None` - путь до файла с примерами в формате `jsonl` или список словарей с ключом `"example"`.  

---

## Работа со скриптами

Является наследником класса Agent. Пример использования:

```
from src.agents.resoning.cot.main import CoT

reasoner = CoT(mode="few-shot", few_shot_examples="examples.txt")
solution = reasoner.run(task) # task: str - текстовое описание задачи 
```

Основной метод - `run` - возвращает рассуждение (`str`).
