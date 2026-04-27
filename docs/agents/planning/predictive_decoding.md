# Predictive-Decoding

- **Модуль**: [`MPCSampleAgent`](../../../src/agents/planning/predictive_decoding/main.py)
- **Cтатья**: [NON-MYOPIC GENERATION OF LANGUAGE MODELS FOR REASONING AND PLANNING](https://arxiv.org/pdf/2410.17195)
- **Официальный репозиторий**: [GitHub](https://github.com/chang-github-00/LLM-Predictive-Decoding)

---

## Принцип работы

На каждом шаге алгоритм параллельно генерирует несколько траекторий предсказания на фиксированную глубину вперед, моделируя возможные варианты развития событий. Исходное распределение вероятностей LLM корректируется (перевзвешивается) на основе оценки этих будущих сценариев, после чего модель выбирает первое действие из наиболее перспективной траектории.

---

## Параметры инициализации

- `model_url: str = "http://localhost:7215/v1"`;
- `model_name: str = "Qwen/Qwen1.5-32B-Chat-AWQ"`;
- `name: str = "MPCSample"` - имя агента;
- `prompt_path: Optional[str] = None` - Путь к `json`-файлу с шаблонами промптов, в файле должны быть ключи `"prompt"` и `"system_msg"`; 
- `lookahead_thought_length: int = 3` - количество предстоящих действий, которые агент планирует заранее (длина цепочки lookahead);
- `lookahead_token_length: Optional[int] = None` - максимальная длина в токенах для lookahead-последовательности, если None, используется `lookahead_thought_length`;
- `reward_threshold: float = 1.0` - пороговое значение вознаграждения, необходимое для принятия действия; 
- `beam_size: int = 8` - количество последовательностей-кандидатов, сохраняемых во время генерации;
- `beam_temperature: float = 0.7` - параметр температуры для сэмплирования при создании кандидатов; 
- `select_temperature: float = 0.1` - параметр температуры для вероятностного выбора финального действия из кандидатов;
- `n_generate_sample: int = 8` - количество генерируемых последовательностей действий за один вызов LLM;
- `value_type: str = "logp"` - тип оценки действий (используется `"logp"` — логарифмическая вероятность);
- `do_sample: bool = True` - флаг, указывающий, использовать ли сэмплирование при выборе действия;
- `use_memory: bool = True` - флаг использования памяти агента для хранения истории действий;
- `max_problem_size: int = 50` - максимальное количество шагов, которые агент может сделать при решении задачи;
- `examples: Optional[Union[str, List[Dict[str, str]]]] = None` - путь до файла с few-shot примерами в формате `jsonl` или список словарей с ключом `"example"`. Можно взять примеры для Plan-and-Solve.

---

## Работа со скриптами

Является наследником класса Agent. Пример использования:

```
from src.agents.planning.predictive_decoding.main import MPCSample

planner = MPCSample(examples="src/agents/planning/plan_and_solve/examples.jsonl")

plan_prompts = {
    "system_msg": (
        "You are an AI assistant that generates only natural language plans. "
        "Never include code, code blocks, or any programming syntax. "
        "Each step must be a separate line, numbered if possible."
    ),
    "prompt": (
        f"Problem: {question}\n\n"
        "Provide a step-by-step plan to solve this problem. "
        "Use plain English only. Do not write any code.\n\n"
        "Stop after the plan; do not add any extra text, separators, or code.\n\n"
        "Plan:\n"
    ),
    "question": ""
}

success, plan = planner.run(question, prompts=plan_prompts, end_suffix="\n\n")
```

Основной метод - `run` - возвращает флаг об успешном завершении работы (`bool`) и план решения задачи (`str`).

---

## Замечание

В официальном репозитории также есть reward-версия метода (в статье она дает лучшие результаты) и версия, которая работает параллельно. Возможно пригодится. Нужно смотреть файлы [mpc_sampling.py](https://github.com/chang-github-00/LLM-Predictive-Decoding/blob/main/agentboard/algorithms/mpc_sampling.py) и [mpc_reward_sampling.py](https://github.com/chang-github-00/LLM-Predictive-Decoding/blob/main/agentboard/algorithms/mpc_reward_sampling.py).