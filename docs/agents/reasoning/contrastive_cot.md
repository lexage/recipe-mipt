# Contrastive Chain-of-Thought

- **Модуль**: [`ContrastiveCoT`](src/agents/reasoning/contrastive_cot/main.py)
- **Cтатья**: [Contrastive Chain-of-Thought Prompting](https://arxiv.org/pdf/2311.09277)
- **Официальный репозиторий**: [GitHub](https://github.com/DAMO-NLP-SG/contrastive-cot)

---

## Принцип работы

Работает в формате Few Shot. Для каждого примера языковой модели предоставляются как верные, так и неверные цепочки рассуждений. Неверные цепочки рассуждений могут отсутствовать, тогда метод сгенерирует их самостоятельно.

---

## Чего не хватает (TODO)

Нужно добавить автоматическую генерацию неверных цепочек рассуждений. 

---

## Параметры инициализации

- `demonstrations: Union[str, List[Dict[str, str]]]` - путь до файла в формате `jsonl` с примерами или список словарей с ключами `"problem"` (текст вопроса), `"correct_explanation"` (верное рассуждение), `"correct_code"` (верное решение), `"incorrect_explanation"` (неверное рассуждение, опционально), `"incorrect_code"` (неверное решение, опционально);
- `name: str = "ContrastiveCoT"` - имя агента.

---

## Работа со скриптами

Является наследником класса Agent. Пример использования:

```
from src.agents.resoning.contrastive_cot.main import ContrastiveCoT

reasoner = ContrastiveCoT("demonstrations.jsonl")
solution = reasoner.run(task) # task: str - текстовое описание задачи 
```

Основной метод - `run` - возвращает решение задачи.