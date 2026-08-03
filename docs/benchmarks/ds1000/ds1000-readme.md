# DS1000 Benchmark Integration

Этот модуль предоставляет интеграцию с бенчмарком DS1000 для оценки качества генерации кода.

## Установка зависимостей

```bash
pip install -r src/benchmarks/ds1000/requirements_ds1000.txt
```

## Быстрый старт

```python
from benchmarks.ds1000 import DS1000

def main():
    # Инициализация бенчмарка
    benchmark = DS1000("data/ds1000.jsonl.gz")
    
    # Определение методов для обработки
    def my_preprocess(task):
        """Предобработка задачи - извлекаем промпт для модели"""
        return task.prompt
    
    def my_run_model(prompt):
        """Запуск модели - здесь ваша логика генерации кода"""
        # Ваша модель генерирует код по промпту
        generated_code = "your_generated_code_here"
        return generated_code
    
    # Запуск оценки
    results = benchmark.eval(
        run_method=my_run_model,
        preprocess_method=my_preprocess,
        save_path="results"
    )


# ВАЖНО: оборачивайте вызов в if __name__ == "__main__"
if __name__ == "__main__":
    main()
```

## Детали реализации

### Метод `eval()`

Метод `eval()` принимает два ключевых аргумента:

- **`run_method`** (обязательный): функция, которая принимает обработанную задачу и возвращает сгенерированный код как строку
- **`preprocess_method`** (опциональный): функция для предобработки `DataItemDS1000` перед передачей в `run_method`
- **`save_path`** (опциональный): путь, по которому будут сохраняться результаты. По умолчанию `results/`. Каждый отдельный прогон будет сохраняться по этому пути в подпапке с таймстемпом. Напр. `results\20251101_115040`
- **`continue_exp`** (опциональный): Таймстэмп эксперимента, по которому уже есть результаты прогона и их надо просто оценить. Например если `20251101_115040`, то скрипт загрузит результаты из `results\20251101_115040` и оценит их

#### Примеры использования:

**Без предобработки:**
```python
def run_directly(task):
    # task - объект DataItemDS1000
    return generate_code(task.prompt)

benchmark.eval(run_method=run_directly)
```

**С предобработкой:**
```python
def preprocess(task):
    # Можно модифицировать промпт, добавить контекст и т.д.
    return f"Python code: {task.prompt}"

def run_model(modified_prompt):
    return your_llm.generate(modified_prompt)

benchmark.eval(
    run_method=run_model,
    preprocess_method=preprocess
)
```

### Структура DataItemDS1000

Каждая задача содержит:
```python
@dataclass
class DataItemDS1000:
    p_id: int             # ID проблеммы
    prompt: str           # Текст задачи на естественном языке
    reference_code: str   # Эталонное решение
    metadata: dict        # Метаданные (библиотека, тип пертурбации)
    code_context: str     # Контекст кода (импорты, тесты)
```

## Критически важный нюанс: многопроцессорность

**Всегда используйте `if __name__ == "__main__"` при запуске бенчмарка!**

### Пример:
```python
from benchmarks.ds1000 import DS1000

def main():
    benchmark = DS1000("./data/ds1000.jsonl.gz")
    summary = benchmark.eval(run_method=my_model)

if __name__ == "__main__":
    main()
```

Бенчмарк использует `ProcessPoolExecutor` для параллельного выполнения кода. Без защиты `if __name__ == "__main__"` дочерние процессы будут рекурсивно импортировать главный модуль, что приводит к:
- `RuntimeError` при создании процессов
- `BrokenProcessPool` из-за аварийного завершения процессов
- Невозможности завершить оценку

## Пример с реальной моделью

```python
from benchmarks.ds1000 import DS1000
import your_llm_library

def main():
    # Инициализация вашей модели
    model = your_llm_library.load_model("your-model")
    
    benchmark = DS1000("./data/ds1000.jsonl.gz")
    
    def preprocess_for_llm(task):
        """Подготовка промпта для LLM"""
        return f"""
Write Python code to solve the following task:
{task.prompt}

Return only the code without any explanations.
"""
    
    def run_llm(prompt):
        """Запрос к LLM"""
        response = model.generate(prompt, max_tokens=200)
        return extract_code_from_response(response)
    
    # Запуск оценки
    results = benchmark.eval(
        run_method=run_llm,
        preprocess_method=preprocess_for_llm,
        save_path='results/test_model'
    )
 

if __name__ == "__main__":
    main()
```

## Интерпретация результатов

Результаты содержат:
- **Overall accuracy**: общая точность по всем задачам
- **Per-library accuracy**: точность по каждой библиотеке (NumPy, Pandas, etc.)
- **Per-perturbation accuracy**: точность по типам пертурбаций

Пример вывода:
```
score      
count  1000.000
mean      0.450
          score      
library   count  mean
numpy     200.0  0.600
pandas    200.0  0.350
...
```

## Структура данных

Убедитесь, что файл с данными находится по пути `./data/ds1000.jsonl.gz` или укажите правильный путь при инициализации `DS1000("./your/path/ds1000.jsonl.gz")`.