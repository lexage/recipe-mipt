# QuRating

- **Пайплайн**: [`qurating_pipeline`](../../src/filtering/qurating/main.py)
- **Cтатья**: [QuRating: Selecting High-Quality Data for Training Language Models](https://arxiv.org/pdf/2402.09739)
- **Официальный репозиторий**: [GitHub](https://github.com/princeton-nlp/QuRating)

---

## Принцип работы

Процесс начинается со сбора попарных суждений от LLM-судьи, который сравнивает чанки по четырем критериям: ясность и читаемость кода, техническая точность и корректность, образовательная ценность для программирования и практическая полезность и применимость. Полученные сравнительные оценки преобразуются в скалярные рейтинги с помощью модели Брэдли-Терри, на которых затем обучается эффективная модель QuRater (на базе `Sheared-Llama-1.3B`). Обученная модель аннотирует весь целевой корпус, присваивая каждому чанку баллы качества. Наконец, проводится финальная фильтрация через экспоненциальную выборку - вероятностный метод, при котором шанс документа попасть в итоговый набор прямо пропорционален экспоненте его рейтинга, деленной на параметр температуры. В этой схеме баллы качества рассматриваются как логиты над чанками, из которых производится выборка без возвращения.

---

## Аргументы функции

- `chunks_for_training: List[Chunk]` - список чанков для сбора данных, на которых будет обучаться QuRater;
- `chunks_for_filtering: List[Chunk]` - список чанков для фильтрации.

Для каждого этапа работы метода используются настраиваемые конфигурации параметров. Ознакомиться с ними и их описанием можно [здесь](../../src/filtering/qurating/configs).

---

## Работа со скриптами

Пример использования:

```
from src.filtering.qurating.main import qurating_pipeline

filtered_chunks = qurating_pipeline(chunks_for_training, chunks_for_filtering)
```

Возвращает `List[Chunk]`. 

Промпты для попарного сравнения текстов по заданным критериям качества лежат в `"src/filtering/qurating/prompting/templates"`.

Данные для обучения QuRater сохраняются в `"src/filtering/qurating/datasets/judgement_data"` (разбиты по критериям качества).

Обученный QuRater сохраняется в `"src/filtering/qurating/checkpoints-preferences/trained_qurater"`.

Набор данных, аннотированный обученным QuRater'ом, сохраняется в `"src/filtering/qurating/datasets/annotated_data"` в формате `Arrow`.

Отобранные чанки, разбитые по шардам, сохраняются в `"src/filtering/qurating/datasets/selected_data"` в формате `Arrow`.