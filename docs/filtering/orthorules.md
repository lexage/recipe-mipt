# Orthorules

- **Пайплайн**: [`orthorules_pipeline`](../../src/filtering/orthorules/main.py)
- **Cтатья**: [Selection of LLM Fine-Tuning Data based on Orthogonal Rules](https://arxiv.org/pdf/2410.04715)
- **Официальный репозиторий**: [GitHub](https://github.com/XiaominLi1998/Submission-OrthoRules/)

---

## Принцип работы

Сначала модель генерирует широкий набор разнообразных правил оценки качества текста. Полученные правила валидируются человеком. На основе этих правил LLM-судья оценивает небольшую выборку данных, формируя векторы оценок. После этого с помощью детерминантного точечного процесса (DPP) выбирается подмножество наиболее независимых (ортогональных) правил. В завершение весь массив данных оценивается по выбранным правилам. Полученные оценки позволяют рассчитать вероятности для сэмплирования $k$ итоговых чанков.

---

## Аргументы функции

- `chunks: List[Chunk]` - список чанков для фильтрации;
- `subset_size: int = 10000` - размер подмножества, которое будет оцениваться по всем правилам;
- `batch_size: int = 1000` - размер батча (для оценивания данные разбиваются на батчи), должен быть меньше `subset_size`; 
- `r: int = 10` - количество выбранных правил; 
- `tau: float = 1.0` - температура в формуле для сэмплирования итоговых чанков;  
- `k: int = 20000` - число чанков, которое будет отобрано;
- `sampling_method: str = "gumbel"` - метод сэмплирования итоговых данных, может быть `"gumbel"` (Gumbel top-k trick), `"stochastic_sampling"` и `"top-k"`.

## Работа со скриптами

Пример использования:

```
from src.filtering.orthorules.main import orthorules_pipeline

filtered_chunks = orthorules_pipeline(chunks)
```

Возвращает `List[Chunk]`.

В [rating_prompts.py](../../src/filtering/orthorules/experiments_ds1000/rating/rating_prompts.py) лежат все необходимые промпты, а также сами правила. 