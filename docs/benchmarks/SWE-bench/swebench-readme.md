# SWE-bench Benchmark Integration

Получение и оценка предсказаний на бенчмарке [SWE-bench](https://github.com/SWE-bench/SWE-bench). 

## Установка

Для работы с SWE-bench потребуется [установить Docker](https://docs.docker.com/engine/install/). Если у Вас Linux, обратите внимание также на эти [послеустановочные шаги](https://docs.docker.com/engine/install/linux-postinstall/).

Далее выполняем следующие команды:

```bash
cd src/benchmarks/SWE-bench
pip install -e .
```

Можно протестировать свою установку:

```bash
python -m swebench.harness.run_evaluation \
    --predictions_path gold \
    --max_workers 1 \
    --instance_ids sympy__sympy-20590 \
    --run_id validate-gold
```

**Важно**: eсли вы используете MacOS серии M или другие системы на базе ARM, добавьте в приведенный выше скрипт `--namespace ''`.

## Использование

*Пока здесь реализация только для метода `dummy_run`. В будущем надо будет импортировать другие методы и использовать их.* 

Запускать из `src/benchmarks/SWE-bench`. 

```bash
python -m swebench.inference.core \
    --method_name dummy \
    --dataset_path SWE-bench/SWE-bench_Lite \
    --run_id <run_id> \
    --max_workers <num_workers>
```

Отдельно получить предсказания можно с помощью следующей команды:

```bash
python -m swebench.inference.run_agent_pipeline \
    --method_name dummy \
    --dataset_path SWE-bench/SWE-bench_Lite
```

Оценить готовые предсказания можно, используя:

```bash
python -m swebench.harness.run_evaluation \
    --dataset_name SWE-bench/SWE-bench_Lite \
    --predictions_path <path_to_predictions> \
    --max_workers <num_workers> \
    --run_id <run_id>
    # use --predictions_path 'gold' to verify the gold patches
    # use --run_id to name the evaluation run
    # use --modal true to run on Modal
```

