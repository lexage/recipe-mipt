# Конфиги испытаний pk3 (ПМИ)

Два самодостаточных эксперимента, у каждого свой бейзлайн. Конфиги — копии
конфигов успешных экспериментов; изменены только путь к корпусу (JSON),
каталог индекса и, для фильтрации, порт модели на alibaba (7217). Ретриверы у
экспериментов разные, как и в исходных экспериментах.

| Файл | Роль | Источник |
|---|---|---|
| `filtration/f0_full.yaml` | бейзлайн фильтрации: полный корпус | `test_configs_experimental_7/filtration/simple_example_pure_full.yaml` |
| `filtration/f1_api_genre.yaml` | метод: корпус после отдельной фильтрации, шага фильтра в пайплайне нет | тот же пайплайн, что `f0_full` |
| `filtration/f1_api_genre_inline.yaml` | метод: фильтр внутри пайплайна; отсюда `filter_corpus.py` берёт метод и параметры | `test_configs_experimental_7/filtration/simple_example_api_genre_full.yaml` |
| `generation/g0_base.yaml` | бейзлайн генерации: системный промпт без правил | `final_test_configs/c0_base.yaml` |
| `generation/g1_rules.yaml` | метод: генератор правил в пайплайне, `rebuild: True` | `final_test_configs/c1_general_per_problem.yaml` |

Отдельные запуски (из корня recipe-mipt, в контейнере с venv):

```bash
python filter_corpus.py -c pmi_configs/filtration/f1_api_genre_inline.yaml -o data/filtered/docs_database_examples.api_genre.json.gz
python generate_rules.py -c pmi_configs/generation/g1_rules.yaml -o results/pmi_rules/standalone
python run_ds1000.py -c pmi_configs/generation --log-chunks
```

В испытаниях эти конфиги не запускаются напрямую: `recipe-mipt-eval/scripts/run-evaluation.sh`
делает их копии с путями индекса и журнала правил внутри каталога прогона, чтобы
каждый прогон строил индекс с нуля и не трогал результаты прошлых прогонов.
