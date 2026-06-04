#!/usr/bin/env bash
# Несколько прогонов папки с конфигами — для сбора статистики (повторы).
#
# Свежий индекс на каждый фильтр уже задан в самих конфигах (уникальный
# path_to_vector_db), поэтому здесь ничего чистить не нужно — просто повторяем.
# Каждый прогон кладёт результаты в новый results/<config>/<timestamp>/.
#
# Если меняем сам фильтр и хотим пересобрать индексы — один раз удалить папку:
#   rm -rf data/vdb_exp3

python run_ds1000.py -c test_configs_experimental_3
python run_ds1000.py -c test_configs_experimental_3
python run_ds1000.py -c test_configs_experimental_3
# для 5 повторов — допиши ещё две такие строки
