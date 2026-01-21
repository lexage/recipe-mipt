# Doc Parser

Инструмент для парсинга технической документации (NumPy, Pandas, TensorFlow и др.) с помощью Scrapy.

## Установка

```bash
pip install -r requirements.txt
```

## Запуск спайдера

Для запуска парсинга документации используйте команду:

```bash
scrapy crawl <spider_name>
```

Конфиги и названия доступных пауков: [`spiders_config.py`](doc_parser/spiders_configs.py)

Пример:
```bash
scrapy crawl numpy_user_guide
```

Результаты сохраняются в папку `results/`.

## Работа с базой данных

Скрипт [`create_db.py`](scripts/create_db.py) создает SQLite базу данных из спарсенных файлов:

```bash
python scripts/create_db.py --db documentation.db --docs results 
```

Параметры:
- `--db` — путь к файлу БД (по умолчанию: `documentation.db`)
- `--docs` — путь к папке с результатами парсинга (по умолчанию: `results`)

### Схема БД

![Schema](assets/db_schema.png)