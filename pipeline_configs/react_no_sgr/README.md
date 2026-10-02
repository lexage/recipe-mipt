# ReAct без SGR: абляция для таблицы 9

Новый агент — `src/agents/pipelines/react_no_sgr.py`, класс
`ReActAgentNoSGR`, тип компонента `REACT_AGENT_NO_SGR`.
Основа — `react_sgr.py` из ветки `merge-dev-to-react-exp`.
Используется прежний pipeline `type: REACT` и прежний runner `run_ds1000.py`.
Исходные `react_sgr.py` и `react.py` не изменены.

## Что именно отключено

Вместо `chat.completions.parse(response_format=AgentStep)` агент вызывает
`chat.completions.create` с `model`, `messages`, `temperature`.
Нет `response_format`, JSON mode, JSON Schema, guided decoding или OpenAI
function calling в запросах основного ReAct-агента. Нет Pydantic-модели шага,
`message.parsed`, поля `is_final` и требования вернуть весь шаг в JSON.

Один шаг генерируется как обычный текст:

```text
Thought: I need to check the documentation.
Action: db_search
Action Input: {"query": "pandas iloc reorder rows"}
```

Метки разбираются после генерации. Словарь в `Action Input` нужен только для
передачи именованных аргументов Python-инструменту; это не схема генерации
ответа. Поддерживаются JSON и литерал Python-словаря, включая примеры с
одинарными кавычками; используется `ast.literal_eval`, не `eval`.
Описания параметров инструментов сохранены. Проверяется совместимость
аргументов с Python-сигнатурой; реализация самих инструментов не меняется.
Внутренние LLM-вызовы Critic/RAG сохраняют исходное поведение.

При `Action: finish` выполняется тот же отдельный запрос для финального
ответа, с исходным `FINISH_PROMPT_TEMPLATE`. Таким образом, no-tools baseline
тоже состоит из шага рассуждения/завершения и финального запроса, как SGR-вариант.
Сохраняются лимит 10 итераций, предел 5 последовательных неудачных попыток,
окно истории из 5 сообщений, защита от повторяющихся вызовов и обработка ошибок.
Статические константы и few-shot примеры импортируются из `react_sgr.py`;
класс SGR не наследуется и его методы не вызываются. Во всех новых ReAct-конфигах
явно установлен `few_shot_type: zero_shot`.

Это абляция всего формата SGR: меняются и способ генерации, и формат шага в
промпте/истории. Ее результат не следует трактовать как эффект только одного
API-параметра `response_format` при неизменном промпте.

## Конфиги и соответствие архивам

Каждый YAML содержит путь к исходному конфигу в комментарии `Source`.

| Новый конфиг | Эксперимент из приложенных архивов |
| --- | --- |
| `table9/baseline_llm.yaml` | `baseline_10_04_26_chat_3`: прямой LLM baseline, уже без SGR |
| `table9/react_code_tool.yaml` | `react_sgr_exp_temp_0.7_8_fix_code_tool_qwen2_5` |
| `table9/react_llm_tool.yaml` | `react_sgr_exp_temp_0.7_8_fix_llm_tool_qwen2_5` |
| `table9/react_critic_tool.yaml` | `react_sgr_exp_temp_0.7_9_fix_db_search_qwen2_5_critic_tool` |
| `table9/react_simple_rag.yaml` | `react_exp_temp_0.7_19_fix_db_search_qwen2_5_simple_semantic_rag` |
| `table9/react_instruct_rag.yaml` | `react_exp_temp_0.7_19_fix_db_search_qwen2_5_instruct_rag` |
| `table9/react_corag.yaml` | `react_exp_temp_0.7_19_fix_db_search_qwen2_5_corag` |
| `baseline/react_no_tools.yaml` | `exps_react_baseline_24_09/react_baseline.yaml` |

`baseline_llm.yaml` — контроль из таблицы 9, а `react_no_tools.yaml` — отдельная
пара к приложенному ReAct baseline с SGR. Это разные экспериментальные условия.

Названия моделей, URL, температура и используемые настройки retrieval взяты
из архивов. Пути логов заменены на относительные `logs/react_no_sgr/...`.
Для Code/LLM/Critic удалены неиспользуемые embedder/DB/context-компоненты:
эти инструменты не используют retrieval. В Critic-конфиге `name: CriticTool`
заменено на `name: critic`: это допустимый ключ текущего `CriticTool`.
Параметры `documents_as_chunks` у InstructRetriever и `return_docs_as_chunks`
у SimpleRetriever отсутствуют в конструкторах этой ветки и раньше молча
игнорировались фабрикой; они исключены из новых конфигов.

### Расхождение в CoRAG

В приложенном YAML, несмотря на `qwen2_5` в имени, основной агент использует
`deepseek-ai/deepseek-coder-33b-instruct` на порту 7217. Это значение сохранено,
а не заменено по имени файла. При этом `src/rag/corag/retriever.py` в исходной
ветке жестко задает для retrieval модель `Qwen/Qwen2.5-32B-Instruct`, а архивный
YAML направляет оба компонента на один URL.

Перед запуском CoRAG нужно согласовать фактически обслуживаемые модели и
адреса. Один сервер с единственной моделью обычно не примет оба имени.
Для эксперимента полностью на Qwen задайте в `components.agent.params`
`model_name: Qwen/Qwen2.5-32B-Instruct` и его URL, а в
`components.retriever.params.url` — URL Qwen. Для пары DeepSeek-agent +
Qwen-retriever укажите раздельные соответствующие URL. Те же значения должны
использоваться в парном SGR-запуске. Архив не позволяет однозначно установить,
какой набор моделей использовался при получении прежней метрики.

## Запуск

Команды выполняются из корня репозитория, в окружении для DS-1000.
До запуска укажите доступные адреса моделей и, для RAG, пути к той же базе
документов/векторов. В table9 сохранены адреса `172.18.0.1:7215/7216/7217`,
а в no-tools baseline — `127.0.0.1:11456`, как в приложенных YAML.
Если модель теперь доступна через туннель, обновите все нужные URL согласованно.
`LocalDB` этой ветки использует Qdrant для векторов: архивный путь должен
указывать на базу совместимого формата. Прямой `DS1000Solver` выбирает модель
из первого элемента `/models` и игнорирует `model_name` в YAML; проверьте,
что этот endpoint обслуживает нужный Qwen.

Все варианты таблицы 9:

```bash
python run_ds1000.py -c pipeline_configs/react_no_sgr/table9 -d data/ds1000/ds1000.jsonl.gz -s results/react_no_sgr/table9 -n 1
```

Дополнительный ReAct baseline без инструментов:

```bash
python run_ds1000.py -c pipeline_configs/react_no_sgr/baseline -d data/ds1000/ds1000.jsonl.gz -s results/react_no_sgr/baseline -n 1
```

`-c` принимает папку, не путь к отдельному YAML; для одного варианта положите
нужный конфиг в отдельную папку. Runner создаст отдельные каталоги результатов
по имени конфига и времени. Логи новых запусков отделены от старых.

В исходной SGR-реализации `tools or [LLMTool(...)]` заменяет даже `tools: []`
инструментом по умолчанию. Новый агент использует fallback только при
`tools is None`: явный пустой список остается пустым. Для повторного SGR
baseline нужна уже применявшаяся правка `if tools is None else tools` и
исходная JSON-инструкция из `react_baseline.yaml`; текстовую инструкцию нового
baseline переносить в SGR-конфиг не нужно.

В логах нового агента выводятся:

```text
REACT_MODE: text; SGR: false; response_format: unset
AVAILABLE_TOOLS: []
```

`TOOL_CALL: <name>` пишется непосредственно перед вызовом доступного
инструмента. `ACTION: execute` само по себе означает только ответ модели:
неизвестное действие возвращает ошибку и ничего не запускает.

## Условия сопоставимости

Для пар SGR/no-SGR используйте одинаковые модель и sampling, задачи, базу,
инструменты, few-shot, лимиты и окно истории. Новые configs не меняют состав
DS-1000; `zero_shot` не добавляет примеры. Сравнивайте одинаковые `problem_id`
и учитывайте ошибки формата/исчерпание лимита вместе с accuracy.
При температуре 0.7 желательно сравнивать средние нескольких повторов.

В исходном `run_ds1000.py` один агент используется всеми рабочими потоками,
а `react_sgr.py` хранит историю на экземпляре. Поэтому для строгой пары с
неизмененной SGR-реализацией выше указан `-n 1` для обеих сторон. Новый агент
изолирует историю и счетчики на каждый `run()` и проверен на одновременных
вызовах; потокобезопасность внешних инструментов остается свойством самих
инструментов. Исторические результаты из архивов полезны как ориентир,
но сами по себе не гарантируют одинаковую версию кода и условия запуска.

## Проверка без сервера модели

```bash
python -m unittest discover -s tests -p test_react_no_sgr.py -v
```

Тесты проверяют текстовый парсер, содержимое запросов настоящего OpenAI SDK
через mock HTTP, вызов инструмента и observation, отдельный финальный запрос,
пустой список инструментов, ошибки/лимиты/циклы, изоляцию историй и загрузку
всех YAML. Для baseline проверена сборка через реальный registry/factory/builder.
Тяжелые неиспользуемые импорты пакетов изолированы в тестах.
Полный DS-1000 и внешние Critic/RAG-сервисы этими тестами не запускаются.
