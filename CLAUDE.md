# Recipe-MIPT — Filtration Research (project notes for Claude)

This file is the entry point for any new session. Read it end-to-end before
acting; it contains the **what / where / why** of the work that was done
so far, plus the conventions and the next plan.

---

## ⛔ ABSOLUTE RULE — Claude NEVER touches the server

**You (Claude) must NEVER run any command that interacts with the remote
server.** No `rsync`, no `ssh`, no pulls, no pushes — nothing that reaches
`mipt_aigrant_cluster` / `VizilterK@proxy2.cod.phystech.edu`. **Only
Konstantin works with the server.**

Your job is strictly: **read local files and write code.** When a server
command is needed (rsync, runs, etc.), you write out the command and
Konstantin runs it himself. Do not execute it.

(This rule exists because it was violated once — Claude ran an rsync. It must
never happen again. When in doubt about whether an action is allowed, don't
do it; hand it to Konstantin.)

---

## ⌨️ Соглашение по командам (как подавать Konstantin)

Каждая команда — **в отдельном блоке кода, строго по одной**, чтобы удобно
копировать. **Внутри блоков кода — никаких комментариев**: в поле кода только
сама команда, все пояснения — обычным текстом до/после блока. Не склеивать
несколько команд через `&&`/переносы в один блок и не придумывать лишних шагов
(временные папки и т.п.), которых не было в принятом воркфлоу, если не просят.

---

## Где мы остановились — 10.06.2026 (ОПТИМИЗАЦИИ ВНЕСЕНЫ — ждём смоук-замер времени)

> Серия exp5 вставала из-за катастрофической скорости LLM-фильтров (конфиг #1
> `topic_llm_dirty` шёл ~18 ч). Провели «большое обсуждение» и **внесли согласованные
> output-preserving правки** (ускорение, смысл метрики R не меняем). Код готов к запуску.
> Следующий шаг — Konstantin делает **смоук-замер времени на одном грязном LLM-конфиге**,
> прежде чем гнать все 12. Диагноз, по которому встали, — в блоке ⚠️ ниже (для истории).

### ✅ Что сделано 10.06 (ускорение фильтров, без изменения смысла R)
Стратегия: «параллелим LLM-вызовы + батч-эмбеддинг + авто-пороги темы + прогресс-бары»,
решения фильтра по сути не меняем. Конкретно:
1. **Параллельная починка кода** (F2.2/F3.2) — `CodeAwareLLMCleaner._handle_code`: серийный
   дешёвый AST-гейт → worklist битых код-чанков → `ThreadPoolExecutor(num_workers)` на сетевые
   `_fix_code` (tqdm `:code-fix`) → серийная валидация. AST-гейт оставлен серийным
   (`warnings.catch_warnings` не thread-safe). Убирает 12-часовую «тишину».
2. **Авто-пороги темы** (F3.1/F3.2) — зашитый `topic_gate: 0.4` УБРАН из кода и конфигов.
   `_TopicMixin._prepare_documents` делает ОДИН батч-пред-проход: батч-эмбед всех доков,
   косинусы к `benchmark_topic`, `_auto_topic_bands` учит ДВА порога из распределения
   (Оцу/Фишер — та же математика, что в `_keep_mask`): нижний (явный офтоп → дроп БЕЗ LLM) и
   верхний (явно своё → keep без LLM); LLM только на средней «неуверенной» полосе, параллельно
   (`num_workers`, tqdm `:topic-llm`). Ширина полосы — безразмерный `band_z` (1.0), не магический
   косинус. На не-бимодальном корпусе (чистый, без офтопа) дроп без LLM не включается (защита).
3. **Дроп офтоп-дока ЦЕЛИКОМ** — раньше тема резала только текст, код утекал в индекс. Теперь
   дроп по теме выкидывает ВСЕ чанки дока (`chunk.doc_id in dropped_docs` в финальной сборке),
   и `_handle_code` не чинит код дропнутых доков.
4. **Прогресс-бары на всех этапах**: `:topic-embed`, `:topic-llm`, `:clean`, `:code-fix`.
5. **Потокобезопасный счётчик токенов** — `token_tracker`: лок в `add()` + `get_active()`;
   воркеры пула «перевзводят» трекер (`initializer=set_active`), иначе токены фильтра терялись.
6. **Ручки в конфигах**: `num_workers: 4`; `band_z` (дефолт в коде); `topic_gate` — опц. ручной
   override (None = авто). Файлы: `src/filtering/{self_clean_v2,code_aware,topic_aware}/main.py`,
   `src/utils/token_tracker.py`, 6 конфигов в `test_configs_experimental_5/filtration/`.

### ⏳ ОСТАЁТСЯ узким местом (замерить на смоуке, кандидат на следующий раунд)
Бар `filter:clean` (базовая текст-чистка L2/L3) сам по себе тяжёлый: L3 семантик-проверка шлёт
~1 запрос эмбеддеру **на строку** последовательно → на 35k грязных доков это бывшие ~4 ч. Эти
правки его НЕ трогают (он output-defining, общий для всех фильтров). Если смоук покажет, что он
всё ещё долгий — следующий output-preserving шаг: распараллелить/сбатчить эмбеддер-вызовы
базового цикла по докам (доки независимы, ключи `(ci,li)` не пересекаются). НЕ делали молча —
сначала мерить и обсудить.

### ⚠️ КРИТИЧНО: почему exp5 встал (диагноз 10.06)
Симптом: конфиг #1 `topic_llm_dirty` за ~18 ч не дошёл до `Vectorizing`/бенча; в логе только
бар `filter: 100% [3:54:24]`, затем 12 ч+ тишины с потоком `SyntaxWarning`. Остановлен вручную.

**Диагноз — два узких места, оба при сборке (`filter.apply`) на ПОЛНОМ корпусе (~35k доков):**
1. **Тема (F3.x) по-документно** ≈ 4 ч: эмбеддит каждый док + зовёт LLM на каждый «дальний»
   док, а дальние — все ~17.8k вики (половина корпуса). Это и есть бар `filter: 100%`.
2. **`_handle_code` (F2.2/F3.2) LLM-починка кода** — 12 ч+ и не закончилось: шлёт в LLM
   **каждый битый чанк кода во всём корпусе, по одному, последовательно, без лимита и без
   прогресс-бара**. Поток `SyntaxWarning` = `ast.parse` в проверке кода (заглушено), сами
   починки идут молча → выглядит как зависание.

**Корень:** LLM-фильтры (тема + починка кода) на build-time по всему корпусу **не масштабируются**
(дни). Прогресс-бар добавлен только на цикл по докам, на `_handle_code` его НЕТ.

**Что уже было и откатили по просьбе Konstantin (били ровно в эти места):** батч-эмбеддинг +
двусторонний порог темы (явный офтоп режется без LLM → F3 за минуты); `max_llm_fixes` (лимит
починки). Откат вернул полный объём LLM-вызовов.

**Опции для обсуждения (новая сессия):**
- (A) вернуть оптимизации: батч-эмбеддинг + двусторонний порог (тема без LLM на явном офтопе)
  + лимит/прогресс на `_handle_code`;
- (B) перенести LLM-фильтры с build-time на **пост-ретрив** (чистить только извлечённый контекст —
  единицы чанков на запрос); это смена архитектуры/зоны — обсудить;
- (C) **параллелить** LLM-вызовы (vLLM тянет конкурентность; сейчас строго последовательно);
- (D) гонять по полному корпусу только дёшевые (pure/f1v2/code_select), а LLM-фильтры — на
  подвыборке/меньшем junk-frac;
- (E) минимум — прогресс-бар на `_handle_code`, чтобы видеть, что идёт.
Konstantin хочет «большое серьёзное обсуждение» — НЕ латать молча, выбрать стратегию вместе.

### Главный разворот — почему семантический RAG
Ушли с API-пайплайна на **семантический RAG над `docs_database_examples.db`**
(`SIMPLE_RETRIEVER` + `CORAG_CONTEXT_ASSEMBLER` + `return_examples: True`, без query-enhancer).
Причина: на API-ретривере (достаёт по имени API) контекст даёт всего **~3%** над голой LLM
(слова коллеги-RAG), и грязь/офтоп **не доезжали до ответа** → R неизмерим (просадка `pure`
≈ 0.012 даже при 75% шума — в пределах шумового пола). На семантическом поиске грязь реально
портит retrieval и контекст → `pure` на грязном просядет → R измерим. (exp4 — архив, см. ниже.)

### Метрика R (единственная) — определение и как читать
Три числа на ОДНОМ корпусе: **(1)** `pure` на чистом (потолок), **(2)** `pure` на грязном (пол),
**(3)** фильтр на грязном. `R = (3 − 2) / (1 − 2)`.
- `R=0` — не помог (застрял на грязном), `R=1` — полностью восстановил, `R>1` — стало чище
  оригинала, `R<0` — навредил.
- Почему так: на грязном эффект фильтра заведомо крупный → выходит за шумовой пол → значимо.
  Это прямая демонстрация «устойчивости к шуму» (требование гранта). По отчёту §3.3.2 —
  эмпирическая оценка (замороженная модель, на задаче, RAG, без дообучения) = наш DS1000 PASS@1.
- **Шумовой пол DS1000 ≈ 1.4%** (~14 задач из 1000); один прогон шумный, разницам < ~1.5%
  не верить (память `noise-floor-rag-pipeline`).
- **ГЛАВНОЕ при чтении:** сначала смотри `pure_clean − pure_dirty`. Если < ~0.02 — грязь слабая,
  знаменатель = шум, R недостоверен → поднять `--noise` и переснять. Если просадка крупная — R
  по фильтрам осмыслен.

### Логика эксперимента
- **1 чистый + 1 грязный** корпус (потом 2+ уровня). Все 6 фильтрационных конфигов (`pure` +
  5 наших) гоняются на **обоих**; `pure_clean`/`pure_dirty` — общие потолок/пол.
- **Что меряем:** насколько каждый фильтр возвращает PASS@1, просевший от грязи (= R).
- **Что какой фильтр должен ловить:** in-doc грязь (mojibake/HTML/`w→vv`/boilerplate) → F1_v2 и выше;
  сломанный код → F2.x; офтоп-доки → F3.x. NB: офтоп при семантическом поиске почти не извлекается,
  так что его вклад в R мал — F3 в основном про чистоту индекса; если захотим мерить удаление офтоп
  честно, нужна отдельная intrinsic-метрика (обсуждали, отложили — Konstantin был против усложнения).
- **Генеративный бейзлайн:** 6 преинференс-генераторов на чистом, без фильтра → ΔPASS@1 vs `pure_clean`.
  Цель — увидеть, что дают существующие методы генерации, как **база для НАШЕГО метода генерации**
  (вторая половина ТЗ 2.1, следующий этап).

### 5 фильтров (детально). Постановка: ОЧИСТИТЕЛЬ, «текст — сам себе эталон». **Код-строки бережём дословно** (примеры вмёрджены в чанки), текст-чистка их не трогает; код обрабатывают F2/F3.
- **F1_v2** `SelfConsistencyCleanerV2` (`src/filtering/self_clean_v2/`) — текст-очиститель под
  реалистичную грязь. Нормализаторы: снятие **HTML**, починка **mojibake** (`cafÃ©`→`café`,
  обратный latin-1→utf-8). Затем три уровня: **L1** убрать мусорные символы в словах; **L2** оставить
  взаимно-связную группу строк (плотность соседей, относительный порог, гашение дословных дублей);
  **L3** срезать чужие слова (буквенная n-грамм-модель по корпусу + семантика leave-one-out против
  ядра L2). Ручки: `strip_html`, `fix_mojibake`, `sep_z`, `char_z`, `semantic_word_check`, `clean_words`,
  `min_doc_lines`. *(Сквозная подмена `w→vv` — трудный остаток: ловится частично L3-A.)*
- **F2.1** `CodeAwareSelectCleaner` (`code_aware/`) — F1_v2 + построчная классификация код/текст;
  **код→дроп чанка, если не парсится** (AST, допуск на REPL `>>>`). Без LLM, универсален (на прозе no-op).
  Ручка: `bad_code_ratio`.
- **F2.2** `CodeAwareLLMCleaner` — F1_v2 + **LLM чинит сломанный код**, получая ВЕСЬ документ как контекст;
  LLM только на плохом коде (дёшево-гейт по AST), не починил → дроп. Ручки: `llm_url`, `llm_model`,
  `bad_code_ratio`, `max_context_chars`.
- **F3.1** `TopicSelectCleaner` (`topic_aware/`) = F2.1 + **шаг 0: режем офтоп-доки**: эмбеддинг-гейт
  (косинус дока к `benchmark_topic` из конфига) → LLM подтверждает только ДАЛЬНИЕ. Ручки: `benchmark_topic`,
  `topic_gate`, `llm_url`, `llm_model`.
- **F3.2** `TopicLLMCleaner` = F2.2 + тот же шаг 0.
- Старый `self_clean` (F1, exp4) сохранён, но в серию НЕ входит.

### Загрязнение — реалистичное (`make_noised_realistic.py`; старый `make_noised_db.py` цел)
Стадия 1 — порча ИСХОДНЫХ доков, `--noise f` = **доля грязных слов** (фикс. пропорции): mojibake /
HTML / сквозная подмена символа (`w→vv`) / boilerplate. Стадия 2 — добавляем офтоп-ДОКИ из Википедии
(`fetch_offtopic.py` → `data/offtopic_docs.jsonl`), `--junk-frac` = доля от числа исходных, **НЕ входит
в f** (они грязь сами по себе). В запущенной серии: `--noise 0.25 --junk-frac 1.0` (вики = половина БД).

### Как проверить результаты (новая сессия)
1. **rsync** результатов на WSL (команда в «Quick reference» ниже).
2. `python compute_R_exp5.py` → таблица + `results/R_exp5/<ts>/{R.json,summary.txt}`:
   - `pure_clean` (потолок), `pure_dirty` (пол) и по каждому фильтру `clean / dirty / R`;
   - блок генерации: PASS@1 каждого генератора и Δ к `pure_clean`.
3. **Как читать:** (а) `pure_clean−pure_dirty` > шумового пола? если нет → грязь слабая, R не верить;
   (б) R по фильтрам — кто значимо > 0 / ближе к 1; сравни F1_v2 < F2.x < F3.x (растёт ли польза);
   (в) **collateral** — сравни `<фильтр>_clean` с `pure_clean`: если фильтр на ЧИСТОМ просел, он режет
   лишнее; (г) генерация — какой метод даёт +Δ.
4. Сырьё: `results/simple_example_<stem>/<ts>/` → `results.csv` (бинарный score на задачу), `summary.txt`,
   `runtime_stats.json` (время/токены/память).

### О чём думать дальше
- Если R измерим и фильтры помогают → добавить **2-3 уровня грязи** (кривая устойчивости) и **затюнить
  ручки** (`char_z`, `topic_gate`, `sep_z`, `bad_code_ratio`).
- Если `pure` не просел → усилить `--noise`/проверить, доезжает ли грязь до контекста (выгрузить
  retrieved context для пары задач).
- **Свой метод генерации** (ТЗ 2.1) — отталкиваясь от генеративного бейзлайна; затем связка
  **генерация→фильтрация** (сгенерить доки → почистить нашим фильтром), мерить ΔPASS@1.
- Если F3/офтоп важен для отчёта, но R его не ловит → вернуть idea про intrinsic-метрику удаления офтоп.

### Команды (сервер):
```bash
python fetch_offtopic.py --n 18000         # где есть интернет → data/offtopic_docs.jsonl
python make_noised_realistic.py --in data/docs_database_examples.db \
    --out data/docs_database_examples_realistic.db \
    --offtopic data/offtopic_docs.jsonl --noise 0.25 --junk-frac 1.0 --seed 7
python smoke_run.py -c test_configs_experimental_5 --max-docs 15   # пред-запуск (рекурсивно, обе подпапки)
# rm -rf data/vdb_exp5   # ТОЛЬКО при пересъёмке (если индексы уже построены прошлым прогоном)
bash run_exp5.sh                                                   # фильтрация → генерация
python compute_R_exp5.py
```
Конфиги генерит `make_exp5_configs.py` (локально, в `.gitignore`).

**Архив:** exp4 — `self_clean` («документ сам себе эталон», 3 уровня) на API-пайплайне с
`make_noised_db.py` (`--noise`, 25/50/75%) и `compute_R_levels.py`. R оказался неизмерим
из-за слабого контекста API-RAG (см. «Главный разворот»).

**Процесс.** Konstantin делает весь git и сервер; Claude только читает файлы и пишет код.
Пуш: `src/filtering/{self_clean_v2,code_aware,topic_aware}/`, `src/pipelines/constants.py`,
`src/db/docs_db/db.py` (добавлен `max_docs` для smoke), `test_configs_experimental_5/`,
`make_noised_realistic.py`, `fetch_offtopic.py`, `smoke_run.py`, `run_exp5.sh`,
`run_ds1000.py`, `compute_R_exp5.py`, `test_self_clean_v2.py`, `CLAUDE.md`, `.gitignore`.
Генераторы (`make_exp*_configs.py`)/планы/индексы/`*.jsonl` — в `.gitignore`.

---

## Environment

This project lives on **two machines** and we keep them in sync by hand:

| Where | What for |
|---|---|
| **WSL (Ubuntu) on Konstantin's laptop** — `/home/constantin2508/code_folder/grant_ac/recipe-mipt` | Where **you** (Claude) edit code, generate Excel reports, run lightweight smoke-tests. WSL is the active workspace. |
| **Remote server `mipt_aigrant_cluster`** (`VizilterK@proxy2.cod.phystech.edu:10209`) at `/home/VizilterK/code/recipe-mipt` | Where **Konstantin** runs the real DS1000 benchmarks (each takes ~40 min/config × N configs ≈ many hours). |

The two are connected via Git (push from WSL → pull on server) and via `rsync`
for results that aren't in git (because `results/` is `.gitignore`d).

### rsync pull command (results from server → WSL)

Konstantin runs this in WSL after a benchmark finishes:

```bash
rsync -avz -e "ssh -p 10209 -i ~/.ssh/mipt/rsa_asap" \
    VizilterK@proxy2.cod.phystech.edu:/home/VizilterK/code/recipe-mipt/results/ \
    /home/constantin2508/code_folder/grant_ac/recipe-mipt/results/
```

(There is a short form using ssh-config alias, but the WSL ssh-config didn't
have the alias as of last session — use the full form above.)

### Workflow loop

1. You edit code / create configs **in WSL**.
2. Konstantin commits + pushes from WSL, pulls on server.
3. Konstantin runs `python run_ds1000.py -c <configs_dir>` on the server.
4. After it finishes Konstantin rsyncs `results/` back to WSL.
5. You read `runtime_stats.json` and `results.csv`, update
   `experiment_data.py`, regenerate `filter_summary.xlsx`.

---

## Grant context (ТЗ) — why this project exists

This repo is the experimental code for **one thematic task** of a large MFTI
research grant (НИР). The grant docs (`ТЗ.pdf`, `Дополнение к ТЗ.pdf`,
`Программа центра.pdf`, `Отчет 2025.pdf`) make the *why* behind every
constraint concrete.

**The НИР**: «Разработка методов создания и обучения цифровых ИИ-ассистентов
профессионального уровня», executed by the **Исследовательский центр агентных
систем ИИ МФТИ** (ИЦ АС ИИ), 1 Jul 2025 – 31 Dec 2026. Deliverables are
experimental-code "компоненты" that plug into the **«Платформа А2.pro»**.
Engineering reqs for the eventual component: Python, Dockerized, JSON I/O,
English text, success probability ≥ 0.5 (3.5.4.1).

**Our slice** is task **2.1**: «Разработка метода фильтрации текстовых данных
**и метода генерации** текстовых данных для последующего формирования
профессиональных навыков» → deliverable: a **«компонент фильтрации и
генерации»** that takes raw text and produces a filtered + augmented
**knowledge base for RAG**.

### Why the "universal filter" constraint exists

The grant's научная новизна is explicitly framed *against* existing methods.
The Программа центра names them — **Self-Instruct, Auto-Instruct, Textbooks
Are All You Need** — and faults them for being «ограничены недостаточной
способностью обрабатывать **широкий спектр задач** и подвержены влиянию
**шумовых данных**». So the contribution must be a filter/generator that is
(a) **universal** across task types, (b) **robust to noise**, (c) works on
arbitrary "целевые задачи / материалы". That is exactly why filters must not
use benchmark metadata: universality *is* the deliverable, not a nicety.

### ТЗ component map ↔ repo `docs/`

The grant's task tree maps cleanly onto the repo's documented methods:

| ТЗ task | Component | Repo docs / code |
|---|---|---|
| 1.2 | RAG (QA-pairs) | `docs/rag/` — CoRAG, RAPTOR, InstructRAG |
| **2.1** | **Фильтрация + генерация** ← **WE ARE HERE** | `src/filtering/`, `docs/filtration/`, `src/generation/`, `docs/agents/generation/` |
| 2.2 | RAG + iCL | `docs/icl/` — FewShot, ICCL, LENS, ICV |
| 2.3 | PRC (Planning-Reasoning-Critic) | `docs/agents/critique/` (CRITIC, Self-Refine, Reflexion, DeCRIM), `docs/agents/pipelines/panel.md` (PANEL) |

### "Генерация" here = synthetic DATA generation, not answer best-of-N

Terminology trap. In task 2.1, "генерация" means **synthetic corpus/data
generation** (Self-Instruct / OSS-Instruct / Code-Evol-Instruct /
error-injection style) — augmenting the RAG corpus with *generated documents*.
The repo's `src/generation/` generators (ZeroShot, OneShot, RandomTopic,
RandomWord, Instruct, CodeEval, IncorrectExample) are exactly these.
**Answer-side** methods (best-of-N, self-consistency, critic-and-revise) are a
*different* ТЗ component — **2.3 PRC** (and partly 2.2) — see "Plans".

### Evaluation is itself a deliverable — and our noise work fills its gap

The 2025 interim report (раздел 3) surveyed evaluation methods and **chose the
empirical / external-on-task approach**: compare a frozen base model's
downstream score with filtered vs unfiltered context in RAG/iCL mode — i.e.
**exactly our DS1000 PASS@1 setup** — picked for iteration speed over costly
train-from-scratch eval. BUT the report never addresses **run-to-run noise**,
and our per-task analysis shows single-shot external eval on DS1000 is
dominated by a **~14-task sampling-noise floor** (see `analyze_significance.py`
and the `noise-floor-rag-pipeline` memory). So the **replication +
McNemar/bootstrap significance protocol** is not just hygiene — it is the
missing rigor in the report's own chosen "способ оценки", hence a legitimate
part of the ТЗ deliverable "перечень подходящих способов оценки". Cheap
diagnostics the report mentions (repetition rate, informativeness coeff.,
KL-reduction, nearest-neighbour semantic distance) can corroborate noisy
empirical numbers.

### Filter novelty (known method vs our heuristic)

The 2025 report's own lit review lists most of our dedup/quality filters as
**existing** methods. Honest accounting (matters for научная новизна +
patent-cleanliness §4, and for citing them in the обзор):

| Our filter | Known method? | Source |
|---|---|---|
| `doc_dedup`, `exact_substr` | **Known** — MinHash/LSH + suffix-array exact dedup | ExactSubstr+NearDup (Lee 2021); standard C4/Pile dedup |
| `semantic_dedup`, `doc_semantic_dedup` | **Known** — embedding near-dup | SemDeDup (Abbas 2023); D4 (Tirumala 2023) |
| `perplexity`, `doc_perplexity` | **Known** — PPL-tail filtering | CCNet (Wenzek 2019) family |
| `education_value` | **Known** — RF on LLM labels | Textbooks Are All You Need (Gunasekar 2023) |
| `tfidf` | Known primitive, our framing | TF-IDF informativeness density |
| `code_density`, `doc_code_density`, `ast_complexity` | **Ours** — AST + code/prose heuristics | no named published equivalent |
| `aggressive_pattern`, `simple_lexical`, `length` | **Ours** — regex / length heuristics | — |
| `hybrid` | **Ours** — composition | doc_dedup ∘ code_density |

Takeaway: the dedup / PPL / semantic filters are **baselines** (cite them,
don't claim novelty). The genuinely novel surface is the **AST/code-aware**
family and the **combined filter+generation pipeline** evaluated under a
rigorous noise-aware protocol.

---

## Project goal & current phase

The big picture: tune a **retrieval-augmented code-generation pipeline** for
the DS1000 benchmark by **improving the document corpus** that feeds
retrieval. We do **not** touch the retriever, the assembler, or the
generation model — only the **filtration step** before / after chunking.

**Constraint**: filters must be universal. They may rely on Python AST
heuristics but **must not** use benchmark-specific metadata
(like `task.library` in DS1000). The same filter has to be deployable on any
code-corpus benchmark. (*Why:* see **Grant context** above — universality and
noise-robustness are the grant's научная новизна.)

**Current state (3 June 2026)**: 4 benchmark runs completed, 15 filter
configurations tested across them. `filter_tfidf` is the nominal PASS@1 leader
at 0.453 — but **per-task significance analysis shows no filter beats `pure`
at p<0.05 in a single run**; the leaderboard sits inside a ~14-task noise
floor. `tfidf` is the only *borderline* signal (p≈0.1). Establishing the noise
floor by replication is the gating next step.

**Next phase**: generation methods (see "Plans"). NB: "generation" splits into
**(2.1) synthetic data generation** — the ТЗ partner to filtration — and
**(2.3) answer-side methods** (best-of-N / self-consistency / critic). Decide
which to pursue; they are different ТЗ components.

---

## Project layout

```
recipe-mipt/
├── CLAUDE.md                            ← this file
├── experiment_data.py                   ← single source of truth (PASS@1 + stats)
├── build_filter_summary_xlsx.py         ← THE report builder — uses experiment_data
├── filter_summary.xlsx                  ← THE current report
├── run_ds1000.py                        ← benchmark runner (writes runtime_stats.json)
│
├── filter_experiments.xlsx              ← archive (20 sheets, exp 1+2 detailed)
├── results_summary.xlsx                 ← archive (very first compact table)
│
├── src/
│   ├── filtering/                       ← all our filters live here
│   │   ├── code_density/                ← post-chunker, AST + code/prose ratio
│   │   ├── semantic_dedup/              ← post-chunker, embedding cosine
│   │   ├── aggressive_pattern/          ← post-chunker, regex (junk/ToC/URLs)
│   │   ├── perplexity/                  ← post-chunker, Qwen-32B logprobs
│   │   ├── ast_complexity/              ← post-chunker, AST nodes/depth/complexity
│   │   ├── tfidf/                       ← post-chunker, TF-IDF density
│   │   ├── doc_dedup/                   ← pre-chunker (Document), MD5+MinHash
│   │   ├── doc_code_density/            ← pre-chunker (Document), code_density
│   │   ├── doc_semantic_dedup/          ← pre-chunker (Document), embeddings
│   │   ├── doc_perplexity/              ← pre-chunker (Document), Qwen-32B
│   │   └── ... (legacy: exactsubstr, lexical_filtration, simple_filters,
│   │       textbooks_are_all_you_need)
│   ├── pipelines/
│   │   ├── constants.py                 ← register every new filter here
│   │   └── templates/
│   │       ├── simple_pipeline.py       ← measures filter_apply_time
│   │       └── simple_with_doc_filter.py← + measures document_filter_apply_time
│   ├── agent_constructor/
│   │   └── filters.py                   ← Filter + DocumentFilter base classes
│   └── utils/
│       └── token_tracker.py             ← monkey-patches openai client, tracks usage
│
├── test_configs_experimental_1/         ← run 3 (02–03 Jun): exp 1
├── test_configs_experimental_2/         ← run 4 (03 Jun): exp 2 — CURRENT BEST
├── test_configs/                        ← run 1 (27 May): historical
├── test_configs_no_rag/                 ← run 2 (28 May): historical
│
└── results/
    └── <config_name>/<YYYYMMDD_HHMMSS>/
        ├── answers.jsonl                ← raw model outputs
        ├── results.csv                  ← per-task score (binary 0/1)
        ├── summary.txt                  ← aggregate scores per library / perturbation
        └── runtime_stats.json           ← what we measure (added in exp 1)
```

---

## What `runtime_stats.json` contains

Written by `run_ds1000.py` after each config. **All metrics we measure:**

```json
{
  "config_name": "simple_example_filter_tfidf",
  "init_time_s":                 34.38,
  "filter_apply_time_s":          3.99,
  "document_filter_apply_time_s": null,
  "total_time_s":                 2472.67,
  "mean_task_time_s":             2.4727,
  "peak_rss_mb":                  9305.6,
  "config_time_s":                2507.53,
  "eval_input_tokens":            1832749,
  "eval_output_tokens":           71167,
  "eval_total_tokens":            1903916,
  "eval_mean_input_tokens":       1832.75,
  "eval_mean_output_tokens":      71.17,
  "eval_mean_total_tokens":       1903.92,
  "eval_llm_calls":               1000,
  "init_input_tokens":            0,
  "init_output_tokens":           0,
  "init_total_tokens":            0,
  "init_llm_calls":               0
}
```

Tokens are captured via a one-time monkey-patch on the OpenAI client
(`src/utils/token_tracker.py`). RSS via a `psutil` background sampler.

---

## The filters (all 15 we've tested)

Filter "where applied" semantics:

- **post-chunker** = applied to `List[Chunk]` after `chunker.chunk(doc)`,
  in `SimplePipeline`. The class extends `Filter`.
- **pre-chunker** = applied to `List[Document]` before `chunker.chunk(doc)`,
  in `SimplePipelineWithDocFilter`. The class extends `DocumentFilter`.

| Filter (our code) | Where | Idea |
|---|---|---|
| `LengthFilter` (`simple_filters`) | post | Drop chunks below N chars |
| `ExactSubstrFiltrator` (`exactsubstr`) | post | Suffix-array byte-level dedup. **Damages chunks**, not recommended |
| `SimpleLexicalFiltrator` (`lexical_filtration`) | post | Regex cleaning of nav-lines, junk patterns, MD5 dedup, sliding window |
| `EducationValueClassifierFilter` (`textbooks_are_all_you_need`) | post | RF on embeddings, 20 LLM-labelled examples ("Textbooks Are All You Need"). Unstable on small training set. |
| `CodeDensityFilter` (`code_density`) | post | code/prose ratio + at least one parseable AST block. **Was leader of exp 1** |
| `SemanticDedupFilter` (`semantic_dedup`) | post | Cosine-similarity dedup of chunks by embedding (Qwen3-Embedding-4B) |
| `AggressivePatternFilter` (`aggressive_pattern`) | post | Aggressive regex: ToC / URL-heavy / non-ASCII heavy. **Hurts results** |
| `PerplexityFilter` (`perplexity`) | post | Per-token Qwen-32B perplexity, drop distribution tails. **Very expensive, weak effect** |
| `ASTComplexityFilter` (`ast_complexity`) | post | AST node count / depth / "interesting" nodes (functions/loops/…). Tighter `min_nodes` than v1, **hurts** |
| `TfIdfInformativenessFilter` (`tfidf`) | post | Mean TF-IDF density, drop low-density quantile. **Current leader (0.453)** |
| `DocumentDedupFilter` (`doc_dedup`) | pre | Exact MD5 + MinHash/LSH on whole documents |
| `DocumentCodeDensityFilter` (`doc_code_density`) | pre | code_density logic at document granularity |
| `DocumentSemanticDedupFilter` (`doc_semantic_dedup`) | pre | Cosine-similarity dedup of whole documents (needs `embedder` dep) |
| `DocumentPerplexityFilter` (`doc_perplexity`) | pre | Perplexity at document level (first N chars). **Doesn't recover the perplexity idea — same weak effect** |
| Hybrid (no new class) | pre + post | `DocumentDedupFilter` (pre) + `CodeDensityFilter` (post) in the same YAML using `SIMPLE_WITH_DOC_FILTER` |

**Quick PASS@1 leaderboard** (best across 4 runs):

1. **filter_tfidf** — 0.453 (run 4 only)
2. filter_code_density — 0.449 (run 3); 0.442 (run 4) — **noisy**
3. filter_doc_dedup — 0.447 (run 3); 0.446 (run 4) — **very stable**
4. filter_hybrid — 0.447 (run 4 only)

Pure baseline ranges 0.430–0.444 across runs (range 0.014 — bigger than most
filter effects!). This is the main lesson: **a single run is noisy**.

---

## The 4 runs we did

| # | Date | Folder | Configs | Pipeline differences |
|---|---|---|---|---|
| 1 | 27.05 | `test_configs/` | pure + 4 historic filters | Full RAG: QueryGenerator + CoRAGContextAssembler |
| 2 | 28.05 | `test_configs_no_rag/` | same 5 configs, `_no_rag` versions | Without QueryGenerator, SimpleContextAssembler; for pure also strips retriever |
| 3 | 02–03.06 | `test_configs_experimental_1/` | 10 configs: full control + 5 new exp-1 filters | Full RAG. **First run with `runtime_stats.json`** |
| 4 | 03.06 | `test_configs_experimental_2/` | 10 configs: 4 controls + 6 new exp-2 filters | Full RAG. **First run with token counters** |

Historical runs 1 and 2 don't have `runtime_stats.json` — they ran before
the instrumentation existed.

---

## The report — `filter_summary.xlsx`

**This is THE deliverable.** It has 7 sheets:

1. **Filters** — table: filter name | where it's applied | what it does | why
2. **Metrics** — table: metric | runtime_stats.json key | what it measures | units
3. **Repro PASS@1** — rows = configs, cols = 4 runs + min/max/range/mean.
   *Best per run column = green*, *worst = red*, our-filter names = bold,
   pure row = blue.
4. **Repro Speed** — same layout, metric = `total_time_s`
5. **Repro Memory** — same layout, metric = `peak_rss_mb`
6. **Repro Tokens** — same layout, metric = `eval_mean_total_tokens`
7. **Latest Run** — for the last run (run 4): rows = 10 configs, cols = all
   9 metrics, with best/worst/baseline highlighting.

### How to update the report after a new run

1. **rsync** the results back to WSL.

2. **Find** the new `runtime_stats.json` files:
   ```bash
   find results -maxdepth 3 -name "runtime_stats.json" -newer ...
   ```

3. **Open `experiment_data.py`** — only this file needs edits.

4. **Add PASS@1** for each config:
   ```python
   # For non-pure configs:
   FILTERS["filter_tfidf"]["runs"]["test_configs_experimental_2"] = 0.453

   # For pure:
   PURE_RUNS["test_configs_experimental_2"] = 0.444
   ```

5. **Paste runtime stats**:
   ```python
   RUN_STATS[("filter_tfidf", "test_configs_experimental_2")] = {
       "init_time_s": 34.38,
       "filter_apply_time_s": 3.99,
       "document_filter_apply_time_s": None,
       "total_time_s": 2472.67,
       "mean_task_time_s": 2.473,
       "peak_rss_mb": 9306,
       "eval_input_tokens": 1832749,
       "eval_output_tokens": 71167,
       "eval_total_tokens": 1903916,
       "eval_mean_input_tokens": 1832.75,
       "eval_mean_output_tokens": 71.17,
       "eval_mean_total_tokens": 1903.92,
   }
   ```

6. **If it's a new run-series**, also add to `RUN_VARIANTS`:
   ```python
   RUN_VARIANTS["test_configs_experimental_3"] = {
       "date": "XX.06.2026",
       "generation":        ("QueryGenerator", [("options", 1)]),
       "retriever":         ("SimpleRetriever", [("top_k", 5)]),
       "context_assembler": ("CoRAGContextAssembler", [("-", "-")]),
   }
   ```
   And in `build_filter_summary_xlsx.py` → `RUN_COLUMNS` add the new column label.

7. **Regenerate**:
   ```bash
   python3 build_filter_summary_xlsx.py
   ```

That's it — the xlsx is rebuilt with the new data.

### How to add a NEW filter

1. Create class under `src/filtering/<your_filter>/main.py` extending
   `Filter` or `DocumentFilter`. **Do NOT use `from __future__ import annotations`**
   in any file with `embedder: Agent` / similar Block deps — the registry's
   introspection breaks on string-typed annotations.

2. Create `src/filtering/<your_filter>/__init__.py` exporting the class.

3. Register in `src/pipelines/constants.py` under `ComponentNames`:
   ```python
   YOUR_NEW_FILTER = "src.filtering.your_filter.YourFilter"
   ```
   For document-level filters, group them under "Document Filters" section.

4. If the filter needs a new pipeline (i.e. it's pre-chunker but
   `SimplePipelineWithDocFilter` doesn't fit), add the pipeline and register
   it in `PipelinesNames`. **Don't forget to add the class import to
   `src/pipelines/templates/__init__.py`** — this caught us twice.

5. Create YAML in `test_configs_experimental_N/`.

6. Add the filter entry to `experiment_data.py:FILTERS`:
   ```python
   "filter_your_name": {
       "filter_class": "YourFilter",
       "filter_params": [("param1", "value1")],
       "pipeline_name": "SimplePipeline",  # or SimplePipelineWithDocFilter
       "runs": { "test_configs_experimental_N": None },  # filled in post-run
   },
   ```
   And to `CFG_ORDER` in `build_filter_summary_xlsx.py`.

7. Add to `OUR_FILTERS` set in `build_filter_summary_xlsx.py` so its name is
   bold in repro sheets.

---

## Dependencies (server-side)

```bash
python -m pip install psutil scikit-learn datasketch
```

- `psutil` — for `peak_rss_mb`
- `scikit-learn` — for `TfIdfInformativenessFilter` and `EducationValueFilter`
- `datasketch` — for MinHash near-dup stage in `DocumentDedupFilter`
  (optional — exact-only mode works without)

OpenAI client (`openai`) and embedder are already installed.

---

## Common gotchas (lessons learned)

1. **`from __future__ import annotations` breaks dependency injection.** The
   pipeline builder reads `inspect.signature` and matches param types
   against `Block` subclasses. With future-annotations, types become
   strings; `embedder: Agent` is no longer recognized as a dep.
   Symptom: `ValueError: Required parameter 'embedder' not found for ...`.
   Fix: don't use it in any filter that takes Block deps.

2. **Forgot to add new pipeline class to `templates/__init__.py`?**
   Symptom: `AttributeError: module 'src.pipelines.templates' has no
   attribute 'YourPipeline'`. Always add new pipelines to `__init__.py`.

3. **Empty config folder = silent exit.** `run_ds1000.py` doesn't warn if
   the configs directory is empty. Always `ls` the folder before running.

4. **Qdrant local mode holds an exclusive lock.** When running multiple
   configs sequentially, the lock from previous config must be released
   before the next one starts. We solved this with `pipeline.close()` in
   the runner's `finally` block.

5. **Run-to-run noise is bigger than most filter effects.** `pure` ranges
   0.430–0.444 (range 0.014) across 4 runs. Don't trust a single-run
   improvement of < 0.015. The proper way is repeated trials with
   different random seeds — we haven't done it yet, that's open work.

---

## Plans — Phase 2: generation methods

We've largely closed the corpus-filtration phase. The next direction is
**generation-side improvements**:

- **Multi-solution + LLM judge** (best-of-N): solver produces N answers,
  a critic (same LLM) picks the best. Expected: +0.02…+0.06.
- **Self-consistency**: sample N answers, vote on most common.
- **Critic-and-revise loops**: the model evaluates its own output and
  optionally re-tries with feedback.

Constraint reminder: we still don't touch the retriever or assembler.

### Open quality questions before Phase 2

- **Reproducibility**: re-run top-3 filters (`tfidf`, `doc_dedup`,
  `code_density`) + `pure` 3 times each to estimate noise floor.
  Without this we can't tell signal from noise on Phase-2 improvements.
- **Hybrid breakdown**: `tfidf + doc_dedup` hasn't been tried — could be
  additive since they remove orthogonal noise.
- **Per-library and per-perturbation PASS@1**: we have raw data in
  `results.csv`. Would be a useful diagnostic sheet — currently absent
  from `filter_summary.xlsx`.

---

## Quick reference

### Run benchmarks on server
```bash
python run_ds1000.py -c test_configs_experimental_N
```

### Pull results to WSL
```bash
rsync -avz -e "ssh -p 10209 -i ~/.ssh/mipt/rsa_asap" \
    VizilterK@proxy2.cod.phystech.edu:/home/VizilterK/code/recipe-mipt/results/ \
    /home/constantin2508/code_folder/grant_ac/recipe-mipt/results/
```

### Regenerate the report
```bash
python3 build_filter_summary_xlsx.py
```

### Inspect what a config produced
```bash
ls results/simple_example_filter_<name>/<latest_timestamp>/
cat results/simple_example_filter_<name>/<latest_timestamp>/runtime_stats.json
```

---

That's the whole story so far. The filter set is explored, the leader board
is in `filter_summary.xlsx → Repro PASS@1`, and the next move is generation-
side methods. Good luck, future-Claude.
