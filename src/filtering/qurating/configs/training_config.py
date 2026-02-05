from dataclasses import dataclass, field
from typing import List

@dataclass
class QuraterTrainingConfig:
    """Configuration for QuRater model training."""
    
    model: str = field(
        default="princeton-nlp/Sheared-LLaMA-1.3b",
        metadata={"description": "Идентификатор модели, на базе которой будет обучаться QuRater."}
    )
    bsz: int = field(
        default=512,
        metadata={"description": "Общее количество примеров, обрабатываемых за один шаг оптимизации всеми устройствами суммарно."}
    )
    seq: int = field(
        default=16,
        metadata={"description": "Количество последовательностей на один GPU за один проход."}
    )
    lr: float = field(
        default=5e-5,
        metadata={"description": "Скорость обучения."}
    )
    epochs: int = field(
        default=2,
        metadata={"description": "Количество эпох обучения."}
    )
    warmup: float = field(
        default=0.1,
        metadata={"description": "Доля шагов обучения, в течение которых скорость обучения линейно возрастает до целевого значения."}
    )
    confidence: float = field(
        default=0.5,
        metadata={"description": "Порог уверенности, используемый для фильтрации обучающих данных; в обучение попадают только те попарные суждения LLM-судьи, где разница вероятностей между вариантами составляет не менее 0.5."}
    )
    labeltemp: float = field(
        default=1.0,
        metadata={"description": "Температура для масштабирования меток при расчете функции потерь, связанная с вероятностной моделью Брэдли-Терри."}
    )
    label_index: str = field(
        default="all",
        metadata={"description": "Указывает, на каких критериях качества обучаться; значение 'all' активирует многозадачный режим, при котором модель одновременно обучается предсказывать все четыре типа рейтингов через разные линейные головы."}
    )
    suffix: str = field(
        default="",
        metadata={"description": "Технический параметр для идентификации эксперимента и формирования названий выходных файлов и логов."}
    )
    run_name: str = field(
        default="trained_qurater",
        metadata={"description": "Технический параметр для идентификации эксперимента и формирования названий выходных файлов и логов."}
    )
    save_steps: int = field(
        default=200,
        metadata={"description": "Интервал в шагах обучения, через который происходит автоматическое сохранение чекпоинта модели на диск."}
    )
    
    
    config_overrides: str = field(
        default="",
        metadata={"description": "Строка, позволяющая динамически переопределять параметры конфигурации базовой модели без изменения файлов кода."}
    )
    use_fast_tokenizer: bool = field(
        default=False,
        metadata={"description": "Флаг, определяющий использование оптимизированной реализации токенизатора из библиотеки HuggingFace для ускорения подготовки данных."}
    )
    max_length: int = field(
        default=2048,
        metadata={"description": "Лимит длины входной последовательности в токенах."}
    )
    eval_split_size: float = field(
        default=1.0,
        metadata={"description": "Доля данных, выделяемая из общего набора для финальной валидации."}
    )
    eval_split_size_train: float = field(
        default=0.1,
        metadata={"description": "Доля обучающей выборки, которая используется для промежуточной валидации модели в процессе обучения."}
    )
    cache_dir: str = field(
        default=".src/filtering/qurating/cache",
        metadata={"description": "Путь к локальной директории, где будут кэшироваться загруженные модели и обработанные наборы данных."}
    )
    single_label_ablation: int = field(
        default=-1,
        metadata={"description": "Индекс конкретной метки качества (от 0 до 3), используемый для проведения исследований влияния отдельных критериев качества на общую производительность модели. Значение -1 активирует стандартное обучение."}
    )
    
    
    log_level: str = field(
        default="info",
        metadata={"description": "Устанавливает порог важности сообщений для системного логгера (например, info, warning или debug)."}
    )
    logging_steps: int = field(
        default=1,
        metadata={"description": "Частота (в шагах обучения), с которой модель записывает текущие метрики в лог или систему мониторинга."}
    )
    disable_tqdm: bool = field(
        default=True,
        metadata={"description": "Позволяет отключить графические индикаторы прогресса."}
    )
    evaluation_strategy: str = field(
        default="steps",
        metadata={"description": "Стратегия запуска валидации; значение steps указывает модели проводить оценку на проверочном наборе через каждые save_steps шагов обучения."}
    )
    
    
    load_best_model_at_end: bool = field(
        default=True,
        metadata={"description": "Если установлено в True, по завершении процесса обучения скрипт автоматически загрузит веса модели с наилучшими показателями на валидационной выборке, а не просто версию последнего шага."}
    )
    metric_for_best_mode: str = field(
        default="eval_validation_acc",
        metadata={"description": "Целевая метрика для сравнения моделей."}
    )
    greater_is_better: bool = field(
        default=True,
        metadata={"description": "Логический флаг, указывающий, что чем выше значение выбранной метрики, тем лучше модель."}
    )
    dataloader_num_workers: int = field(
        default=2,
        metadata={"description": "Количество параллельных подпроцессов для загрузки данных с диска и их предварительной обработки."}
    )
    overwrite_output_dir: bool = field(
        default=True,
        metadata={"description": "Позволяет скрипту перезаписывать файлы в директории сохранения результатов."}
    )
    remove_unused_columns: bool = field(
        default=False,
        metadata={"description": "Если установлено в False, скрипт не будет автоматически удалять из датасета столбцы, которые не заявлены в сигнатуре метода forward модели."}
    )
    report_to: List[str] = field(
        default_factory=lambda: ["wandb"],
        metadata={"description": "Список систем для визуализации и логирования процесса обучения."}
    )
    
    
    do_train: bool = field(
        default=True,
        metadata={"description": "Флаг, активирующий режим обучения."}
    )
    do_eval: bool = field(
        default=True,
        metadata={"description": "Флаг, активирующий режим валидации."}
    )
    max_grad_norm: float = field(
        default=1.0,
        metadata={"description": "Параметр для ограничения нормы градиента."}
    )
    weight_decay: float = field(
        default=0.1,
        metadata={"description": "Коэффициент L2-регуляризации."}
    )
    
    
    bf16: bool = field(
        default=True,
        metadata={"description": "Активирует использование формата точности bfloat16 во время обучения."}
    )
    bf16_full_eval: bool = field(
        default=True,
        metadata={"description": "Активирует использование формата точности bfloat16 во время валидации."}
    )
    
    
    ddp_find_unused_parameters: bool = field(
        default=False,
        metadata={"description": "Параметр для режима распределенного обучения Distributed Data Parallel (DDP). Значение False отключает проверку на наличие неиспользуемых параметров в графе вычислений перед обратным проходом."}
    )
    fsdp: List[str] = field(
        default_factory=lambda: ["auto_wrap"],
        metadata={"description": "Список настроек для стратегии Fully Sharded Data Parallel (FSDP). Значение ['auto_wrap'] указывает системе автоматически определять, какие блоки модели следует шардировать."}
    )
    
    
    log_confidences: List[float] = field(
        default_factory=lambda: [0.5, 0.8],
        metadata={"description": "Определяет список порогов уверенности, при которых будет рассчитываться точность модели на валидационной выборке."}
    )


@dataclass
class EnvConfig:
    """Environment and infrastructure configuration for QuRater training."""
    
    wandb_project: str = field(
        default="lm-data-selection",
        metadata={"description": "Определяет название проекта в системе Weights & Biases."}
    )
    wandb_mode: str = field(
        default="offline",
        metadata={"description": "Устанавливает режим работы системы логирования. Значение offline означает, что все данные о ходе обучения будут сохраняться только локально на сервере."}
    )
    fsdp_sharding_strategy: str = field(
        default="5",
        metadata={"description": "Задает стратегию шардирования данных в рамках технологии FSDP."}
    )
    fsdp_state_dict_type: str = field(
        default="FULL_STATE_DICT",
        metadata={"description": "Определяет формат сохранения чекпоинта модели при использовании распределенного обучения. Значение FULL_STATE_DICT указывает системе собрать все распределенные части весов со всех GPU и сохранить их в один файл."}
    )