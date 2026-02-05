from dataclasses import dataclass, field
from typing import Optional, List

@dataclass
class AnnotationConfig:
    """Configuration for QuRater annotation of datasets."""
    
    model: str = field(
        default=".src/filtering/qurating/checkpoints-preferences/trained_qurater",
        metadata={"description": "Путь к директории, содержащей веса дообученной модели QuRater."}
    )
    output: str = field(
        default=".src/filtering/qurating/datasets/annotated_data",
        metadata={"description": "Путь к директории, в которую будет сохранен итоговый аннотированный датасет в формате HuggingFace datasets."}
    )
    tokens: int = field(
        default=512,
        metadata={"description": "Максимальная длина текстового сегмента для оценки. Поскольку QuRater обучался на фрагментах длиной до 512 токенов, длинные документы разбиваются на части такого размера, оцениваются, а затем их баллы усредняются для получения рейтинга всего документа."}
    )
    map_batch_size: int = field(
        default=512,
        metadata={"description": "Количество примеров из датасета, которые одновременно загружаются в оперативную память для обработки функцией dataset.map."}
    )
    device_batch_size: int = field(
        default=16,
        metadata={"description": "Размер батча для GPU."}
    )
    num_workers: int = field(
        default=1,
        metadata={"description": "Количество параллельных процессов (CPU), выделяемых для загрузки и предварительной обработки данных перед подачей в модель."}
    )
    text_field: str = field(
        default="text",
        metadata={"description": "Название столбца в данных, который содержит необработанный текст чанка."}
    )
    tokens_field: str = field(
        default="input_ids",
        metadata={"description": "Название столбца в датасете, в котором хранятся токенизированные последовательности."}
    )
    labels: Optional[List[str]] = field(
        default_factory=lambda: [
            "code_clarity_readability", 
            "technical_accuracy_correctness", 
            "educational_value_programming", 
            "practical_utility_applicability"
        ],
        metadata={"description": "Список критериев качества. Порядок имен в этом списке должен строго соответствовать порядку линейных голов в архитектуре модели QuRater. Скрипт автоматически создаст для каждого имени две колонки: оценки отдельных фрагментов (_chunks) и взвешенное среднее по всему документу (_average)."}
    )
    data_files: Optional[List[str]] = field(
        default_factory=list,
        metadata={"description": "Список путей к исходным файлам данных (например, в формате JSONL), которые необходимо аннотировать. В нашем случае не используется."}
    )
    shard: List[int] = field(
        default_factory=lambda: [0, 1],
        metadata={"description": "Параметр для распределенной аннотации больших корпусов. Список вида [index, total] (например, [0, 4]) указывает скрипту обрабатывать только index-ю часть данных из total возможных. Это позволяет запускать процесс параллельно на разных GPU или серверах."}
    )