from dataclasses import dataclass, field
from typing import List, Optional

@dataclass
class SelectionConfig:
    """Configuration for data selection"""
    
    inputs: List[str] = field(
        default_factory=lambda: [".src/filtering/qurating/datasets/annotated_data"],
        metadata={"description": "Список путей к аннотированным наборам данных."}
    )
    output: str = field(
        default=".src/filtering/qurating/datasets/selected_data",
        metadata={"description": "Путь к директории, в которую будут сохранены отобранные данные. Скрипт записывает результат в виде нескольких локальных датасетов HuggingFace."}
    )
    
    # Attribute paths
    attributes: List[str] = field(
        default_factory=list,
        metadata={"description": "Технические списки путей или идентификаторов для работы со специфическими атрибутами или метаданными доменов при загрузке данных."}
    )
    domains: List[str] = field(
        default_factory=list,
        metadata={"description": "Технические списки путей или идентификаторов для работы со специфическими атрибутами или метаданными доменов при загрузке данных."}
    )
    metrics: List[str] = field(
        default_factory=list,
        metadata={"description": "Технические списки путей или идентификаторов для работы со специфическими атрибутами или метаданными доменов при загрузке данных."}
    )
    
    # Field names
    seq_len_field: str = field(
        default="input_len",
        metadata={"description": "Название столбца, содержащего количество токенов в каждой записи (например, input_len). Это обязательный параметр, так как система использует его для контроля достижения лимита токенов."}
    )
    metric_field: Optional[List[str]] = field(
        default=None,
        metadata={"description": "Список названий столбцов с рейтингами качества, которые будут использоваться как логиты для вероятностного отбора."}
    )
    reference_field: Optional[str] = field(
        default=None,
        metadata={"description": "Поле, используемое для сравнительного анализа или в качестве эталона."}
    )
    domain_field: Optional[str] = field(
        default=None,
        metadata={"description": "Название столбца с меткой домена (например, «GitHub» или «Wikipedia»). Если этот параметр указан, скрипт будет отбирать данные так, чтобы сохранить исходные пропорции доменов в итоговой выборке."}
    )
    
    # Selection parameters
    tokens: int = field(
        default=5_000_000_000,
        metadata={"description": "Целевой бюджет токенов. Скрипт прекращает отбор, как только суммарное количество токенов в выбранных документах достигает этого значения."}
    )
    temperature: float = field(
        default=1.0,
        metadata={"description": "Температура сэмплирования."}
    )
    sample: bool = field(
        default=False,
        metadata={"description": "Флаг, включающий режим сэмплирования вместо детерминированного выбора лучших документов."}
    )
    normalize: bool = field(
        default=False,
        metadata={"description": "Если включено, скрипт нормализует среднее значение и стандартное отклонение метрики качества по всему набору данных перед началом отбора."}
    )
    select_bottom: bool = field(
        default=False,
        metadata={"description": "Флаг для проведения инвертированного сэмплирования."}
    )
    margin: float = field(
        default=0.1,
        metadata={"description": "Порог уверенности, используемый при фильтрации или расчете вероятностей выбора."}
    )
    tokens_per_shard: int = field(
        default=500_000_000,
        metadata={"description": "Определяет максимальный размер (в токенах) каждого выходного шарда данных для удобства хранения и последующего чтения."}
    )
    shard_suffix: str = field(
        default="",
        metadata={"description": "Суффикс, добавляемый к именам файлов создаваемых шардов."}
    )
    
    # Processing parameters
    json: bool = field(
        default=False,
        metadata={"description": "Флаг, указывающий на использование формата JSON для входных или выходных данных (если не используются стандартные датасеты HuggingFace)."}
    )
    seed: int = field(
        default=42,
        metadata={"description": "Числовое значение для генератора случайных чисел, обеспечивающее воспроизводимость процесса выборки."}
    )
    num_workers: Optional[int] = field(
        default=None,
        metadata={"description": "Количество параллельных процессов CPU для ускорения загрузки и обработки данных."}
    )