from typing import Optional, Tuple
from dataclasses import dataclass, field


@dataclass
class PairwiseComparisonConfig:
    """Configuration for pairwise comparison scoring."""
    
    # Sampling parameters
    num_compare_all: int = field(
        default=2,
        metadata={"description": "Количество чанков, используемых для одного сравнения; значение «2» означает создание классических попарных сравнений."}
    )
    num_examples: int = field(
        default=100,
        metadata={"description": "Общее количество чанков, которое будет выбрано из исходного набора данных для формирования пар."}
    )
    offset: int = field(
        default=0,
        metadata={"description": "Индекс чанка в исходном датасете, с которого начинается выборка (используется для пропуска уже обработанных данных)."}
    )
    num_examples_proportion: Optional[float] = field(
        default=None,
        metadata={"description": "Доля от общего объема датасета (от 0.0 до 1.0), которую нужно обработать вместо фиксированного количества примеров."}
    )
    num_examples_proportion_start: float = field(
        default=0.0,
        metadata={"description": "Начальная точка (в пропорции), с которой начинается выбор данных из датасета."}
    )
    
    # Randomness and model parameters
    seed: int = field(
        default=42,
        metadata={"description": "Фиксированное значение для генератора случайных чисел, обеспечивающее воспроизводимость отбора текстовых фрагментов и ответов LLM."}
    )
    model: str = field(
        default="gpt-3.5-turbo",
        metadata={"description": "Идентификатор модели-судьи (например, gpt-3.5-turbo, gpt-4 или claude-2), которая будет проводить сравнение."}
    )
    generations: int = field(
        default=20,
        metadata={"description": "Количество независимых ответов модели для каждой пары. Результаты усредняются для вычисления уверенности модели в своем выборе."}
    )
    
    # Prompting parameters
    template: str = field(
        default="",
        metadata={"description": "Путь к файлу шаблона промпта, который содержит критерий качества и инструкции для модели-судьи."}
    )
    labels: Tuple[str, str] = field(
        default=("A", "B"),
        metadata={"description": "Метки вариантов ответа (обычно «A» и «B»), которые LLM должна вернуть в качестве результата сравнения."}
    )
    system_prompt: str = field(
        default="You are a helpful assistant.",
        metadata={"description": "Системное сообщение для модели. Авторы обнаружили, что стандартное 'You are a helpful assistant.' работает лучше всего."}
    )
    
    # Text processing parameters
    text_field: str = field(
        default="text",
        metadata={"description": "Название столбца в датасете, в котором хранится сырой текст чанка."}
    )
    token_field: str = field(
        default="input_ids",
        metadata={"description": "Название поля, содержащего токенизированную версию текста."}
    )
    tokens_min: int = field(
        default=256,
        metadata={"description": "Минимальная длина фрагментов, извлекаемых из чанков. Это необходимо, так как LLM-судьи работают стабильнее на коротких отрывках."}
    )
    tokens_max: int = field(
        default=512,
        metadata={"description": "Максимальная длина фрагментов, извлекаемых из чанков. Это необходимо, так как LLM-судьи работают стабильнее на коротких отрывках."}
    )
    probability_tokens_max: float = field(
        default=0.5,
        metadata={"description": "Вероятность, с которой скрипт выберет фрагмент именно максимальной длины. В QuRating фрагменты фиксированной длины 512 выбирались в половине случаев (0.5), а в остальных — случайно из диапазона."}
    )
    tokenizer: str = field(
        default="meta-llama/Llama-2-7b-hf",
        metadata={"description": "Идентификатор токенизатора (обычно на базе Llama), используемого для корректного разделения текста на фрагменты нужной длины."}
    )
    
    # Output format
    flat_output_format: bool = field(
        default=False,
        metadata={"description": "Флаг, определяющий структуру выходного файла. Если True, результаты сохраняются в упрощенном (плоском) виде, удобном для последующего анализа."}
    )