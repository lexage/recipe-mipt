"""
Данный модуль определяет структуры полученных данных в результате обхода
GitHub-репозитория
"""

from pathlib import Path
from typing import Iterator, Literal, Optional, Tuple

from github.Repository import Repository
from pydantic import BaseModel, Field
import yaml


# определяем абсолютный путь модуля (нужно для конфига)
github_fetcher_module_path = Path(__file__).resolve().parent
DEFAULT_CONFIG_NAME = "default_github_fetcher_config.yaml"


class RepoFile(BaseModel):
    """
    Определяет структуру сырого файла, полученного из GitHub-репозитория
    """

    path: str = Field(
        description="Полный путь к файлу относительно корня репозитория"
    )
    content: bytes = Field(description="Сырое содержимое файла в байтах")


class RepoWalkResult(BaseModel):
    """
    Определяет результат обхода репозитория
    """

    readme: str = Field(description="Текстовое содержимое файла README")
    target_repo: Repository = Field(
        description="Анализируемый объект репозитория GitHub"
    )
    files_iterator: Iterator[RepoFile] = Field(
        description="Итератор, возвращающий объекты типа RepoFile для "
        "обработки большого количества файлов в репозитории"
    )
    # нужно, чтобы Pydantic не выдавал ошибку генерации на неизвестных ему
    # типах
    model_config = {"arbitrary_types_allowed": True}


class ExtractedExample(BaseModel):
    """
    Определяет структуру данных для загрузки в БД
    """

    source_object_type: Literal["function", "class"] = Field(
        description="Тип исходного объекта: функция или класс"
    )
    source_object_name: str = Field(
        description="Имя извлеченного класса или функции"
    )
    source_object_path: str = Field(
        description="Путь к файлу, откуда был извлечен объект"
    )
    task_description: str = Field(
        description="Полученное из докстринги объекта описание задачи "
        "(часть докстринги до раздела Examples)"
    )
    solution_code: str = Field(
        description="Полученное из докстринги объекта решение задачи "
        "(часть докстринги после раздела Examples)"
    )
    metadata_source_code: str = Field(
        description="Исходный код всего объекта без изменений"
        # добавлено по предложению Лёши. Возможно, нам по каким-то причинам
        # потребуется лезть в исходный код
    )
    references: str = Field(
        description="Ссылки на научные работы, на которых основан объект. "
        "Если их нет, пустая строка"
    )
    # добавлено по предложению Лёши. В будущем предполагается определить
    # логику извлечения ссылок на статьи и их парсинга


class GitHubLoaderConfig(BaseModel):
    """
    Определяет конфигурацию класса-загрузчика файлов из репозитория GitHub
    """

    allowed_file_extensions: Tuple[str, ...] = Field(
        default=(".py", ".md", ".rst"),
        description="Список расширений файлов, которые необходимо "
        "обрабатывать. На текущий момент это питоновские и текстовые файлы",
    )
    skip_file_patterns: Tuple[str, ...] = Field(
        default=("init", "test"),
        description="Список подстрок. Если они встречаются в пути файла, файл "
        "пропускается. На текущий момент это инициализационные и тестовые "
        "файлы, поскольку в них обычно нет явных примеров",
    )  # TODO: в будущем на подумать: Лёша предложил проанализировать и как-то
    # обработать тесты, т.к. в них могут быть примеры использования


class GitHubExampleFetcherConfig(BaseModel):
    """
    Основная конфигурация для загрузки и обработки примеров с
    GitHub-репозитория
    """

    repo_for_analyzing: str = Field(
        default="Lightning-AI/torchmetrics",
        description="Идентификатор репозитория в формате 'владелец/название'",
    )
    specific_folder_for_analyzing: str = Field(
        default="",
        description="Относительный путь к конкретной папке/файлу для анализа "
        "(если пустая строка - анализируется весь репозиторий)",
    )
    config_for_github_loader: GitHubLoaderConfig = Field(
        default=GitHubLoaderConfig(),
        description="Конфигурация загрузчика файлов репозитория",
    )


def load_config_file(
    file_path: Optional[str] = None,
) -> GitHubExampleFetcherConfig:
    """
    Загружает конфигурацию из YAML-файла.

    Если путь не передан, пытается загрузить конфигурацию по умолчанию
    из папки, где находится данный модуль.

    Args:
        file_path (Optional[str]): Путь к YAML-файлу конфигурации.

    Returns:
        GitHubExampleFetcherConfig: Валидированный объект конфигурации.
    """
    if file_path is not None:
        with open(file_path, encoding="utf-8") as f:
            config_data = yaml.safe_load(f)
    else:
        with open(
            Path(github_fetcher_module_path, DEFAULT_CONFIG_NAME),
            encoding="utf-8",
        ) as f:
            config_data = yaml.safe_load(f)
    return GitHubExampleFetcherConfig(**config_data)
