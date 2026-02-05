from typing import Iterator, Optional, Tuple, Literal

from pydantic import BaseModel, Field
import yaml
from github.Repository import Repository


# здесь чет из разряда сус пас для определения дефолтного пути к конфигу
# TODO: докстринги! + порядок импортов


class RepoFile(BaseModel):
    """Представление сырого файла из репозитория."""

    # TODO: написать description для каждого поля
    path: str
    content: bytes


class RepoWalkResult(BaseModel):
    """Представление сырого файла из репозитория."""

    # TODO: написать description для каждого поля
    readme: str
    target_repo: Repository
    files_iterator: Iterator[RepoFile]


class ExtractedExample(BaseModel):
    """Единица данных для датасета (задача + решение)."""

    # сюда добавить путь к файлу!
    source_object_type: Literal["function", "class"]
    source_object_name: str  # Имя класса или функции
    source_object_path: str
    task_description: str
    solution_code: str


class GitHubLoaderConfig(BaseModel):
    allowed_file_extensions: Tuple[str, ...] = Field(
        default=(".py", ".md", ".rst")
    )
    skip_file_patterns: Tuple[str, ...] = Field(default=("init", "test"))


class GitHubExampleFetcherConfig(BaseModel):
    """
    Docstring for GitHubParserConfig
    """

    # TODO: написать description для каждого поля
    repo_for_analyzing: str = Field(default="Lightning-AI/torchmetrics")
    specific_folder_for_analyzing: str = Field(default="")
    config_for_github_loader: GitHubLoaderConfig


def load_config_file(
    file_path: Optional[str] = None,
) -> GitHubExampleFetcherConfig:
    """Бла"""
    if file_path is not None:
        with open(file_path, encoding="utf-8") as f:
            config_data = yaml.safe_load(f)
    else:
        with open("default_github_parser_config.yaml", encoding="utf-8") as f:
            config_data = yaml.safe_load(f)
    return GitHubExampleFetcherConfig(**config_data)
