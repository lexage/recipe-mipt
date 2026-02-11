"""
Модуль для сборки датасета из исходного кода GitHub-репозиториев.

Данный модуль интегрирует загрузчик файлов (TorchGitHubLoader) и парсер кода
(CSTCodeParser) и собирает структурированные примеры кода (ExtractedExample)
из конкретных репозиториев для последующей загрузки в БД.
"""

import logging
from typing import List
from pathlib import Path

from src.utils.github_parser.config import ExtractedExample
from src.utils.github_parser.text_file_parsers import MarkdownParser, RSTParser
from src.utils.github_parser.cst_parser import CSTCodeParser
from src.utils.github_parser.github_repo_loader import TorchGitHubLoader


logger = logging.getLogger(__file__)


class DatasetBuilder:
    """
    Класс-координатор процесса создания датасета для загрузки в БД.

    Управляет жизненным циклом процесса: загружает список файлов через loader,
    передает их содержимое в parser и агрегирует полученные данные.

    Attributes:
        loader (TorchGitHubLoader): Клиент для загрузки данных из GitHub.
        parser (CSTCodeParser): Инструмент для анализа синтаксического дерева
            кода.
    """

    def __init__(
        self,
        loader: TorchGitHubLoader,
        code_parser: CSTCodeParser,
        md_parser: MarkdownParser,
        rst_parser: RSTParser,
    ) -> None:
        """
        Инициализирует DatasetBuilder необходимыми компонентами.

        Args:
            loader (TorchGitHubLoader): Объект, отвечающий за скачивание
                файлов из репозитория.
            parser (CSTCodeParser): Объект, отвечающий за извлечение логики из
                скачанных файлов.
        """

        self.loader = loader
        self.code_parser = code_parser
        self.md_parser = md_parser
        self.rst_parser = rst_parser

    def run(
        self, target_repo: str, target_folder: str
    ) -> List[ExtractedExample]:
        """
        Запускает полный цикл сборки датасета из указанного репозитория.

        Метод выполняет обход файлов, для каждого найденного файла
        вызывает парсер и собирает результаты в единый список.

        Args:
            target_repo (str): Полное имя репозитория.
            target_folder (str): Путь к папке внутри репозитория для поиска
                файлов (или конкретный файл).

        Returns:
            List[ExtractedExample]: Список извлеченных примеров кода и
                метаданных.
        """
        all_data: List[ExtractedExample] = []

        files_from_repo = self.loader.repo_walk(
            target_repo,
            target_folder,
        )

        for file in files_from_repo.files_iterator:
            logger.debug("Обрабатываем файл %s", file.path)
            if file.path.endswith(".py") and "examples" not in file.path:
                fetched_examples = self.code_parser.parse(
                    file.content, file.path
                )
            elif file.path.endswith(".md"):
                fetched_examples = self.md_parser.parse(
                    file.content, file.path
                )
            elif file.path.endswith(".rst"):
                fetched_examples = self.rst_parser.parse(
                    file.content, file.path
                )
            # вписать сюда обработку examples
            else:
                logger.error(
                    "Не поддерживается файл формата %s", Path(file.path).suffix
                )
                raise NotImplementedError(
                    f"Не поддерживается файл формата {Path(file.path).suffix}"
                )
            if fetched_examples:
                logger.info(
                    "Получили %s примера (-ов) in %s",
                    len(fetched_examples),
                    file.path,
                )
                all_data.extend(fetched_examples)

        return all_data
