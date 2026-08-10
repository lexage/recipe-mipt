"""
Модуль для сборки датасета из исходного кода GitHub-репозиториев.

Данный модуль интегрирует загрузчик файлов (TorchGitHubLoader) и парсеры кода
различных форматов (CSTCodeParser, MarkdownParser, RSTParser), собирая
структурированные примеры кода (ExtractedExample) для последующей загрузки в
БД.
"""

import logging
from typing import List
from pathlib import Path

from src.utils.github_parser.config import ExtractedExample
from src.utils.github_parser.cst_parser import CSTCodeParser
from src.utils.github_parser.github_repo_loader import TorchGitHubLoader
from src.utils.github_parser.text_file_parsers import MarkdownParser, RSTParser


logger = logging.getLogger(__file__)


class DatasetBuilder:
    """
    Класс-координатор процесса создания датасета для загрузки в БД.

    Управляет жизненным циклом процесса: загружает список файлов через loader,
    распределяет их содержимое между соответствующими парсерами в зависимости
    от расширения файла и агрегирует полученные данные.

    Attributes:
        loader (TorchGitHubLoader): Клиент для загрузки данных из GitHub.
        parser (CSTCodeParser): Инструмент для анализа синтаксического дерева
            кода.
        md_parser (MarkdownParser): Инструмент для парсинга Markdown-файлов.
        rst_parser (RSTParser): Инструмент для парсинга
            reStructuredText-файлов.
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
            md_parser (MarkdownParser): Парсер для извлечения примеров из .md
                файлов.
            rst_parser (RSTParser): Парсер для извлечения примеров из .rst
                файлов.
        """

        self.loader = loader
        self.code_parser = code_parser
        self.md_parser = md_parser
        self.rst_parser = rst_parser

    def run(
        self,
        target_repo: str,
        target_folder: str,
        extract_examples_from_root_readme: bool,
        root_readme_target_sections: List[str],
    ) -> List[ExtractedExample]:
        """
        Запускает полный цикл сборки датасета из указанного репозитория.

        Метод выполняет обход файлов, для каждого найденного файла
        вызывает соответствующий парсер и собирает результаты в единый список.
        Также отдельно обрабатывает корневой README, если это указано в
            параметрах.

        Args:
            target_repo (str): Полное имя репозитория.
            target_folder (str): Путь к папке внутри репозитория для поиска
                файлов (или конкретный файл).
            extract_examples_from_root_readme (bool): Флаг, указывающий, нужно
                ли извлекать примеры из основного README файла репозитория.
            root_readme_target_sections (List[str]): Список заголовков секций
                в README, из которых нужно извлекать примеры.

        Returns:
            List[ExtractedExample]: Список извлеченных примеров кода и
                метаданных.

        Raises:
            NotImplementedError: Если встретился файл с расширением, для
                которого не реализован парсер.
        """
        all_data: List[ExtractedExample] = []

        # загружаем файлы репозитория
        files_from_repo = self.loader.repo_walk(
            target_repo,
            target_folder,
        )

        # обрабатываем отдельно readme файл
        if extract_examples_from_root_readme:
            logger.debug("Обрабатываем корневой README файл...")
            readme_examples = self.md_parser.parse_root_readme(
                files_from_repo.readme_content,
                files_from_repo.readme_path,
                root_readme_target_sections,
            )
            if readme_examples:
                all_data.extend(readme_examples)
            

        # итерируемся по файлам репозитория
        for file in files_from_repo.files_iterator:
            # Пропускаем README, так как он уже обработан отдельно
            if file.path == files_from_repo.readme_path:
                continue
                
            logger.debug("Обрабатываем файл %s", file.path)
            
            fetched_examples = None
            
            # Python файлы (обрабатываем ВСЕ .py файлы, включая examples!)
            if file.path.endswith(".py"):
                fetched_examples = self.code_parser.parse(
                    file.content, file.path
                )
            
            # Markdown файлы
            elif file.path.endswith(".md"):
                fetched_examples = self.md_parser.parse(
                    file.content, file.path
                )
            
            # RST файлы
            elif file.path.endswith(".rst"):
                fetched_examples = self.rst_parser.parse(
                    file.content, file.path
                )
            
            # Если файл с неподдерживаемым расширением - просто пропускаем
            else:
                logger.debug(
                    "Пропускаем файл с неподдерживаемым расширением: %s", 
                    file.path
                )
                continue
            
            if fetched_examples:
                logger.info(
                    "Получили %s примера (-ов) in %s",
                    len(fetched_examples),
                    file.path,
                )
                all_data.extend(fetched_examples)

        return all_data
"""
Модуль для сборки датасета из исходного кода GitHub-репозиториев.

Данный модуль интегрирует загрузчик файлов (TorchGitHubLoader) и парсеры кода
различных форматов (CSTCodeParser, MarkdownParser, RSTParser), собирая
структурированные примеры кода (ExtractedExample) для последующей загрузки в
БД.
"""

import logging
from typing import List
from pathlib import Path

from src.utils.github_parser.config import ExtractedExample
from src.utils.github_parser.cst_parser import CSTCodeParser
from src.utils.github_parser.github_repo_loader import TorchGitHubLoader
from src.utils.github_parser.text_file_parsers import MarkdownParser, RSTParser


logger = logging.getLogger(__file__)


class DatasetBuilder:
    """
    Класс-координатор процесса создания датасета для загрузки в БД.

    Управляет жизненным циклом процесса: загружает список файлов через loader,
    распределяет их содержимое между соответствующими парсерами в зависимости
    от расширения файла и агрегирует полученные данные.

    Attributes:
        loader (TorchGitHubLoader): Клиент для загрузки данных из GitHub.
        parser (CSTCodeParser): Инструмент для анализа синтаксического дерева
            кода.
        md_parser (MarkdownParser): Инструмент для парсинга Markdown-файлов.
        rst_parser (RSTParser): Инструмент для парсинга
            reStructuredText-файлов.
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
            md_parser (MarkdownParser): Парсер для извлечения примеров из .md
                файлов.
            rst_parser (RSTParser): Парсер для извлечения примеров из .rst
                файлов.
        """

        self.loader = loader
        self.code_parser = code_parser
        self.md_parser = md_parser
        self.rst_parser = rst_parser

    def run(
        self,
        target_repo: str,
        target_folder: str,
        extract_examples_from_root_readme: bool,
        root_readme_target_sections: List[str],
    ) -> List[ExtractedExample]:
        """
        Запускает полный цикл сборки датасета из указанного репозитория.

        Метод выполняет обход файлов, для каждого найденного файла
        вызывает соответствующий парсер и собирает результаты в единый список.
        Также отдельно обрабатывает корневой README, если это указано в
            параметрах.

        Args:
            target_repo (str): Полное имя репозитория.
            target_folder (str): Путь к папке внутри репозитория для поиска
                файлов (или конкретный файл).
            extract_examples_from_root_readme (bool): Флаг, указывающий, нужно
                ли извлекать примеры из основного README файла репозитория.
            root_readme_target_sections (List[str]): Список заголовков секций
                в README, из которых нужно извлекать примеры.

        Returns:
            List[ExtractedExample]: Список извлеченных примеров кода и
                метаданных.

        Raises:
            NotImplementedError: Если встретился файл с расширением, для
                которого не реализован парсер.
        """
        all_data: List[ExtractedExample] = []

        # загружаем файлы репозитория
        files_from_repo = self.loader.repo_walk(
            target_repo,
            target_folder,
        )

        # обрабатываем отдельно readme файл
        if extract_examples_from_root_readme:
            logger.debug("Обрабатываем корневой README файл...")
            readme_examples = self.md_parser.parse_root_readme(
                files_from_repo.readme_content,
                files_from_repo.readme_path,
                root_readme_target_sections,
            )
            if readme_examples:
                all_data.extend(readme_examples)
            

        # итерируемся по файлам репозитория
        for file in files_from_repo.files_iterator:
            # Пропускаем README, так как он уже обработан отдельно
            if file.path == files_from_repo.readme_path:
                continue
                
            logger.debug("Обрабатываем файл %s", file.path)
            
            fetched_examples = None
            
            # Python файлы (обрабатываем ВСЕ .py файлы, включая examples!)
            if file.path.endswith(".py"):
                fetched_examples = self.code_parser.parse(
                    file.content, file.path
                )
            
            # Markdown файлы
            elif file.path.endswith(".md"):
                fetched_examples = self.md_parser.parse(
                    file.content, file.path
                )
            
            # RST файлы
            elif file.path.endswith(".rst"):
                fetched_examples = self.rst_parser.parse(
                    file.content, file.path
                )
            
            # Если файл с неподдерживаемым расширением - просто пропускаем
            else:
                logger.debug(
                    "Пропускаем файл с неподдерживаемым расширением: %s", 
                    file.path
                )
                continue
            
            if fetched_examples:
                logger.info(
                    "Получили %s примера (-ов) in %s",
                    len(fetched_examples),
                    file.path,
                )
                all_data.extend(fetched_examples)

        return all_data