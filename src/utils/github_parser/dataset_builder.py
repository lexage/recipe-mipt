"""
Модуль для сборки датасета из исходного кода GitHub-репозиториев.
"""

import logging
import hashlib
import re
from typing import List, Optional

from src.utils.github_parser.config import ExtractedExample
from src.utils.github_parser.cst_parser import CSTCodeParser
from src.utils.github_parser.github_repo_loader import TorchGitHubLoader
from src.utils.github_parser.text_file_parsers import MarkdownParser, RSTParser


logger = logging.getLogger(__file__)


class DatasetBuilder:
    """
    Класс-координатор процесса создания датасета для загрузки в БД.
    """

    def __init__(
        self, 
        loader: TorchGitHubLoader, 
        code_parser: CSTCodeParser,
        md_parser: MarkdownParser,
        rst_parser: RSTParser
    ) -> None:
        """
        Инициализирует DatasetBuilder необходимыми компонентами.

        Args:
            loader (TorchGitHubLoader): Объект, отвечающий за скачивание
                файлов из репозитория.
            code_parser (CSTCodeParser): Объект для парсинга Python-кода.
            md_parser (MarkdownParser): Объект для парсинга Markdown-файлов.
            rst_parser (RSTParser): Объект для парсинга RST-файлов.
        """
        self.loader = loader
        self.code_parser = code_parser
        self.md_parser = md_parser
        self.rst_parser = rst_parser
        self.files_processed = 0
        self.examples_extracted = 0

    def run(
        self, 
        target_repo: str, 
        target_folder: str,
        extract_from_readme: bool = False,
        readme_sections: List[str] = None
    ) -> List[ExtractedExample]:
        """
        Запускает полный цикл сборки датасета из указанного репозитория.

        Args:
            target_repo (str): Полное имя репозитория.
            target_folder (str): Путь к папке внутри репозитория для поиска
                файлов (или конкретный файл).
            extract_from_readme (bool): Извлекать ли примеры из README.
            readme_sections (List[str]): Секции README для парсинга.

        Returns:
            List[ExtractedExample]: Список извлеченных примеров кода и
                метаданных.
        """
        all_data: List[ExtractedExample] = []
        self.files_processed = 0
        self.examples_extracted = 0

        # Логируем параметры запуска
        if extract_from_readme and readme_sections:
            logger.info(f"Будет выполнен парсинг README по секциям: {readme_sections}")

        files_from_repo = self.loader.repo_walk(
            target_repo,
            target_folder,
        )

        for file in files_from_repo.files_iterator:
            self.files_processed += 1
            logger.debug("Обрабатываем файл %s", file.path)
            
            fetched_examples = None
            
            # Обрабатываем файлы из папки examples особым образом
            if 'example' in file.path.lower() and file.path.endswith('.py'):
                fetched_examples = self._process_example_file(
                    file.content, file.path
                )
            
            # Python файлы (не examples)
            elif file.path.endswith('.py'):
                if hasattr(self, 'code_parser') and self.code_parser:
                    try:
                        fetched_examples = self.code_parser.parse_python_module(
                            file.content, file.path
                        )
                    except Exception as e:
                        logger.error(f"Ошибка при парсинге Python файла {file.path}: {e}")
                else:
                    logger.warning(f"code_parser не найден или не инициализирован")
            
            # Markdown файлы
            elif file.path.endswith('.md'):
                if hasattr(self, 'md_parser') and self.md_parser:
                    try:
                        fetched_examples = self.md_parser.parse_text_file(
                            file.content, file.path
                        )
                    except Exception as e:
                        logger.error(f"Ошибка при парсинге Markdown файла {file.path}: {e}")
                else:
                    logger.warning(f"md_parser не найден или не инициализирован")
            
            # RST файлы
            elif file.path.endswith('.rst'):
                if hasattr(self, 'rst_parser') and self.rst_parser:
                    try:
                        fetched_examples = self.rst_parser.parse_text_file(
                            file.content, file.path
                        )
                    except Exception as e:
                        logger.error(f"Ошибка при парсинге RST файла {file.path}: {e}")
                else:
                    logger.warning(f"rst_parser не найден или не инициализирован")
            
            if fetched_examples:
                self.examples_extracted += len(fetched_examples)
                logger.info(
                    "Получили %s примера (-ов) in %s",
                    len(fetched_examples),
                    file.path,
                )
                all_data.extend(fetched_examples)

        logger.info(f"Обработано файлов: {self.files_processed}")
        logger.info(f"Извлечено примеров: {self.examples_extracted}")
        return all_data
    
    def _process_example_file(
        self, file_content: bytes, file_path: str
    ) -> List[ExtractedExample]:
        """
        Обрабатывает файлы из папки examples.
        Сохраняет каждый файл как один целый документ (без разбиения на чанки).
        """
        try:
            content = file_content.decode('utf-8')
            
            # Если файл небольшой (менее 200 строк), сохраняем целиком
            if len(content.split('\n')) < 200:
                # Генерируем уникальный ID на основе контента
                content_hash = hashlib.md5(content.encode()).hexdigest()[:8]
                filename = file_path.split('/')[-1].replace('.py', '')
                
                return [
                    ExtractedExample(
                        source_object_type="example_file",
                        source_object_name=f"{filename}_{content_hash}",
                        source_object_path=file_path,
                        task_description=f"Файл с примерами использования: {file_path}",
                        solution_code=content,
                        metadata_source_code=content,
                        references=""
                    )
                ]
            
            # Если файл большой, ищем функции и классы
            functions = list(re.finditer(r'def\s+(\w+)\s*\([^)]*\)\s*:', content))
            classes = list(re.finditer(r'class\s+(\w+)\s*(?:\([^)]*\))?\s*:', content))
            
            all_matches = sorted(
                functions + classes,
                key=lambda x: x.start()
            )
            
            if len(all_matches) > 0:
                # Сохраняем каждую функцию/класс как отдельный документ
                examples = []
                for i, match in enumerate(all_matches):
                    start_pos = match.start()
                    end_pos = all_matches[i + 1].start() if i + 1 < len(all_matches) else len(content)
                    
                    block_content = content[start_pos:end_pos].strip()
                    if len(block_content) > 50:  # Минимальная длина блока
                        obj_type = "example_function" if match in functions else "example_class"
                        obj_name = match.group(1)
                        content_hash = hashlib.md5(block_content.encode()).hexdigest()[:8]
                        
                        examples.append(
                            ExtractedExample(
                                source_object_type=obj_type,
                                source_object_name=f"{obj_name}_{content_hash}",
                                source_object_path=file_path,
                                task_description=f"{obj_type.replace('example_', '')} {obj_name} из файла examples",
                                solution_code=block_content,
                                metadata_source_code=content,
                                references=""
                            )
                        )
                return examples if examples else []
            
            # Если не нашли функций/классов, сохраняем целиком как один документ
            content_hash = hashlib.md5(content.encode()).hexdigest()[:8]
            filename = file_path.split('/')[-1].replace('.py', '')
            
            return [
                ExtractedExample(
                    source_object_type="example_file",
                    source_object_name=f"{filename}_{content_hash}",
                    source_object_path=file_path,
                    task_description=f"Файл с примерами использования: {file_path}",
                    solution_code=content,
                    metadata_source_code=content,
                    references=""
                )
            ]
                
        except Exception as e:
            logger.error(f"Ошибка обработки файла examples {file_path}: {e}")
            return []