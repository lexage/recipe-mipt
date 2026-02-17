"""
Инструменты для парсинга файлов репозитория и извлечения примеров кода.

Модуль содержит абстрактный базовый класс парсера, предоставляющий
интерфейс и инструментарий для:

- поиска блоков кода с помощью регулярных выражений,
- удаления пересекающихся совпадений,
- очистки кода от REPL-символов и doctest-комментариев,
- извлечения описания задачи перед блоком кода,
- формирования объектов `ExtractedExample`.
"""

from abc import ABC, abstractmethod
from pathlib import Path
import re
from typing import List, Literal, Optional

from src.utils.github_parser.config import ExtractedExample
from src.utils.github_parser.helping_functions import (
    clean_code_lines_from_repl_symbols_and_doctest_comments,
)


class BaseRepoParser(ABC):
    """
    Абстрактный базовый класс для парсеров файлов репозитория.
    Предоставляет общую логику извлечения структурированных примеров
    из исходных файлов.
    """

    def __init__(self, min_code_length_for_analyzing: int) -> None:
        self.min_code_length_for_analyzing = min_code_length_for_analyzing

    def _should_skip_line(self, line: str) -> bool:
        """
        Метод для фильтрации строк описания. Наследники будут переопределять
        этот метод для своей логики.
        Нужен для чистоты кода, чтобы, например, в `MarkdownParser` не попадала
        спефицическая для обработки .rst файлов логика и наоборот.

        Returns:
            bool: True, если строку нужно пропустить, иначе False.
        """
        if line.strip().startswith("from ") and " import " in line.strip():
            return True
        return False

    def _extract_task_description(
        self,
        content: str,
        start_pos: int,
        end_pos: int,
        file_path: str,
    ) -> str:
        """
        Извлекает текстовое описание задачи перед блоком кода.

        Метод получает фрагмент текста между `start_pos`
        (конец предыдущего совпадения) и `end_pos`
        (начало текущего совпадения), после чего выполняет очистку.

        Если после очистки текст пустой, возвращается
        значение-заглушка на основе имени файла.

        Args:
            content (str): Полное текстовое содержимое файла.
            start_pos (int): Начальная позиция извлекаемого фрагмента.
            end_pos (int): Конечная позиция извлекаемого фрагмента.
            file_path (str): Путь к исходному файлу (используется
                для формирования fallback-описания).
            rst_skip_code_lines_patterns (Tuple[str, ...]): Кортеж строковых
                префиксов, по которым определяются строки RST, подлежащие
                    удалению.

        Returns:
            str: Очищенное описание задачи.
        """
        raw_lines = content[start_pos:end_pos].split("\n")
        cleaned_lines: List[str] = []

        for line in raw_lines:
            if self._should_skip_line(line):
                continue
            cleaned_lines.append(line)

        task_description = "\n".join(cleaned_lines).strip()

        if not task_description:
            return f"Пример из {file_path}"

        return task_description

    def process_regex_matches(
        self,
        patterns: List[str],
        content: str,
    ) -> List[re.Match[str]]:
        """
        Находит и фильтрует совпадения регулярных выражений.

        Args:
            patterns (List[str]): Список регулярных выражений для поиска.
            content (str): Полное текстовое содержимое файла.

        Returns:
            List[re.Match[str]]: Список непересекающихся объектов `re.Match`,
            отсортированных по порядку появления в тексте.
        """
        found_matches: List[re.Match[str]] = []
        unique_matches: List[re.Match[str]] = []

        for pattern in patterns:
            for match in re.finditer(
                pattern, content, re.DOTALL | re.IGNORECASE
            ):
                found_matches.append(match)

        # сортируем по началу вхождения
        found_matches.sort(key=lambda m: m.start())

        # убираем пересечения: если старт предыдущего матча
        # совпадает с началом прошлого или меньше его - это пересечение
        last_end = -1
        for match in found_matches:
            if match.start() >= last_end:
                unique_matches.append(match)
                last_end = match.end()
        return unique_matches

    def clean_and_extract_task_description(
        self,
        unique_matches: List[re.Match[str]],
        content: str,
        file_path: str,
        obj_type: Literal[
            "function",
            "class",
            "markdown_example",
            "rst_example",
        ],
    ) -> List[ExtractedExample]:
        """
        Формирует структурированные примеры на основе найденных совпадений.

        Для каждого совпадения выполняются следующие шаги:

        1. Извлекается блок кода из первой группы регулярного выражения.
        2. Код очищается от REPL-символов и doctest-комментариев.
        3. Пропускаются слишком короткие фрагменты кода
           (меньше `min_code_length_for_analyzing`).
        4. Извлекается описание задачи перед блоком кода.
        5. Создается объект `ExtractedExample`.

        Args:
            unique_matches (List[re.Match[str]]): Список непересекающихся
                совпадений.
            content (str): Полное текстовое содержимое файла.
            file_path (str): Путь к исходному файлу.
            obj_type (Literal ["function", "class", "markdown_example",
                "rst_example"]: Тип исходного объекта.

        Returns:
            List[ExtractedExample]: Список объектов `ExtractedExample`,
                содержащих описание задачи и очищенный код решения.
        """
        examples: List[ExtractedExample] = []
        last_match_end = 0

        for i, match in enumerate(unique_matches, 1):
            code_block = match.group(1).strip()

            cleaned_code = (
                clean_code_lines_from_repl_symbols_and_doctest_comments(
                    code_block
                ).strip()
            )

            # чистим небольшие строки кода.
            # Подробнее про то, зачем их чистить, можно посмотреть в конфиге
            # в описании поля 'min_code_length_for_analyzing'
            if len(cleaned_code) < self.min_code_length_for_analyzing:
                last_match_end = match.end()
                continue

            task_description = self._extract_task_description(
                content,
                last_match_end,
                match.start(),
                file_path,
            )

            example_name = f"{Path(file_path).name}_ex_{i}"

            examples.append(
                ExtractedExample(
                    source_object_type=obj_type,
                    source_object_name=example_name,
                    source_object_path=file_path,
                    task_description=task_description,
                    solution_code=cleaned_code,
                    metadata_source_code=content,
                    references="",
                )
            )

            last_match_end = match.end()

        return examples

    @abstractmethod
    def parse(
        self, file_content: bytes, file_path: str
    ) -> Optional[List[ExtractedExample]]:
        """
        Выполняет парсинг файла и извлекает примеры.

        Args:
            file_content (bytes): Содержимое файла в байтах.
            file_path (str): Путь к файлу.

        Returns:
            Optional[List[ExtractedExample]]: Список извлеченных примеров,
                если они найдены, либо None, если формат файла не
                поддерживается или примеры отсутствуют.
        """

    pass
