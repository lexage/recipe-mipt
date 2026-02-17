"""
Модуль для парсинга файлов документации репозитория.

Содержит реализации парсеров для форматов Markdown (.md) и
ReStructuredText (.rst). Модуль предоставляет инструменты для:
- очистки содержимого файлов от служебной разметки;
- поиска блоков кода;
- извлечения описания задачи для найденных примеров.
"""

import logging
import re
from typing import List, Optional, Tuple

from src.utils.github_parser.abstract_parser import BaseRepoParser
from src.utils.github_parser.config import ExtractedExample
from src.utils.github_parser.helping_functions import (
    decode_file_content,
)


logger = logging.getLogger(__file__)

# паттерны фильтрации
MARKDOWN_TRASH_PATTERN_1 = r"<div[^>]*>.*?</div>"
MARKDOWN_TRASH_PATTERN_2 = r"# Looking for GPUs\?.*?(?=\n#)"
MARKDOWN_TRASH_PATTERN_3 = r"_{10,}"
MARKDOWN_TRASH_PATTERN_BOLD_ITALIC = r"\*\*|__"  # очистка символов курсива
# или жирного шрифта
MARKDOWN_TRASH_PATTERN_HTML_COMMENTS = r"<!--.*?-->"  # в корневом README.md
# файле есть комментарии вида <!--phmdoctest-mark.skip-->. Эта регулярка их
# удаляет

# паттерны извлечения нужных сущностей из файлов формата Markdown
MARKDOWN_ROOT_README_HEADER_PATTERN = (
    r"^(?:(#+)\s+(.*)|<summary>(.*)</summary>)"
)
MARKDOWN_ROOT_README_CODE_FENCE_PATTERN = r"^\s*```"  # находит начало блока
# кода, которое может быть с отступом
MARKDOWN_CODE_BLOCK_PATTERN = r"\s*```(?:\w+)?\s+(.*?)\s*```"  # находит
# содержимое блока кода
MARKDOWN_DOCTEST_PATTERN = (
    r"(?:>>>|\.\.\.)\s+(.*?)(?=\n(?:>>>|\.\.\.)|\n\s*\n|\Z)"
)
MARKDOWN_EXAMPLE_SECTION_PATTERN = (
    r"\n\s*(?:Example|Usage|Quick\s*Start)"
    r"[s]?:?\s*\n\s*(?:```python\s*\n)?(.*?)(?:```\s*\n)?(?=\n\s*\n|\Z)"
)

# паттерны извлечения нужных сущностей из файлов формата .rst
RST_CODE_BLOCK_PATTERN = (
    r"(?m)"
    r"^\s*\.\.\s+(?:code-block|sourcecode|testcode|plot)[^\n]*\n"
    r"((?:^[\t ]+[^\n]*\n|^[\t ]*\n)+)"
)
RST_DOCTEST_PATTERN = (
    r"(?m)"
    r"(?:^[\t ]*>>>\s+[^\n]*\n"
    r"(?:^[\t ]*(?:>>>|\.\.\.)\s+[^\n]*\n)*)"
)
RST_EXAMPLE_SECTION_PATTERN = (
    r"(?m)" r"^[\t ]*Examples?::\s*\n" r"((?:^[\t ]+[^\n]*\n|^[\t ]*\n)+)"
)


class MarkdownParser(BaseRepoParser):
    """
    Парсер для файлов формата Markdown (.md).

    Наследуется от `BaseRepoParser` и реализует специфичную для Markdown
    логику очистки и извлечения примеров кода.
    """

    def _clean_readme_content(self, content: str) -> str:
        """
        Очищает содержимое README-файла от нежелательных элементов.

        Удаляет HTML-блоки div, специфические секции (например, мусор про GPU),
        длинные разделители подчеркивания.

        Args:
            content (str): Исходное текстовое содержимое файла.

        Returns:
            str: Очищенный текст.
        """
        first_clean = re.sub(
            MARKDOWN_TRASH_PATTERN_1,
            "",
            content,
            flags=re.DOTALL | re.IGNORECASE,
        )
        second_clean = re.sub(
            MARKDOWN_TRASH_PATTERN_2,
            "",
            first_clean,
            flags=re.DOTALL | re.IGNORECASE,
        )
        third_clean = re.sub(MARKDOWN_TRASH_PATTERN_3, "", second_clean)
        return third_clean

    def parse(
        self, file_content: bytes | str, file_path: str
    ) -> Optional[List[ExtractedExample]]:
        """
        Выполняет парсинг файла формата Markdown.

        Декодирует содержимое файла, применяет регулярные выражения для поиска
        блоков кода и секций с примерами, после чего формирует
        структурированные объекты примеров.

        Args:
            file_content (bytes | str): Содержимое файла (массив байтов или
                строка).
            file_path (str): Путь к файлу.

        Returns:
            Optional[List[ExtractedExample]]: Список извлеченных примеров
            или None, если примеров не найдено.
        """
        str_file_content = decode_file_content(file_content, file_path)

        patterns_list = [
            MARKDOWN_CODE_BLOCK_PATTERN,
            MARKDOWN_DOCTEST_PATTERN,
            MARKDOWN_EXAMPLE_SECTION_PATTERN,
        ]
        found_matches = self.process_regex_matches(
            patterns_list, str_file_content
        )
        examples = self.clean_and_extract_task_description(
            found_matches,
            str_file_content,
            file_path,
            "markdown_example",
        )
        return examples if examples else None

    def parse_root_readme(
        self,
        file_content: bytes | str,
        file_path: str,
        root_readme_target_sections: List[str],
    ) -> Optional[List[ExtractedExample]]:
        """
        Специализированный метод для парсинга корневого README файла.

        Разбивает файл на логические секции, ориентируясь на заголовки
        Markdown. Если заголовок секции совпадает с одной из целевых секций
        (`root_readme_target_sections`), из нее извлекаются примеры кода.

        Args:
            file_content (bytes | str): Содержимое файла.
            file_path (str): Путь к файлу.
            root_readme_target_sections (List[str]): Список названий секций,
                из которых необходимо извлекать примеры.

        Returns:
            Optional[List[ExtractedExample]]: Список извлеченных примеров
            или None, если ничего не найдено.
        """

        def process_buffer(title: str, buffer_lines: List[str]) -> None:
            """
            Внутренняя функция для обработки накопленного буфера строк
            одной секции.

            Проверяет, входит ли текущий заголовок секции в список целевых
            секций. Если да, ищет блоки кода внутри буфера, очищает
            предшествующий текст и создает объекты `ExtractedExample`.

            Args:
                title (str): Текущий заголовок секции.
                buffer_lines (List[str]): Список строк, относящихся к секции.

            """
            block_content = "\n".join(buffer_lines)

            # ищем частичное совпадение текущего заголовка с заданными
            target_section = None
            for section in root_readme_target_sections:
                if section.lower() in title.lower():
                    target_section = section
                    break
            # если это не целевая секция - пропускаем
            if not target_section:
                return

            # ищем блоки кода в найденной целевой секции
            logger.info("Обрабатываем целевую секцию: %s", title)
            last_match_end = 0
            code_matches = list(
                re.finditer(
                    MARKDOWN_CODE_BLOCK_PATTERN, block_content, re.DOTALL
                )
            )

            for i, match in enumerate(code_matches, 1):
                raw_code = match.group(1).strip()
                if (
                    len(raw_code) < self.min_code_length_for_analyzing
                    or not raw_code
                ):
                    continue

                preceding_text = block_content[
                    last_match_end: match.start()
                ].strip()
                source_object_name = (
                    f"README_{title.replace(' ', '_').replace(':', '')}_{i}"
                )

                # чистим текст от Markdown-артефактов
                clean_context = re.sub(
                    MARKDOWN_TRASH_PATTERN_BOLD_ITALIC, "", preceding_text
                )
                clean_context = re.sub(r"\n{2,}", "\n", clean_context).strip()
                clean_context = re.sub(
                    MARKDOWN_TRASH_PATTERN_HTML_COMMENTS,
                    "",
                    clean_context,
                    flags=re.DOTALL,
                )

                if clean_context:
                    if (
                        title.lower()
                        in clean_context.lower()[: len(title) + 5]
                    ):
                        task_desc = clean_context
                    else:
                        task_desc = f"{title}\n{clean_context}"
                else:
                    task_desc = title

                examples.append(
                    ExtractedExample(
                        source_object_type="markdown_example",
                        source_object_name=source_object_name,
                        source_object_path=file_path,
                        task_description=task_desc,
                        solution_code=raw_code,
                        metadata_source_code=match.group(0),
                        references="",
                    )
                )

        # начинаем основной блок
        str_file_content = decode_file_content(file_content, file_path)
        examples: List[ExtractedExample] = []

        cleaned_readme_content = self._clean_readme_content(str_file_content)
        lines = cleaned_readme_content.split("\n")

        # инициализируем переменные контекста
        current_main_context_title = "Intro"
        current_section_title = "Intro"

        current_buffer: List[str] = []
        header_pattern = re.compile(MARKDOWN_ROOT_README_HEADER_PATTERN)
        code_fence_pattern = re.compile(
            MARKDOWN_ROOT_README_CODE_FENCE_PATTERN
        )

        in_code_block = False

        # итерируемся по строкам
        for line in lines:
            if code_fence_pattern.match(line):
                in_code_block = not in_code_block
                current_buffer.append(line)
                continue

            # если мы внутри блока кода, просто сохраняем строку и продолжаем
            if in_code_block:
                current_buffer.append(line)
                continue

            header_match = header_pattern.match(line.strip())

            if header_match:
                if current_buffer:
                    process_buffer(current_section_title, current_buffer)
                    current_buffer = []

                if header_match.group(1):
                    raw_title = header_match.group(2)
                    clean_title = re.sub(r"<[^>]+>", "", raw_title).strip()

                    current_main_context_title = clean_title
                    current_section_title = clean_title

                else:
                    raw_title = header_match.group(3)
                    clean_summary = re.sub(r"<[^>]+>", "", raw_title).strip()

                    # это вложенный заголовок, добавляем к контексту.
                    # Результат будет: "Module metrics: Example using DDP"
                    current_section_title = (
                        f"{current_main_context_title}: {clean_summary}"
                    )

            else:
                current_buffer.append(line)

        if current_buffer:
            process_buffer(current_section_title, current_buffer)

        logger.info("Из корневого README извлечено %s примеров", len(examples))
        return examples if examples else None


class RSTParser(BaseRepoParser):
    """
    Парсер для файлов формата ReStructuredText (.rst). Наследуется от
    `BaseRepoParser` и реализует специфичную для .rst
    логику очистки и извлечения примеров кода.

    Attributes:
        rst_skip_code_lines_patterns (Tuple[str, ...]): Кортеж префиксов
            строк, подлежащих удалению при очистке описания задачи.
    """

    def __init__(
        self,
        rst_skip_code_lines_patterns: Tuple[str, ...],
        min_code_length_for_analyzing: int,
    ) -> None:
        """
        Инициализирует парсер RST.

        Args:
            rst_skip_code_lines_patterns (Tuple[str, ...]): Паттерны для
                игнорирования строк.
            min_code_length_for_analyzing (int): Минимальная длина кода
                для анализа.
        """
        super().__init__(min_code_length_for_analyzing)
        self.rst_skip_code_lines_patterns = rst_skip_code_lines_patterns

    def _should_skip_line(self, line: str) -> bool:
        """
        Метод для фильтрации строк описания. Переопределяем логику фильтрации
        специально для файлов формата .rst.

        Returns:
            bool: True, если строку нужно пропустить, иначе False.
        """
        if super()._should_skip_line(line):
            return True

        if line.strip().startswith(self.rst_skip_code_lines_patterns):
            return True

        # фильтруем служебную разметку rst, служащую для вставки
        # картинки/графика.
        # Обычно это выглядит вот так:
        # :scale: 100
        # :include-source: false
        if (
            line.strip().startswith(":")
            and len(line.strip()) > 1
            and line.strip()[1] != " "
        ):
            return True

        return False

    def parse(
        self, file_content: bytes | str, file_path: str
    ) -> Optional[List[ExtractedExample]]:
        """
        Выполняет парсинг файла формата ReStructuredText (.rst).

        Args:
            file_content (bytes | str): Содержимое файла.
            file_path (str): Путь к файлу.

        Returns:
            Optional[List[ExtractedExample]]: Список извлеченных примеров
                или None, если ничего не найдено.
        """
        str_file_content = decode_file_content(file_content, file_path)

        patterns_list = [
            RST_CODE_BLOCK_PATTERN,
            RST_DOCTEST_PATTERN,
            RST_EXAMPLE_SECTION_PATTERN,
        ]

        found_matches = self.process_regex_matches(
            patterns_list, str_file_content
        )
        examples = self.clean_and_extract_task_description(
            found_matches,
            str_file_content,
            file_path,
            "rst_example",
        )

        return examples if examples else None
