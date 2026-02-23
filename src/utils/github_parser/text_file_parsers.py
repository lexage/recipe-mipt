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
from pathlib import Path
from typing import List, Literal, Optional, Tuple

from src.utils.github_parser.abstract_parser import BaseRepoParser
from src.utils.github_parser.config import ExtractedExample
from src.utils.github_parser.helping_functions import (
    decode_file_content,
    clean_code_lines_from_repl_symbols_and_doctest_comments,
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
MARKDOWN_ROOT_README_HEADER_PATTERN = r"^(?:(#+)\s+(.*)|<summary>(.*)</summary>)"
MARKDOWN_ROOT_README_CODE_FENCE_PATTERN = r"^\s*```"  # находит начало блока
# кода, которое может быть с отступом
MARKDOWN_CODE_BLOCK_PATTERN = r"\s*```(?:\w+)?\s+(.*?)\s*```"  # находит
# содержимое блока кода
MARKDOWN_DOCTEST_PATTERN = r"(?:>>>|\.\.\.)\s+(.*?)(?=\n(?:>>>|\.\.\.)|\n\s*\n|\Z)"
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
    r"(?m)" r"(?:^[\t ]*>>>\s+[^\n]*\n" r"(?:^[\t ]*(?:>>>|\.\.\.)\s+[^\n]*\n)*)"
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
        found_matches = self.process_regex_matches(patterns_list, str_file_content)
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
                re.finditer(MARKDOWN_CODE_BLOCK_PATTERN, block_content, re.DOTALL)
            )

            for i, match in enumerate(code_matches, 1):
                raw_code = match.group(1).strip()
                if len(raw_code) < self.min_code_length_for_analyzing or not raw_code:
                    continue

                preceding_text = block_content[last_match_end : match.start()].strip()
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
                    if title.lower() in clean_context.lower()[: len(title) + 5]:
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
        code_fence_pattern = re.compile(MARKDOWN_ROOT_README_CODE_FENCE_PATTERN)

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

        # фильтруем строки, состоящие только из разделителей заголовков.
        # Нужно, чтобы убрать мусор вида ***, ===, ---- и т.д.
        if re.match(r"^[-=~*#^]{3,}$", line.strip()):
            return True

        return False

    def _find_start_of_last_rst_section(self, text: str) -> int:
        """
        Ищет начало последнего заголовка в тексте.
        Это помогает отделить описание текущей задачи от пояснений
        предыдущего примера.

        Args:
            text (str): Текст для поиска последнего заголовка.

        Returns:
            int: Начало последнего заголовка в тексте.
        """
        pattern = r"(?m)(?:^[-=~*#^]{3,}\s*\n)?^[^\n]+\n[-=~*#^]{3,}\s*$"

        matches = list(re.finditer(pattern, text))

        if matches:
            return matches[-1].start()

        return 0

    def _clean_text_lines(self, text_chunk: str) -> str:
        """Вспомогательная функция для очистки текста от служебных строк."""
        raw_lines = text_chunk.split("\n")
        cleaned_lines: List[str] = []
        for line in raw_lines:
            if self._should_skip_line(line):
                continue
            cleaned_lines.append(line)
        return "\n".join(cleaned_lines).strip()

    def _extract_task_description(
        self,
        content: str,
        start_pos: int,
        end_pos: int,
        file_path: str,
    ) -> str:
        """
        Переопределенный метод извлечения описания абстрактного парсера.
        Сначала находит "грязный" чанк текста между блоками кода,
        затем ищет в нем последний заголовок RST и обрезает все, что до него,
        затем передает результат на стандартную очистку строк.
        """
        raw_chunk = content[start_pos:end_pos]

        # находим точку, где начинается последний заголовок
        cut_index = self._find_start_of_last_rst_section(raw_chunk)

        # если заголовок найден, описание начинается с него.
        # все, что выше - это пояснения к предыдущему примеру
        relevant_chunk = raw_chunk[cut_index:]

        raw_lines = relevant_chunk.split("\n")
        cleaned_lines: List[str] = []

        for line in raw_lines:
            if self._should_skip_line(line):
                continue
            cleaned_lines.append(line)

        task_description = "\n".join(cleaned_lines).strip()

        if not task_description:
            return f"Пример из {file_path}"

        return task_description

    # TODO: поправить докстрингу! Args и т.д.
    def clean_and_extract_task_description(
        self,
        unique_matches: List[re.Match[str]],
        content: str,
        file_path: str,
        obj_type: Literal["rst_example"],
    ) -> List[ExtractedExample]:
        """
        Переопределенный метод для RST: реализует логику "умного" разделения
        по заголовкам и сохранения контекста предыдущего примера.
        """
        examples: List[ExtractedExample] = []
        last_match_end = 0

        for i, match in enumerate(unique_matches, 1):
            code_block = match.group(1)

            cleaned_code = clean_code_lines_from_repl_symbols_and_doctest_comments(
                code_block
            ).strip()

            if len(cleaned_code) < self.min_code_length_for_analyzing:
                last_match_end = match.end()
                continue

            raw_gap_text = content[last_match_end: match.start()]

            split_idx = self._find_start_of_last_rst_section(raw_gap_text)

            pre_header_text = raw_gap_text[:split_idx]
            post_header_text = raw_gap_text[split_idx:]

            cleaned_pre_text = self._clean_text_lines(pre_header_text)
            cleaned_post_text = self._clean_text_lines(post_header_text)

            task_description = cleaned_post_text
            if i > 1 and examples and cleaned_pre_text:
                examples[-1].solution_code += f'\n\n"""\n{cleaned_pre_text}\n"""'

            elif i == 1:
                # Объединяем введение и описание секции
                full_desc = (cleaned_pre_text + "\n\n" + cleaned_post_text).strip()
                task_description = full_desc if full_desc else f"Пример из {file_path}"

            if not task_description:
                task_description = f"Пример из {file_path}"

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

        found_matches = self.process_regex_matches(patterns_list, str_file_content)
        examples = self.clean_and_extract_task_description(
            found_matches,
            str_file_content,
            file_path,
            "rst_example",
        )

        return examples if examples else None
