from abc import ABC, abstractmethod
import re
from pathlib import Path
from typing import List, Literal, Optional

from src.utils.github_parser.config import ExtractedExample
from src.utils.github_parser.helping_functions import (
    clean_code_lines_from_repl_symbols,
)

# TODO: перенести лог в logs/github_parser!
# TODO: переделать названия некоторых функций, пробежаться по смыслу


class BaseRepoParser(ABC):
    """
    Базовый класс для парсеров, использующихся при обработке файлов из
    репозитория.
    """

    def _extract_task_description(
        self, content: str, start_pos: int, end_pos: int, file_path: str
    ) -> str:
        """
        Извлекает контекст как весь текст между start_pos (конец пред. блока)
        и end_pos (начало текущего блока).
        """
        raw_lines = content[start_pos:end_pos].split("\n")
        cleaned_lines: List[str] = []

        for line in raw_lines:
            # добавить это в конфиг
            if line.strip().startswith(
                (
                    ".. testcode::",
                    ".. code-block::",
                    ".. code::",
                    ".. plot::",
                    ".. testsetup::",
                    ".. _",
                )
            ):
                continue
            # фильтруем служебную разметку rst, служащую для вставки картинки
            if (
                line.strip().startswith(":")
                and len(line.strip()) > 1
                and line.strip()[1] != " "
            ):
                continue
            if line.strip().startswith("import ") or (
                line.strip().startswith("from ") and " import " in line.strip()
            ):
                continue
            cleaned_lines.append(line)

        context = "\n".join(cleaned_lines).strip()

        if not context:
            return f"Пример из {file_path}"

        return context

    def process_matches(
        self,
        patterns: List[str],
        content: str,
    ) -> List[re.Match[str]]:
        """ """
        found_matches: List[re.Match[str]] = []
        unique_matches: List[re.Match[str]] = []

        for pattern in patterns:
            for match in re.finditer(
                pattern, content, re.DOTALL | re.IGNORECASE
            ):
                found_matches.append(match)

        # сортируем по началу вхождения
        found_matches.sort(key=lambda m: m.start())

        # убираем пересечения
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
            "torchmetrics_example",
        ],
    ) -> List[ExtractedExample]:
        examples: List[ExtractedExample] = []
        last_match_end = 0

        for i, match in enumerate(unique_matches, 1):
            code_block = match.group(1).strip()

            cleaned_code = clean_code_lines_from_repl_symbols(
                code_block
            ).strip()

            if len(cleaned_code) < 10:
                last_match_end = match.end()
                continue

            task_description = self._extract_task_description(
                content, last_match_end, match.start(), file_path
            )

            file_name = Path(file_path).suffix.replace(".", "")
            example_name = f"{file_name}_ex_{i}"

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
        """ """

    pass
