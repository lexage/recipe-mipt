import logging
from typing import List, Optional

from src.utils.github_parser.abstract_parser import BaseRepoParser
from src.utils.github_parser.config import ExtractedExample
from src.utils.github_parser.helping_functions import (
    decode_byte_content,
)


logger = logging.getLogger(__file__)


# TODO: убираем testsetup и customcarditem? .. plot::?
# https://github.com/Lightning-AI/torchmetrics/blob/master/docs/source/image/perceptual_path_length.rst
# https://github.com/Lightning-AI/torchmetrics/blob/master/docs/source/pages/lightning.rst
# TODO: сейчас в метаданные записывается весь rst кусок. Обсудить с Лёшей
# насколько это ок


# Блоки кода с python (```python)
MARDOWN_CODE_BLOCK_PATTERN = r"```python\s*\n(.*?)\n```"
# Doctest примеры (>>>)
MARKDOWN_DOCTEST_PATTERN = (
    r"(?:>>>|\.\.\.)\s+(.*?)(?=\n(?:>>>|\.\.\.)|\n\s*\n|\Z)"
)
# Примеры после заголовков типа "Example:" или "Usage:"
MARKDOWN_EXAMPLE_SECTION_PATTERN = r"\n\s*(?:Example|Usage|Quick\s*Start)[s]?:?\s*\n\s*(?:```python\s*\n)?(.*?)(?:```\s*\n)?(?=\n\s*\n|\Z)"

RST_CODE_BLOCK_PATTERN = r"\.\.\s*(?:code-block|sourcecode|testcode)(?:::\s*(?:python|.*?)?)?\s*\n\s*\n(.*?)(?=\n\s*\n|$)"
RST_DOCTEST_PATTERN = r"\n\s*>>>\s+(.*?)(?=\n\s*\n|\Z)"
RST_EXAMPLE_SECTION_PATTERN = r"\n\s*Examples?::\s*\n\s*\n(.*?)(?=\n\s*\w|\Z)"


class MarkdownParser(BaseRepoParser):
    """
    Парсер для Markdown (.md) и reStructuredText (.rst) файлов.
    """

    def parse(
        self, file_content: bytes, file_path: str
    ) -> Optional[List[ExtractedExample]]:
        """
        Парсит Markdown файл и извлекает примеры кода.
        Поддерживает torchmetrics формат.
        """
        try:
            str_file_content = decode_byte_content(file_content, file_path)
        except Exception:
            return None

        patterns_list = [
            MARDOWN_CODE_BLOCK_PATTERN,
            MARKDOWN_DOCTEST_PATTERN,
            MARKDOWN_EXAMPLE_SECTION_PATTERN,
        ]
        found_matches = self.process_matches(patterns_list, str_file_content)
        examples = self.clean_and_extract_task_description(
            found_matches, str_file_content, file_path, "markdown_example"
        )

        return examples if examples else None


class RSTParser(BaseRepoParser):
    """ """

    def parse(
        self, file_content: bytes, file_path: str
    ) -> Optional[List[ExtractedExample]]:
        """
        Парсит RST файл и извлекает примеры кода.
        Поддерживает torchmetrics формат документации.
        """
        try:
            str_file_content = decode_byte_content(file_content, file_path)
        except Exception:
            return None

        patterns_list = [
            RST_CODE_BLOCK_PATTERN,
            RST_DOCTEST_PATTERN,
            RST_EXAMPLE_SECTION_PATTERN,
        ]

        found_matches = self.process_matches(patterns_list, str_file_content)
        examples = self.clean_and_extract_task_description(
            found_matches, str_file_content, file_path, "rst_example"
        )

        return examples if examples else None
