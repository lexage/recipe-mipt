"""
Модуль для синтаксического анализа кода репозитория.

Содержит инструменты для обхода абстрактного синтаксического дерева (AST),
извлечения докстрингов из классов и функций, а также парсинга разделов
'Examples' для формирования наборов данных вида "задача-решение".
"""

import logging
import re
from typing import List, Literal, Optional, Tuple

import libcst as cst

from src.utils.github_parser.abstract_parser import BaseRepoParser
from src.utils.github_parser.config import ExtractedExample
from src.utils.github_parser.helping_functions import (
    clean_code_lines_from_repl_symbols_and_doctest_comments,
)


logger = logging.getLogger(__file__)


EXAMPLE_PATTERN = r"(\n\s*(?:(?:Legacy|Legcy)\s+)?Examples?.*?::?(?:\n|$))"
EXAMPLE_PATTERN_FOR_CLEANING_IN_TASK_DESC = r"Examples?|::?"
REFERENCE_PATTERN = r"\n\s*References:"
TRASH_SPHINX_DIRECTIVE_PATTERN = r"\n\s*\.\. plot::.*(?:\n\s+.*|\n\s*$)*"


class DocstringProcessor(cst.CSTVisitor):
    """
    Класс-посетитель (Visitor) для обхода AST дерева LibCST.

    Собирает примеры кода из докстрингов классов и функций, встречающихся
    в процессе обхода модуля.

    Attributes:
        fetched_examples (List[ExtractedExample]): Список извлеченных примеров.
        module_node (cst.Module): Корневой узел модуля
            (нужен для получения исходного кода узлов).
        ignore_internal_functions (bool): Флаг, указывающий, нужно ли
            игнорировать внутренние (приватные) функции.
    """

    def __init__(
        self, module_node: cst.Module, ignore_internal_functions: bool
    ) -> None:
        """
        Инициализирует класс-посетитель.

        Args:
            module_node (cst.Module): Дерево модуля.
            ignore_internal_functions (bool): Игнорировать ли внутренние
                методы/функции.
        """
        self.fetched_examples: List[ExtractedExample] = []
        self.module_node = module_node
        self.ignore_internal_functions = ignore_internal_functions

    def _extract_from_node(
        self,
        node: cst.ClassDef | cst.FunctionDef,
        kind: Literal["function", "class"],
    ) -> None:
        """
        Обрабатывает узел (для наших задач - класс или функцию),
        извлекает и сохраняет пример.

        Args:
            node (cst.ClassDef | cst.FunctionDef): Узел AST.
            kind (Literal["function", "class"]): Тип объекта
                ("function" или "class").
        """
        parsed_code_and_example = self._process_docstring_text(node)
        if parsed_code_and_example:
            node_source_code = self.module_node.code_for_node(node)

            # итерируемся по списку извлеченных примеров
            for description, solution, references in parsed_code_and_example:
                self.fetched_examples.append(
                    ExtractedExample(
                        source_object_type=kind,
                        source_object_name=node.name.value,
                        source_object_path="",
                        task_description=description,
                        solution_code=solution,
                        metadata_source_code=node_source_code,
                        references=references,
                    )
                )

    def _process_docstring_text(
        self,
        node: cst.ClassDef | cst.FunctionDef,
    ) -> Optional[List[Tuple[str, str, str]]]:
        """
        Разбирает докстрингу на описание задачи, код решения и ссылки.
        Ищет ключевые слова "Example", "Examples" или символы REPL (>>>).
        Общее описание до первого примера добавляется к каждому найденному
        примеру.

        Args:
            node (cst.ClassDef | cst.FunctionDef): Узел, у которого берется
                докстринг.

        Returns:
            Optional[List[Tuple[str, str, str]]]]: Список кортежей
                (описание, код, ссылки) или None, если примеры не найдены или
                    узел должен быть проигнорирован.
        """
        # не анализируем внутренние методы, если нам так указывает конфиг
        if self.ignore_internal_functions and node.name.value.startswith("_"):
            return

        docstring = node.get_docstring()

        # поскольку мы понимаем, что в докстринге есть примеры, по ключевым
        # словам или символам REPL, то если их нет, работать нам не с чем
        if not docstring or (
            "Example" not in docstring and ">>>" not in docstring
        ):
            return

        # сначала отделяем ссылки, потому что они, как правило,
        # в конце докстринги
        references = ""
        split_result = re.split(
            REFERENCE_PATTERN, docstring, maxsplit=1, flags=re.IGNORECASE
        )
        docstring_body = split_result[0]
        if len(split_result) > 1:
            references = split_result[1].strip()
        else:
            # если не нашли, оставляем весь докстринг для дальнейшей обработки
            docstring_body = docstring

        example_splitter = re.compile(EXAMPLE_PATTERN)
        parts: List[str] = example_splitter.split(docstring_body)

        # общее описание до первого примера. Включает описание задачи, args,
        # returns и т.д. из докстринги
        common_description = parts[0].strip()
        valid_examples_found: List[Tuple[str, str, str]] = []

        # parts выглядит как: [описание задачи, заголовок1 (чаще всего example)
        # код1, заголовок2, код2 и т.д.]
        for i in range(1, len(parts), 2):
            specific_description = parts[i]
            code_block = parts[i + 1]

            # фильтрация: если в content нет символов REPL,
            # значит, "Example:" был использован просто для описания формата
            # данных (как в src/torchmetrics/text/squad.py)
            if ">>>" not in code_block:
                common_description += specific_description + code_block
            else:
                valid_examples_found.append(
                    (common_description, specific_description, code_block)
                )

        if valid_examples_found:
            results: List[Tuple[str, str, str]] = []
            for common_desc, specific_desc, code_block in valid_examples_found:
                cleaned_header = re.sub(
                    r"(?:(?:Legacy|Legcy)\s+)?Examples?|::?",
                    "",
                    specific_desc,
                    flags=re.IGNORECASE,
                ).strip()
                full_task_description = (
                    common_desc + "\n" + cleaned_header
                ).strip()
                cleaned_description = self._clean_description(
                    full_task_description
                )
                solution_code = (
                    clean_code_lines_from_repl_symbols_and_doctest_comments(
                        code_block
                    )
                )
                if solution_code:
                    results.append(
                        (cleaned_description, solution_code, references)
                    )
            return results

        if ">>>" in docstring_body:
            description, sep, code_part = docstring_body.partition(">>>")

            clean_desc_text = re.sub(
                r"\n\s*(?:(?:Legacy|Legcy)\s+)?Examples?.*::?\s*$",
                "",
                description.strip(),
                flags=re.IGNORECASE,
            )
            cleaned_description = self._clean_description(clean_desc_text)
            solution_code = (
                clean_code_lines_from_repl_symbols_and_doctest_comments(
                    sep + code_part
                )
            )
            if solution_code:
                return [(cleaned_description, solution_code, references)]

    def _clean_description(self, text: str) -> str:
        """Очищает описание от мусора.

        Args:
            text (str): Текст описания, который нужно очистить.

        Returns:
            str: Очищенный текст описания."""
        # чистим от директив сфинкса вида '.. <plot>::' и все
        # последующие строки с отступом
        sphinx_directive_pattern = re.compile(TRASH_SPHINX_DIRECTIVE_PATTERN)
        cleaned_text = sphinx_directive_pattern.sub("", text)
        return cleaned_text.strip()

    def visit_ClassDef(self, node: cst.ClassDef) -> None:
        """
        Метод cst, вызываемый при обходе дерева и входе в узел определения
        класса.

        Args:
            node (cst.ClassDef): Узел класса.
        """
        self._extract_from_node(node, kind="class")

    def visit_FunctionDef(self, node: cst.FunctionDef) -> None:
        """
        Метод cst, вызываемый при обходе дерева и входе в узел определения
        функции.

         Args:
            node (cst.FunctionDef): Узел функции.
        """
        self._extract_from_node(node, kind="function")


class CSTCodeParser(BaseRepoParser):
    """
    Парсер исходного кода Python на базе библиотеки LibCST.

    Отвечает за загрузку байтового содержимого файла, построение AST
    и запуск DocstringProcessor для извлечения примеров.

    Attributes:
        ignore_internal_functions (bool): Флаг игнорирования приватных методов.
    """

    def __init__(
        self,
        ignore_internal_functions: bool,
        min_code_length_for_analyzing: int,
    ) -> None:
        """
        Инициализирует парсер.

        Args:
            ignore_internal_functions (bool): Флаг игнорирования приватных
                методов.
            min_code_length_for_analyzing (int): Минимальная длина строки кода
                для анализа.
        """
        super().__init__(min_code_length_for_analyzing)
        self.ignore_internal_functions = ignore_internal_functions

    def parse(
        self, file_content: bytes, file_path: str
    ) -> Optional[List[ExtractedExample]]:
        """
        Парсит содержимое файла и извлекает примеры из докстрингов.

        Обрабатываются только файлы с расширением .py.

        Args:
            file_content (bytes): Содержимое файла в байтах.
            file_path (str): Путь к файлу.

        Returns:
            Optional[List[ExtractedExample]]: Список извлеченных примеров
                или None в случае ошибки/игнорирования файла.
        """
        if not file_path.endswith(".py"):
            return
        try:
            tree = cst.parse_module(file_content)
            logger.debug(
                "Получили следующее дерево модуля по пути %s: %s",
                file_path,
                tree,
            )
            docstring_processor = DocstringProcessor(
                tree, self.ignore_internal_functions
            )
            tree.visit(docstring_processor)
            results = docstring_processor.fetched_examples
            for example in results:
                example.source_object_path = file_path
            return results
        except cst.ParserSyntaxError:
            logger.warning(
                "Ошибка синтаксиса в файле %s", file_path, exc_info=True
            )
            return
        except Exception:
            logger.error(
                "Ошибка обработки файла %s!", file_path, exc_info=True
            )
            return
