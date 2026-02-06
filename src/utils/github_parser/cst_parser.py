"""
Модуль для синтаксического анализа кода репозитория.

Содержит инструменты для обхода абстрактного синтаксического дерева (AST),
извлечения докстрингов из классов и функций, а также парсинга разделов
'Examples' для формирования наборов данных.
"""

import logging
import re
from typing import List, Literal, Optional, Tuple

import libcst as cst

from src.utils.github_parser.config import ExtractedExample


logger = logging.getLogger(__file__)


class DocstringProcessor(cst.CSTVisitor):
    """
    Класс-посетитель (Visitor) для обхода AST дерева LibCST.

    Собирает примеры кода из докстрингов классов и функций, встречающихся
    в процессе обхода модуля.

    Attributes:
        fetched_examples (List[ExtractedExample]): Список извлеченных примеров.
        module_node (cst.Module): Корневой узел модуля
            (нужен для получения исходного кода узлов).
    """

    def __init__(self, module_node: cst.Module) -> None:
        """
        Инициализирует класс-посетитель.

        Args:
            module_node (cst.Module): Дерево модуля.
        """
        self.fetched_examples: List[ExtractedExample] = []
        self.module_node = module_node

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
            self.fetched_examples.append(
                ExtractedExample(
                    source_object_type=kind,
                    source_object_name=node.name.value,
                    source_object_path="",  # пока пустая, CST не знает о файле
                    task_description=parsed_code_and_example[0],
                    solution_code=parsed_code_and_example[1],
                    metadata_source_code=node_source_code,
                    references=parsed_code_and_example[-1],
                )
            )

    def _process_docstring_text(
        self, node: cst.ClassDef | cst.FunctionDef
    ) -> Optional[Tuple[str, str, str]]:
        """
        Разбирает докстринг на описание задачи, код решения и ссылки.

        Логика разделения основана на поиске ключевых слов "Example",
        "Examples" или >>>.

        Args:
            node (cst.ClassDef | cst.FunctionDef): Узел, у которого берется
                докстринг.

        Returns:
            Optional[Tuple[str, str, str]]: Кортеж (описание, код, ссылки)
                или None, если пример не найден.
        """
        docstring = node.get_docstring()

        if not docstring:
            return

        if "Example" not in docstring and ">>>" not in docstring:
            return

        references = ""
        raw_code_block = ""

        # 1-я попытка: делим по слову "Example: или "Examples:"
        parts = re.split(r"\n\s*Example[s]?:\s*\n", docstring, maxsplit=1)
        if len(parts) >= 2:
            description = parts[0]
            raw_code_block = parts[1]
        else:
            # 2-я попытка: если ничего не нашли, пробуем упрощенный
            # вариант "в лоб"
            parts_fallback = re.split(r"Example[s]?:", docstring, maxsplit=1)
            if len(parts_fallback) >= 2:
                description = parts_fallback[0]
                raw_code_block = parts_fallback[1]
            else:
                # 3-я попытка: формат со скобками (в репе иногда встречается
                # вот так: Example (preds is int tensor))
                parts = re.split(
                    r"\n\s*Example[s]?\s*\([^)]*\):", docstring, maxsplit=1
                )
                if len(parts) >= 2:
                    description = parts[0]
                    raw_code_block = parts[1]
                    raw_code_block = re.sub(r"^\s+", "", raw_code_block)
                else:
                    # 4-я попытка: если Example нет, ищем >>>
                    if ">>>" in docstring:
                        description, sep, code_part = docstring.partition(
                            ">>>"
                        )
                        raw_code_block = sep + code_part
                    else:
                        # ничего не сработало...
                        return None

        # получаем ссылки на работы авторов (чтобы в дальнейшем их парсить)
        # если нашли - делим, если нет - считаем все примером с кодом
        get_references_and_code: List[str] = re.split(
            r"\n\s*References:\s*\n", raw_code_block, maxsplit=1
        )

        if len(get_references_and_code) == 2:
            raw_code_block = get_references_and_code[0]
            references = get_references_and_code[1]
        elif "References:" in raw_code_block:
            # на всякий случай, если References не выделены переносами строк
            ref_split = raw_code_block.split("References:", 1)
            if len(ref_split) == 2:
                raw_code_block = ref_split[0]
                references = ref_split[1]

        # чистим код от символов >>>
        code_lines: List[str] = []
        for line in raw_code_block.split("\n"):
            line = line.strip()
            if line.startswith(">>> ") or line.startswith("... "):
                clean_line = line[4:]
                code_lines.append(clean_line)
            elif not line:
                if code_lines:
                    code_lines.append("")

        solution_code = "\n".join(code_lines)
        if not solution_code:
            return None

        return description, solution_code, references

    def visit_ClassDef(self, node: cst.ClassDef) -> None:
        """
        Метод cst, вызываемый при обходе дерева и входе в узел определения
        класса.
        """
        self._extract_from_node(node, kind="class")

    def visit_FunctionDef(self, node: cst.FunctionDef) -> None:
        """
        Метод cst, вызываемый при обходе дерева и входе в узел определения
        функции.
        """
        self._extract_from_node(node, kind="function")


class CSTCodeParser:
    """
    Парсер исходного кода Python на базе библиотеки LibCST.

    Отвечает за загрузку байтового содержимого файла, построение AST
    и запуск DocstringProcessor для извлечения примеров.
    """

    def parse_python_module(
        self, file_content: bytes, file_path: str
    ) -> Optional[List[ExtractedExample]]:
        """
        Парсит содержимое файла и извлекает примеры из докстрингов.

        Игнорирует файлы с расширениями .md и .rst.

        Args:
            file_content (bytes): Содержимое файла в байтах.
            file_path (str): Путь к файлу.

        Returns:
            Optional[List[ExtractedExample]]: Список извлеченных примеров
                или None в случае ошибки/игнорирования файла.
        """
        if file_path.endswith((".md", ".rst")):
            return
        try:
            tree = cst.parse_module(file_content)
            logger.debug(
                "Получили следующее дерево модуля по пути %s: %s",
                file_path,
                tree,
            )
            docstring_processor = DocstringProcessor(tree)
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
