from typing import List, Literal, Optional, Tuple
import re
import logging

import libcst as cst

from src.utils.github_parser.github_parser_config import ExtractedExample


logger = logging.getLogger(__file__)


class DocstringProcessor(cst.CSTVisitor):
    """
    Класс-посетитель. LibCST запускает его методы, когда
    проходит по соответствующим узлам дерева кода.
    """

    def __init__(self) -> None:
        self.fetched_examples: List[ExtractedExample] = []

    def _extract_from_node(
        self,
        node: cst.ClassDef | cst.FunctionDef,
        kind: Literal["function", "class"],
    ) -> None:
        """Этот метод вызывается каждый раз, когда парсер находит класс."""
        parsed_code_and_example = self._process_docstring_text(node)

        if parsed_code_and_example:
            self.fetched_examples.append(
                ExtractedExample(
                    source_object_type=kind,
                    source_object_name=node.name.value,
                    source_object_path="",  # пока пустая, CST не знает о файле
                    task_description=parsed_code_and_example[0],
                    solution_code=parsed_code_and_example[1],
                )
            )

    def _process_docstring_text(
        self, node: cst.ClassDef | cst.FunctionDef
    ) -> Optional[Tuple[str, str]]:
        """Логика разделения текста на Задачу и Решение (через Example)."""
        docstring = node.get_docstring()

        if not docstring:
            return

        if "Example:" not in docstring:
            return

        # делим по слову "Example:"
        parts = re.split(r"\n\s*Example[s]?:\s*\n", docstring, maxsplit=1)
        if len(parts) < 2:
            # если ничего не нашли, пробуем упрощенный вариант "в лоб"
            parts = re.split(r"Example:", docstring, maxsplit=1)

        if len(parts) < 2:
            return None

        raw_description = parts[0]
        raw_example = parts[1]

        # убираем всё, что идет после возможных заголовков Args, Raises и т.д.,
        # которые обычно есть в начале докстринги после описания
        clean_description = (
            raw_description.split("Args:")[0].split("Raises:")[0].strip()
        )

        # чистим код от символов >>>
        code_lines: List[str] = []
        for line in raw_example.split("\n"):
            line = line.strip()
            if line.startswith(">>> ") or line.startswith("... "):
                clean_line = line[4:]
                code_lines.append(clean_line)

        solution_code = "\n".join(code_lines).strip()
        if not solution_code:
            return None

        return clean_description, solution_code

    def visit_ClassDef(self, node: cst.ClassDef) -> None:
        self._extract_from_node(node, kind="class")

    def visit_FunctionDef(self, node: cst.FunctionDef) -> None:
        self._extract_from_node(node, kind="function")


class CSTCodeParser:
    """
    БЛА БЛА
    """

    def parse_python_module(
        self, file_content: bytes, file_path: str
    ) -> Optional[List[ExtractedExample]]:
        if file_path.endswith((".md", ".rst")):
            return
        try:
            tree = cst.parse_module(file_content)
            docstring_processor = DocstringProcessor()
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
