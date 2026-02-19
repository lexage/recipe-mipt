import sys
import re
import logging
from pathlib import Path
from typing import List, Tuple, Optional, Any

# Импорт вашей функции (путь оставлен как в оригинале)
project_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(project_root))
from doc_parser.universal_parser import extract_docstrings


class ExampleParser:
    """Парсер для извлечения, замены и очистки примеров кода из документации."""

    _SECTION_PATTERN = re.compile(
        r"^\s*(?:Example|Examples|Some examples)\s*:{0,2}\s*\n"
        r"\s*[-=]{4,}\s*\n"
        r"(.+?)"
        r"(?=\n\s*[A-Z][a-z]+\s*:{0,2}\s*\n|\n\s*[-=]{4,}\s*\n|\Z)",
        re.MULTILINE | re.DOTALL,
    )

    _RST_PATTERN = re.compile(
        r"^(?P<indent> *)\.\. code-block:: python\n"
        r"(?P<content>(?:(?:\n|(?P=indent) +.*\n?)+))",
        re.MULTILINE,
    )

    _MD_PATTERN = re.compile(r"(```(?:python|py)\s*\n.*?\n\s*```)", re.DOTALL)

    def __init__(
        self,
        enable_rst: bool = True,
        enable_md: bool = True,
        enable_repl: bool = True,
        enable_free: bool = True,
        logger: Optional[logging.Logger] = None,
    ):
        """Инициализирует парсер, настраивая стратегии извлечения и логгер."""

        self.logger = logger or logging.getLogger("NullLogger")

        self.enable_free = enable_free
        self.strategies = []
        if enable_rst:
            self.strategies.append(self._find_rst_blocks)
        if enable_md:
            self.strategies.append(self._find_md_blocks)
        if enable_repl:
            self.strategies.append(self._extract_repl_blocks)

    def extract(self, docstring: str) -> List[Tuple[int, int, str]]:
        """Управляет процессом извлечения примеров, исключая пересечения диапазонов."""

        all_examples = []

        for match in self._SECTION_PATTERN.finditer(docstring):

            content = match.group(1).rstrip()
            start_global = match.start(1)
            section_examples = []
            occupied = []

            for strategy_func in self.strategies:
                for seg_start, seg_text in self._get_free_segments(content, occupied):
                    found = strategy_func(seg_text)
                    for s, e, txt in found:
                        occupied.append((seg_start + s, seg_start + e))
                        section_examples.append(
                            (
                                seg_start + s + start_global,
                                seg_start + e + start_global,
                                txt,
                            )
                        )

            if not section_examples and self.enable_free:
                found = self._extract_no_format_code(content)
                for s, e, txt in found:
                    section_examples.append((s + start_global, e + start_global, txt))

            all_examples.extend(section_examples)

        all_examples.sort(key=lambda x: x[0])
        return all_examples

    def replace(
        self, docstring: str, ranges: List[Tuple[int, int, str]]
    ) -> Tuple[str, List[int]]:
        """Заменяет извлеченные блоки кода на плейсхолдеры <example_N>."""

        if not ranges:
            return docstring, []

        sorted_ranges = sorted(ranges, key=lambda x: x[0], reverse=True)
        result = docstring
        replaced_indices = []

        total_examples = len(ranges)

        for i, (start, end, _) in enumerate(sorted_ranges):
            real_idx = total_examples - 1 - i
            placeholder = f"<example_{real_idx}>"
            result = result[:start] + placeholder + result[end:]
            replaced_indices.insert(0, real_idx)

        return result, replaced_indices

    def clean_docstring_examples(
        self, raw_examples: List[Tuple[int, int, str]]
    ) -> List[str]:
        """Очищает список сырых примеров от служебных символов."""
        if not raw_examples:
            return []

        texts = [ex[2] for ex in raw_examples]
        final = []
        for text in texts:
            lines = self._remove_output_is_block(text.split("\n"))
            cleaned_lines = []
            for line in lines:
                line = re.sub(r"#\s*doctest:.*$", "", line)
                line = re.sub(r"#\s*Output:.*$", "", line)
                line = line.replace(">>> ", "")
                if line.strip() == "...":
                    continue
                if line.rstrip():
                    cleaned_lines.append(line.rstrip())
            final.append("\n".join(cleaned_lines))
        return final

    def _find_rst_blocks(self, text: str) -> List[Tuple[int, int, str]]:
        """Находит блоки кода в формате reStructuredText."""
        results = []
        for m in self._RST_PATTERN.finditer(text):
            results.append((m.start(), m.end(), m.group("content").rstrip()))
        return results

    def _find_md_blocks(self, text: str) -> List[Tuple[int, int, str]]:
        """Находит блоки кода в формате Markdown."""
        results = []
        for m in self._MD_PATTERN.finditer(text):
            results.append((m.start(), m.end(), m.group(1).rstrip()))
        return results

    def _extract_repl_blocks(self, text: str) -> List[Tuple[int, int, str]]:
        """Находит блоки кода в формате REPL (интерактивная консоль)."""
        example = []
        first_match_idx = -1
        block_end_idx = 0
        current_pos = 0

        for string in text.split("\n"):
            stripped = string.lstrip()
            line_len = len(string) + 1

            if stripped.startswith((">>>", "...", "#")):
                if first_match_idx == -1:
                    first_non_space = len(string) - len(stripped)
                    first_match_idx = current_pos + first_non_space

                example.append(string)
                block_end_idx = current_pos + len(string)

            current_pos += line_len

        if not example or not any(">>>" in item for item in example):
            return []

        return [(first_match_idx, block_end_idx, "\n".join(example))]

    def _extract_no_format_code(self, text: str) -> List[Tuple[int, int, str]]:
        """Находит блоки кода, выделенные только отступами."""

        strings = text.split("\n")
        if not strings:
            return []

        lines_info = []
        current_pos = 0
        for string in strings:
            stripped = string.lstrip()
            indent = len(string) - len(stripped)
            lines_info.append(
                {
                    "text": string,
                    "indent": indent,
                    "start_pos": current_pos,
                    "end_pos": current_pos + len(string),
                    "is_uppercase": bool(stripped and stripped[0].isupper()),
                    "is_empty": not string.strip(),
                }
            )
            current_pos += len(string) + 1

        uppercase_lines = [line for line in lines_info if line["is_uppercase"]]
        if uppercase_lines:
            indent_counts = {}
            for line in uppercase_lines:
                indent_counts[line["indent"]] = indent_counts.get(line["indent"], 0) + 1
            base_indent = max(indent_counts.items(), key=lambda x: x[1])[0]
        else:
            non_empty = [l for l in lines_info if not l["is_empty"]]
            if not non_empty:
                return []
            base_indent = min(l["indent"] for l in non_empty)

        code_lines = []
        first_match_idx = -1
        last_match_end_idx = 0

        for line in lines_info:
            if line["is_empty"]:
                continue
            if line["indent"] > base_indent and not line["is_uppercase"]:
                if first_match_idx == -1:
                    first_match_idx = line["start_pos"]
                code_lines.append(line["text"])
                last_match_end_idx = line["end_pos"]

        if not code_lines:
            return []

        final_end = (
            last_match_end_idx
            if last_match_end_idx >= current_pos
            else last_match_end_idx + 1
        )
        return [(first_match_idx, final_end, "\n".join(code_lines))]

    def _get_free_segments(
        self, text: str, occupied: List[Tuple[int, int]]
    ) -> List[Tuple[int, str]]:
        """Возвращает участки текста, не занятые ранее найденными блоками."""
        segments = []
        last_end = 0
        for start, end in sorted(occupied):
            if start > last_end:
                segments.append((last_end, text[last_end:start]))
            last_end = max(last_end, end)
        if last_end < len(text):
            segments.append((last_end, text[last_end:]))
        return segments

    def _remove_output_is_block(self, lines: List[str]) -> List[str]:
        """Удаляет строки, содержащие маркеры вывода консоли."""
        result = []
        skip = False
        for line in lines:
            if skip:
                if line.lstrip().startswith("#"):
                    continue
                else:
                    skip = False
                    result.append(line)
                    continue
            if re.search(r"#\s*Output is:", line):
                skip = True
                continue
            result.append(line)
        return result


def setup_logger(log_file: str) -> logging.Logger:
    """Настраивает и возвращает логгер для записи в файл."""
    logger = logging.getLogger("agent")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    logger.propagate = False
    Path(log_file).parent.mkdir(parents=True, exist_ok=True)
    handler = logging.FileHandler(log_file, encoding="utf-8")
    handler.setFormatter(logging.Formatter("%(message)s"))
    logger.addHandler(handler)
    return logger


def main(
    docstrings_objects: List[Any], parser: ExampleParser
) -> Tuple[List[Any], int, int]:
    """
    Выполняет полный цикл обработки списка docstrings.

    Returns:
        Tuple: (список обновленных объектов, кол-во секций с примерами, кол-во найденных примеров)
    """
    valid_sections_count = 0
    examples_found_count = 0

    for doc in docstrings_objects:

        if parser._SECTION_PATTERN.search(doc.string):
            valid_sections_count += 1

        original_text = doc.string

        extracted_blocks = parser.extract(original_text)

        if extracted_blocks:

            examples_found_count += 1

            new_text, _ = parser.replace(original_text, extracted_blocks)
            doc.string = new_text

            parser.logger.info(f"ПРИМЕР {examples_found_count}\n")
            parser.logger.info("_" * 10 + "\n")

            parser.logger.info(f"ИСХОДНАЯ DOCSTRING:\n {original_text}\n\n")
            parser.logger.info("_" * 10 + "\n")

            parser.logger.info(f"ОБНОВЛЕННАЯ DOCSTRING:\n {doc.string}\n\n")
            parser.logger.info("_" * 10 + "\n")
            cleaned_examples = parser.clean_docstring_examples(extracted_blocks)
            doc.examples = cleaned_examples

            parser.logger.info("ИЗВЛЕЧЕННЫЕ ПРИМЕРЫ КОДА:\n\n")
            for idx, example in enumerate(doc.examples):
                parser.logger.info(f"ПРИМЕР КОДА {idx+1}:\n{example}\n\n")
            parser.logger.info("_" * 10 + "\n")

        else:
            doc.examples = []
            # parser.logger.info(f"Не найдены примеры:\n {original_text}\n\n")
            # parser.logger.info("_" * 10 + "\n")

    return docstrings_objects, valid_sections_count, examples_found_count


if __name__ == "__main__":

    main_logger = setup_logger("/workspace/data/extract_example.log")

    library_names = [
        "numpy",
        "pandas",
        "scipy",
        "matplotlib",
        "sklearn",
        "tensorflow",
    ]

    for lib in library_names:
        main_logger.info(f"БИБЛИОТЕКА: {lib}\n")
        main_logger.info("_" * 10 + "\n")

        parser = ExampleParser(
            logger=main_logger,
            enable_rst=True,
            enable_md=True,
            enable_repl=True,
            enable_free=True,
        )

        if lib != "tensorflow":
            parser._SECTION_PATTERN = re.compile(
                r"^\s*(?:Example|Examples|Some examples)\s*:{0,2}\s*\n"
                r"(?:\s*[-=]{4,}\s*\n)?"
                r"(.+?)"
                r"(?=\n\s*[A-Z][a-z]+\s*:{0,2}\s*\n|\n\s*[-=]{4,}\s*\n|\Z)",
                re.MULTILINE | re.DOTALL,
            )

        raw_docs = extract_docstrings(
            f"/workspace/venv/lib/python3.11/site-packages/{lib}"
        )

        processed_docs, stats_sec, stats_ex = main(raw_docs, parser)

        main_logger.info(f"Исходное количество извлеченных docstrins: {len(raw_docs)}")
        main_logger.info(
            f"Количество docstrings, в которых была обнаружена секция/ии с примерами: {stats_sec}"
        )
        main_logger.info(
            f"Количество docstrings, в которых были обнаружены примеры: {stats_ex}"
        )
        main_logger.info("_" * 10 + "\n")
