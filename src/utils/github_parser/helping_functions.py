"""
Вспомогательные функции для обработки содержимого файлов и очистки кода.

Данный модуль содержит набор вспомогательных функций, используемых парсерами
для предварительной обработки данных. Основные возможности включают безопасное
декодирование байтового содержимого файлов в текстовый формат и нормализацию
фрагментов Python-кода (например, удаление префиксов REPL и директив doctest).
"""

import logging
import textwrap
from typing import List


logger = logging.getLogger(__file__)


class InvalidFileContent(Exception):
    pass


def decode_file_content(file_content: bytes | str, file_path: str) -> str:
    """Декодирует байтовое содержимое файла в строку UTF-8 или возвращает его,
        если содержимое уже является строкой.

    Args:
        file_content (bytes | str): Содержимое файла (массив байтов или
            строка).
        file_path (str): Путь к файлу.

    Returns:
        str: Cтроковое содержимое файла, при необходимости декодированное.

    Raises:
        InvalidFileContent: Если содержимое файла не является массивом байтов
            или строкой.
        UnicodeDecodeError: Если декодированное содержимое файла не
            соответствует кодировке utf-8.
    """
    if isinstance(file_content, bytes):
        try:
            str_file_content = file_content.decode("utf-8")
        except UnicodeDecodeError:
            logger.error("Не удалось декодировать файл %s", file_path)
            raise
    elif isinstance(file_content, str):
        str_file_content = file_content
    else:
        raise InvalidFileContent(
            f"Неизвестный тип содержимого файла по пути: {file_path}"
        )
    return str_file_content


def clean_code_lines_from_repl_symbols_and_doctest_comments(
    raw_code_block: str,
) -> str:
    """
    Очищает строки кода от символов REPL и технических комментариев doctest.

    Поддерживает два режима:
    1. REPL-блоки (содержат '>>>'): удаляются префиксы и строки вывода.
    2. Обычные блоки кода: сохраняются все строки, удаляются только
    doctest-комментарии.

    Args:
        raw_code_block (str): Необработанный блок кода, содержащий символы REPL
            и комментарии doctest.

    Returns:
        str: Очищенный многострочный код.
    """
    lines = raw_code_block.split("\n")
    is_repl_block = any(line.lstrip().startswith(">>>") for line in lines)

    cleaned_code_lines: List[str] = []
    # мусорные служебные комментарии для модуля doctest
    doctest_comments = [
        "# doctest: +NORMALIZE_WHITESPACE",
        "# doctest: +ELLIPSIS",
        "# doctest: +SKIP",
    ]

    for line in lines:
        clean_line = line
        for comment in doctest_comments:
            clean_line = clean_line.replace(comment, "")

        if not clean_line.strip():
            if not is_repl_block:
                cleaned_code_lines.append("")
            continue

        left_stripped = clean_line.lstrip()

        if is_repl_block:
            if left_stripped.startswith(">>> ") or left_stripped.startswith(
                "... "
            ):
                cleaned_code_lines.append(left_stripped[4:])
            elif left_stripped.startswith(">>>") or left_stripped.startswith(
                "..."
            ):
                cleaned_code_lines.append(left_stripped[3:])
            else:
                # это строка вывода REPL (например "tensor([1.])"),
                # пропускаем ее
                pass
        else:
            cleaned_code_lines.append(clean_line)

    result = "\n".join(cleaned_code_lines)

    if not is_repl_block:
        result = textwrap.dedent(result)

    return result
