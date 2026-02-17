"""
Вспомогательные функции для обработки содержимого файлов и очистки кода.

Данный модуль содержит набор утилит, используемых парсерами для предварительной
обработки данных. Основные возможности включают безопасное декодирование
байтового содержимого файлов в текстовый формат и нормализацию фрагментов
Python-кода (удаление префиксов REPL и технических директив doctest).
"""

import logging
from typing import List


logger = logging.getLogger(__file__)


class InvalidFileContent(Exception):
    pass


def decode_file_content(file_content: bytes | str, file_path: str) -> str:
    """Декодирует байтовое содержимое файла в строку UTF-8 или возвращает его,
        если содержимое уже является строкой.

    Args:
        file_content (bytes): Содержимое файла в байтах.
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

    Функция удаляет префиксы >>> и ... в начале строк, а также вырезает
    служебные директивы модуля doctest, чтобы оставить только
    чистый код.

    Args:
        raw_code_block (str): Необработанный блок кода, содержащий символы REPL
            и комментарии doctest.

    Returns:
        str: Очищенный многострочный код.
    """
    cleaned_code_lines: List[str] = []
    # мусорные служебные комментарии для модуля doctest
    doctest_comments = [
        "# doctest: +NORMALIZE_WHITESPACE",
        "# doctest: +ELLIPSIS",
        "# doctest: +SKIP",
    ]

    for line in raw_code_block.split("\n"):
        # убираем пробелы слева для проверки префикса,
        # но сохраняем оригинал для корректного среза
        left_stripped_line = line.lstrip()
        clean_line = line.rstrip()
        is_repl_line = False

        if left_stripped_line.startswith(
            ">>> "
        ) or left_stripped_line.startswith("... "):
            clean_line = left_stripped_line[4:]
            is_repl_line = True
        elif left_stripped_line.startswith(
            ">>>"
        ) or left_stripped_line.startswith("..."):
            clean_line = left_stripped_line[3:]
            is_repl_line = True

        if is_repl_line:
            for comment in doctest_comments:
                if comment in clean_line:
                    clean_line = clean_line.replace(comment, "")

        # if clean_line:
        #     cleaned_code_lines.append(clean_line)
            if clean_line:
                cleaned_code_lines.append(clean_line)

    return "\n".join(cleaned_code_lines)
