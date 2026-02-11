import logging
from typing import List


logger = logging.getLogger(__file__)


def decode_byte_content(file_content: bytes, file_path: str) -> str:
    try:
        str_content = file_content.decode("utf-8")
    except UnicodeDecodeError:
        logger.warning("Не удалось декодировать файл %s", file_path)
        raise
    return str_content


def clean_code_lines_from_repl_symbols(raw_code_block: str) -> str:
    """
    чистим от символов >>>
    """
    code_lines: List[str] = []
    for line in raw_code_block.split("\n"):
        line = line.strip()
        if line.startswith(">>> ") or line.startswith("... "):
            clean_line = line[4:]
            code_lines.append(clean_line)
        elif line.startswith("```"):
            continue
        elif not line:
            if code_lines:
                code_lines.append("")
        # добавила вот этот блок от Насти. Чекнуть, что работает
        # на моих классах
        else:
            code_lines.append(line)
    final_code_lines = "\n".join(code_lines)
    return final_code_lines
