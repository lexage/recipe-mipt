import sys
import re
import logging
from pathlib import Path
from typing import List

project_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(project_root))

from doc_parser.universal_parser import extract_docstrings


class PlainFormatter(logging.Formatter):
    def format(self, record):
        return record.getMessage()

def setup_logger(log_file):
    logger = logging.getLogger("agent")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    logger.propagate = False

    Path(log_file).parent.mkdir(parents=True, exist_ok=True)

    file_handler = logging.FileHandler(log_file, encoding="utf-8")
    file_handler.setFormatter(PlainFormatter())
    logger.addHandler(file_handler)

    return logger


logger = setup_logger("/workspace/data/extract_example.log")


def extract_examples(docstring: str) -> List[str]:
    """
    Извлекает содержимое секции Examples из докстринга.
    Поддерживает форматы: "Examples", "Examples:", "Examples\n--------".
    """
    # Ищем заголовок секции (с возможным двоеточием и разделителем)
    pattern = (
        r'^\s*Examples\s*:{0,2}\s*\n'          # "Examples" + опц. ":" + перевод строки
        r'(?:\s*[-=]{4,}\s*\n)?'               # Опциональный разделитель (---- или ====)
        r'(.+?)'                               # Содержимое секции (захватываем)
        r'(?=\n\s*[A-Z][a-z]+\s*:{0,2}\s*\n'   # Следующая секция (заголовок с большой буквы)
        r'|\n\s*[-=]{4,}\s*\n'                 # Или новый разделитель
        r'|\Z)'                                # Или конец строки
    )
    match = re.search(pattern, docstring, re.MULTILINE | re.DOTALL)
    
    if match:
        content = match.group(1)
        start = content.find('>>>')
        if start == -1:
            start = content.find('...')
        content = content[start:] if start != -1 else ''
        return split_examples(content.rstrip())
        
    return None


def split_examples(examples: str) -> List[str]:
    """Разделяет примеры"""
    blocks, cur = [], []
    for line in examples.split('\n'):
        s = line.rstrip()
        if s and set(s) <= {'-', '='} and len(s) >= 4:
            continue
        if not s:
            if cur:
                blocks.append('\n'.join(cur))
                cur = []
            continue
        cur.append(s)
    if cur:
        blocks.append('\n'.join(cur))
    return [b for b in blocks if b.strip() and '>>>' in b]


if __name__ == "__main__":
    
    library_names = ["numpy", "scipy", "matplotlib", "torch", "sklearn", "pandas", "tensorflow"]
    

    for library_name in library_names:
        logger.info(f"БИБЛИОТЕКА: {library_name}\n")
        logger.info("_" * 10 + "\n")
        
        docstrings = extract_docstrings(f"/workspace/venv/lib/python3.11/site-packages/{library_name}")
        
        all_examples = []

        i = 0
        for docstring in docstrings:
            docstring_examples = extract_examples(docstring.string)
            if docstring_examples:
                docstring.examples = docstring_examples
                all_examples.append(docstring_examples)
                
                if i < 10:
                    logger.info(f"ПРИМЕР {i+1}\n")
                    logger.info("_" * 10 + "\n")

                    logger.info(f"ИСХОДНАЯ DOCSTRING:\n {docstring.string}\n\n")
                    logger.info("_" * 10 + "\n")
                    
                    logger.info("ИЗВЛЕЧЕННЫЕ ПРИМЕРЫ КОДА:\n\n")
                    for idx, example in enumerate(docstring.examples):
                        logger.info(f"ПРИМЕР КОДА {idx+1}:\n{example}\n\n")
                    i += 1
                    logger.info("_" * 10 + "\n")



        logger.info(f"Количество извлеченных docstrings: {len(docstrings)}\n")
        logger.info("_" * 10 + "\n")
        logger.info(f"Количество docstrings, в которых были обнаружены примеры: {len(all_examples)}\n")
        logger.info("_" * 10 + "\n\n")

        
       

    
    
    
    