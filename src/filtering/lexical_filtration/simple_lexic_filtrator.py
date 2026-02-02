"""
Модуль для лексической фильтрации и нормализации текстовых чанков.

Основные возможности модуля включают удаление "мусорных" строк, определяемых
конфигурационным файлом, а также удаление навигационных элементов,
нормализацию пустых строк и дедупликацию повторяющихся блоков текста.
"""

import logging
import hashlib
from pydantic import BaseModel, Field
import re
from typing import List, Optional, Set

from src.agent_constructor.core import Chunk
from src.agent_constructor.filters import Filter


# TODO: дальнейшая оптимизация после тестирования:
# 1. Удаление всех дублей строк может быть опасно для кода. Например, мб два раза return в разветвленной функции;
# фильтратор решит, что это дубляж, хотя по факту это часть кода. При своем небольшом тестированиии я таких случаев не встречала,
# но это стоит держать в уме. Потенциальное решение: посмотреть по результатам тестирования или создать вспомогательную функцию,
# определяющую, является ли дубляж блоком кода (например, по ключевым словам типа return, def, ...).


logger = logging.getLogger(__name__)


class FilteringConfig(BaseModel):
    remove_junk_blocks: bool = Field(
        description="Флаг удаления мусорных строк (определяется списком "
        "мусорных слов в конструкторе класса)",
        default=True,
    )
    remove_navigation_lines: bool = Field(
        description="Флаг удаления навигационных строк (определяется списком "
        "навигационных слов в конструкторе класса)",
        default=True,
    )
    min_len_for_dedup: int = Field(
        default=5,
        description="Минимальная длина строки для дедупликации по хэшу",
    )
    sliding_window_max_length: int = Field(
        default=50, description="Максимальная длина скользящего окна"
    )
    normalize_empty_lines: bool = Field(
        description="Флаг удаления блоков пустых строк", default=True
    )
    remove_terminal_sections: bool = Field(
        description="Флаг удаления терминальных блоков", default=True
    )


class SimpleLexicalFiltrator(Filter):
    """
    Класс для простой лексической фильтрации, нормализации и дедупликации
    текстовых чанков из БД, содержащих код.

    Использует три основных метода:
    1. Дедупликация последовательных блоков строк с помощью скользящего окна.
    2. Удаление непоследовательных дубликатов строк на основе MD5-хэшей.
    3. Удаление мусорных и навигационных строк с помощью регулярных выражений
    или поиска подвхождений.
    """

    def __init__(self, config: Optional[FilteringConfig] = None) -> None:
        """
        Инициализирует фильтратор с конфигом.

        Args:
            config (Optional[FilteringConfig]): Объект конфигурации. Если не
                передан, используется конфигурация по умолчанию.
        """
        self.config = config or self._get_default_config()

        # навигационные ключевые слова
        self.nav_keywords: List[re.Pattern[str]] = [
            re.compile(re.escape(p), re.IGNORECASE)
            for p in [
                "see also",
                "Go to the end to download the full example code",
                "View on TensorFlow.org",
                "Run in Google Colab",
                "View source on GitHub",
                "Download notebook",
            ]
        ]

        # терминальные секции (т.е. секции, которые обычно помещаются в конце
        # текстового блока чанка и не имеют смысловой ценности)
        self.terminal_patterns = [
            re.compile(r"References#.*", flags=re.IGNORECASE | re.DOTALL)
        ]

        # паттерны для мусорных блоков
        self.junk_patterns: List[re.Pattern[str]] = [
            re.compile(p, re.IGNORECASE)
            for p in [
                r"Last updated:.*",
                r"Created On:.*",
                r"Published:.*",
                r"Author:.*",
                r"arXiv:\d+\.\d+",  # обычно это ссылки на статью с
                # оригинальной имплементацией метода, содержатся в References#
                # оставлены на всякий случай, мб стоит убрать после
                # тестирования и/или рефакторинга
                r"DOI:.*",  # аналогично
                r"ISBN:.*",  # аналогично
                r"\b\d{1,2}/\d{1,2}/\d{4}\b",
                r"\b\d{4}-\d{2}-\d{2}\b",
                r"email:.*@.*",
            ]
        ]

    @staticmethod
    def _get_default_config() -> FilteringConfig:
        """
        Создает и возвращает конфигурацию по умолчанию.

        Returns:
            FilteringConfig: Объект конфигурации с дефолтными значениями.
        """
        return FilteringConfig()

    def _is_junk_line(self, line: str) -> bool:
        """
        Проверяет, является ли строка "мусорной" согласно мусорным паттернам в
        конструкторе класса.

        Args:
            line (str): Строка для проверки.

        Returns:
            bool: True, если строка считается мусорной, иначе False.
        """
        normalized = line.rstrip()
        if not normalized:
            return False
        for pattern in self.junk_patterns:
            if pattern.search(normalized):
                logger.info(
                    "Нашли мусорный паттерн %s в строке %s, удаляем...",
                    pattern,
                    normalized,
                )
                return True
        if len(normalized) < self.config.min_len_for_dedup and re.search(
            r"^\d+$", normalized
        ):
            logger.info(
                "Строка %s слишком маленькая и состоит только из цифр, "
                "удаляем...",
                normalized,
            )
            return True
        # проверка на строки только из спецсимволов
        if re.match(r"^[-\*_=]{5,}$", normalized):
            logger.info(
                "Строка %s состоит только из спецсимволов, удаляем...",
                normalized,
            )
            return True
        return False

    def _count_md5_hash(self, text: str) -> str:
        """
        Считает MD5-хэш от нормализованного текста.

        Args:
            text (str): Входной текст.

        Returns:
            str: MD5-хэш нормализованного текста в виде hex-строки.
        """
        normalized_text = text.rstrip()
        md5_hash = hashlib.md5(normalized_text.encode())
        return md5_hash.hexdigest()

    def _is_navigation(self, line: str) -> bool:
        """
        Проверяет, является ли строка навигационной (что считается
        навигационным, определяется в списке выражений в конструкторе класса).

        Args:
            line (str): Строка для проверки.

        Returns:
            bool: True, если строка содержит навигационные ключевые слова,
                иначе False.
        """
        normalized = line.rstrip()
        if not normalized:
            return False
        for nav_pattern in self.nav_keywords:
            if nav_pattern.search(line):
                return True
        return False

    def remove_terminal_sections(self, chunk: Chunk) -> None:
        """
        Удаляет всё, начиная с ключевого слова и до конца текста. Изменяет
        поле `text` чанка in-place.

        Args:
            chunk (Chunk): Объект чанка для обработки.
        """
        for pattern in self.terminal_patterns:
            match = pattern.search(chunk.text)
            if match:
                logger.info(
                    "Нашли терминальный блок, отрезаем текст с позиции %s",
                    match.start(),
                )
                chunk.text = chunk.text[: match.start()].rstrip()
                break

    def normalize_empty_lines(self, chunk: Chunk) -> None:
        """
        Заменяет множественные пустые строки на одну. Изменяет поле `text`
        чанка in-place.

        Args:
            chunk (Chunk): Объект чанка для обработки.
        """
        # регулярка ищет 3 или больше символов переноса строки подряд.
        # заменяем на одну пустую строку между линиями текста.
        chunk.text = re.sub(r"\n{3,}", "\n\n", chunk.text)

    def filter_navigation_lines_in_chunk(self, chunk: Chunk) -> None:
        """
        Удаляет навигационные строки из текста чанка. Изменяет поле `text`
        чанка in-place.

        Args:
            chunk (Chunk): Объект чанка для обработки.
        """
        lines = chunk.text.split("\n")
        cleaned_lines = [
            line for line in lines if not self._is_navigation(line)
        ]
        chunk.text = "\n".join(cleaned_lines)

    def filter_junk_lines_in_chunk(self, chunk: Chunk) -> None:
        """
        Удаляет мусорные строки из текста чанка. Изменяет поле `text` чанка
        in-place.

        Args:
            chunk (Chunk): Объект чанка для обработки.
        """
        lines = chunk.text.split("\n")
        cleaned_lines = [
            line for line in lines if not self._is_junk_line(line)
        ]

        chunk.text = "\n".join(cleaned_lines)

    def filter_duplicate_lines(self, chunk: Chunk) -> None:
        """
        Фильтрует дублирующиеся строки внутри чанка.

        Сохраняет первую встреченную уникальную строку (с ее оригинальными
        отступами), а все последующие ее дубликаты удаляет.
        Изменяет поле `text` чанка in-place.

        Args:
            chunk (Chunk): Объект чанка для обработки.
        """
        original_chunk_text_lines: List[str] = chunk.text.split("\n")
        unique_lines: List[str] = []
        local_seen_hashes: Set[str] = (
            set()
        )  # локальный сет для текущего чанка, чтобы хэши не накапливались
        # между чанками
        for line in original_chunk_text_lines:
            # cчитаем хэш от нормализованной строки
            line_hash = self._count_md5_hash(line)

            if not line.strip():
                if unique_lines and unique_lines[-1].strip():
                    unique_lines.append("")
                continue

            # защита кода: не удаляем короткие строки типа return True, break.
            # TODO: в будущем заменить на вспомогательную функцию,
            # определяющую, код в строке или нет
            if len(line.strip()) < self.config.min_len_for_dedup:
                unique_lines.append(line)
                continue

            if line_hash not in local_seen_hashes:
                unique_lines.append(line)
                local_seen_hashes.add(line_hash)
            else:
                logger.debug("Найдена и удалена строка-дубликат: %s", line)
        chunk.text = "\n".join(unique_lines)

    def deduplicate_text_per_lines_with_sliding_window(
        self, chunk: Chunk
    ) -> None:
        """
        Устраняет последовательные повторы блоков строк внутри одного чанка.

        Алгоритм ищет повторяющиеся последовательности строк (окна) длиной от
        1 до `sliding_window_max_length` и "схлопывает" их до одного
        экземпляра.
        Метод изменяет поле text чанка in-place.

        Args:
            chunk (Chunk): Объект чанка для обработки.
        """
        # нормализуем строки, сохраняя пробелы только с правой стороны,
        # чтобы не нарушать важные для синтаксиса кода отступы
        lines = [line.rstrip() for line in chunk.text.split("\n")]
        lines_number = len(lines)
        i = 0
        result_lines: List[str] = []

        # пытаемся найти самый длинный блок повтора, начинающийся с i
        while i < lines_number:
            best_k = 0
            max_window = min(
                self.config.sliding_window_max_length, (lines_number - i) // 2
            )
            for k in range(1, max_window + 1):
                block1 = lines[i: i + k]
                block2 = lines[i + k: i + 2 * k]
                logger.debug(
                    "Сравниваем блоки текста длиной %s: %s и %s",
                    k,
                    block1,
                    block2,
                )
                if block1 == block2:
                    # убедимся, что блок не состоит только из пустых строк
                    if any(line.strip() for line in block1):
                        best_k = k
                        logger.info(
                            "Найден дублирующийся блок из %s строк!", best_k
                        )
            if best_k > 0:
                result_lines.extend(lines[i: i + best_k])
                # перепрыгиваем блок-дубляж
                i += best_k
                # делаем дополнительный цикл для проверки,
                # что блок повторяется только раз
                while i + best_k <= lines_number:
                    next_block = lines[i: i + best_k]
                    if next_block == lines[i - best_k: i]:
                        i += best_k
                    else:
                        break
            else:
                # повторов не найдено, просто добавляем текущую строку
                result_lines.append(lines[i])
                i += 1
        chunk.text = "\n".join(result_lines)

    def apply(self, chunks: List[Chunk]) -> List[Chunk]:
        """
        Выполняет полную лексическую фильтрацию для списка чанков.

        Процесс дедупликации и фильтрации каждого чанка (в зависимости от
        конфигурационного файла):
        1. Удаление мусорных строк.
        2. Удаление навигационных строк.
        3. Нормализация пустых строк.
        4. Дедупликация последовательных блоков текста (скользящее окно).
        5. Дедупликация повторяющихся строк.

        Args:
            List[Chunk]: Список исходных чанков.

        Returns:
            List[Chunk]: Отфильтрованный список чанков.
        """
        final_chunks: List[Chunk] = []
        for idx, chunk in enumerate(chunks, start=1):
            logger.info("Начинаем лексическую фильтрацию чанка № %s...", idx)
            if self.config.remove_terminal_sections:
                self.remove_terminal_sections(chunk)
            if self.config.remove_junk_blocks:
                self.filter_junk_lines_in_chunk(chunk)
            if self.config.remove_navigation_lines:
                self.filter_navigation_lines_in_chunk(chunk)
            if self.config.normalize_empty_lines:
                self.normalize_empty_lines(chunk)

            # удаляем точные дубликаты с помощью скользящего окна
            self.deduplicate_text_per_lines_with_sliding_window(chunk)
            logger.debug(
                "После дедупликации методом скользящего окна текст чанка "
                "стал: %s",
                chunk.text,
            )
            # удаляем строки-дубликаты
            self.filter_duplicate_lines(chunk)
            logger.debug(
                "После удаления строк-дубликатов текст чанка стал: %s",
                chunk.text,
            )
            logger.info("Лексическая фильтрация успешно завершена!")
            if chunk.text.strip():
                final_chunks.append(chunk)
        return final_chunks
