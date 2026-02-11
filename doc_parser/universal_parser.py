import libcst as cst
from dataclasses import dataclass
from typing import List, Optional
import logging
from pathlib import Path
import os


# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class DocType(str):
    """Типы объектов для docstrings."""
    MODULE = "module"
    CLASS = "class"
    METHOD = "method"
    FUNCTION = "function"
    ASYNC_FUNCTION = "async_function"
    ASYNC_METHOD = "async_method"


@dataclass
class ObjectInfo:
    """Информация об объекте."""
    library: str  # Название библиотеки/пакета (берется из папки)
    type: DocType  # Тип объекта
    name: str  # Название класса/функции/метода
    filepath: str  # Полный путь к файлу
    signature: Optional[str] = None  # Сигнатура функции/метода
    parent_class: Optional[str] = None  # Только для методов


@dataclass
class String:
    """Контейнер для docstring."""
    information: ObjectInfo
    string: str  # Сам docstring


class LibCSTDocstringExtractor:
    """Извлекает docstrings с использованием LibCST."""

    def __init__(self):
        self.docstrings: List[String] = []
        self.current_library = ""
        self.current_class = None
        self.current_filepath = ""

    def _get_docstring_content(self, node: cst.CSTNode) -> Optional[str]:
        """Извлекает содержимое docstring из узла."""
        body = getattr(node, 'body', None)

        # Для модулей body это список
        if isinstance(body, list) and len(body) > 0:
            first_stmt = body[0]
        # Для классов/функций body это IndentedBlock
        elif isinstance(body, cst.IndentedBlock):
            if len(body.body) > 0:
                first_stmt = body.body[0]
            else:
                return None
        else:
            return None

        # Проверяем, является ли первый statement docstring'ом
        if isinstance(first_stmt, cst.SimpleStatementLine):
            stmt = first_stmt.body[0]
            if isinstance(stmt, cst.Expr) and isinstance(stmt.value, cst.SimpleString):
                raw_string = stmt.value.value

                # Убираем кавычки
                if raw_string.startswith(('"""', "'''")):
                    # Тройные кавычки
                    content = raw_string[3:-3] if raw_string.endswith(raw_string[:3]) else raw_string[3:]
                else:
                    # Одинарные/двойные кавычки
                    content = raw_string[1:-1]

                return content.strip()

        return None

    def _get_signature(self, node: cst.FunctionDef) -> str:
        """Извлекает сигнатуру функции."""
        try:
            # Начинаем с имени функции
            signature = node.name.value + "("

            # Обрабатываем обычные параметры
            param_strings = []
            for param in node.params.params:
                param_str = param.name.value
                param_strings.append(param_str)

            # Обрабатываем *args
            if node.params.star_arg:
                param_str = "*" + node.params.star_arg.name.value
                param_strings.append(param_str)

            # Обрабатываем keyword-only параметры
            for param in node.params.kwonly_params:
                param_str = param.name.value
                param_strings.append(param_str)

            # Обрабатываем **kwargs
            if node.params.star_kwarg:
                param_str = "**" + node.params.star_kwarg.name.value
                param_strings.append(param_str)

            # Собираем все параметры
            signature += ", ".join(param_strings)
            signature += ")"

            # Добавляем возвращаемый тип если есть
            if node.returns:
                returns_str = self._node_to_code(node.returns)
                signature += f" -> {returns_str}"

            return signature

        except Exception as e:
            # В случае ошибки пытаемся получить хотя бы имена параметров
            logger.debug(f"Ошибка при извлечении детальной сигнатуры для {node.name.value}: {e}")

            # Пробуем более простой способ
            try:
                signature = node.name.value + "("
                param_names = []

                # Просто собираем имена параметров
                for param in node.params.params:
                    param_names.append(param.name.value)

                if node.params.star_arg:
                    param_names.append(f"*{node.params.star_arg.name.value}")

                for param in node.params.kwonly_params:
                    param_names.append(param.name.value)

                if node.params.star_kwarg:
                    param_names.append(f"**{node.params.star_kwarg.name.value}")

                signature += ", ".join(param_names)
                signature += ")"

                return signature

            except Exception as e2:
                # Если и это не получилось, возвращаем хотя бы имя функции
                logger.debug(f"Не удалось извлечь параметры для {node.name.value}: {e2}")
                return f"{node.name.value}(...)"

    def _node_to_code(self, node: cst.CSTNode) -> str:
        """Конвертирует узел CST в строку с кодом."""
        try:
            return cst.Module([]).code_for_node(node).strip()
        except:
            return str(node)

    def _is_async_function(self, node: cst.CSTNode) -> bool:
        """Проверяет, является ли функция асинхронной."""
        # Проверяем разные способы определения async функции в зависимости от версии LibCST
        try:
            # Способ 1: Проверяем наличие атрибута async
            if hasattr(node, 'asynchronous') and node.asynchronous:
                return True

            # Способ 2: Проверяем по имени класса
            return type(node).__name__ == 'AsyncFunctionDef'

        except:
            return False

    def _create_docstring(
        self,
        node: cst.CSTNode,
        node_type: str,
        name: str,
        is_function_node: bool = False
    ) -> Optional[String]:
        """Создает String объект из узла."""
        content = self._get_docstring_content(node)

        if not content:
            return None

        # Определяем точный тип для async функций
        if node_type in [DocType.FUNCTION, DocType.METHOD] and self._is_async_function(node):
            if node_type == DocType.FUNCTION:
                node_type = DocType.ASYNC_FUNCTION
            else:
                node_type = DocType.ASYNC_METHOD

        # Получаем сигнатуру для функций/методов
        signature = None
        if is_function_node and isinstance(node, cst.FunctionDef):
            signature = self._get_signature(node)

        # Создаем информацию об объекте
        info = ObjectInfo(
            library=self.current_library,
            type=node_type,
            name=name,
            filepath=self.current_filepath,
            signature=signature,
            parent_class=self.current_class,
        )

        return String(information=info, string=content)

    def _process_node(self, node: cst.CSTNode) -> None:
        """Рекурсивно обрабатывает узел AST."""

        # Обработка класса
        if isinstance(node, cst.ClassDef):
            class_name = node.name.value

            # Сохраняем контекст класса
            previous_class = self.current_class
            self.current_class = class_name

            # Извлекаем docstring класса
            class_doc = self._create_docstring(node, DocType.CLASS, class_name)
            if class_doc:
                self.docstrings.append(class_doc)

            # Обрабатываем содержимое класса
            for child in node.body.body:
                self._process_node(child)

            # Восстанавливаем контекст
            self.current_class = previous_class

        # Обработка функции/метода
        elif isinstance(node, cst.FunctionDef):
            func_name = node.name.value

            # Определяем тип
            if self.current_class is not None:
                doc_type = DocType.METHOD
            else:
                doc_type = DocType.FUNCTION

            # Извлекаем docstring
            func_doc = self._create_docstring(node, doc_type, func_name, is_function_node=True)
            if func_doc:
                self.docstrings.append(func_doc)

        # Рекурсивная обработка дочерних узлов
        elif hasattr(node, 'body'):
            body = node.body
            if isinstance(body, (list, tuple)):
                for child in body:
                    self._process_node(child)
            elif hasattr(body, 'body'):
                for child in body.body:
                    self._process_node(child)

    def extract_from_file(self, filepath: Path, library_name: str) -> None:
        """Извлекает docstrings из одного файла."""
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                code = f.read()
                
            print(len(code))
            # Парсим модуль
            module = cst.parse_module(code)

            # Сохраняем текущий контекст
            self.current_library = library_name
            self.current_filepath = str(filepath)

            # Извлекаем docstring модуля
            module_doc = self._create_docstring(module, DocType.MODULE, "")
            if module_doc:
                self.docstrings.append(module_doc)

            # Рекурсивно обрабатываем все узлы
            for node in module.body:
                self._process_node(node)

        except UnicodeDecodeError:
            logger.error(f"Невозможно прочитать файл: {filepath}")
        except cst.ParserSyntaxError as e:
            logger.error(f"Синтаксическая ошибка в {filepath}: {e}")
        except Exception as e:
            logger.error(f"Ошибка при обработке {filepath}: {type(e).__name__}: {e}")

    def extract_from_directory(self, directory_path: Path, library_name: str) -> List[String]:
        """Извлекает docstrings из директории."""
        # Рекурсивно обходим все Python файлы
        for root, dirs, files in os.walk(directory_path):
            # Пропускаем служебные директории
            dirs[:] = [
                d for d in dirs 
                if not d.startswith('.') 
                and d != '__pycache__'
                and d != 'tests'
                and d != 'test'
            ]

            for file in files:
                if file.endswith('.py'):
                    # Пропускаем тестовые файлы
                    if file.startswith('test_') or file.endswith('_test.py'):
                        continue

                    filepath = Path(root) / file
                    self.extract_from_file(filepath, library_name)

        # Удаляем пустые docstrings
        self.docstrings = [ds for ds in self.docstrings if ds.string]

        logger.info(f"Библиотека {library_name}: извлечено {len(self.docstrings)} docstrings")
        return self.docstrings

    def _is_venv_directory(self, directory_path: Path) -> bool:
        """Проверяет, является ли директория venv."""
        # Проверяем стандартные признаки venv
        venv_indicators = [
            directory_path / 'pyvenv.cfg',
            directory_path / 'Scripts',
            directory_path / 'bin',
        ]

        return any(indicator.exists() for indicator in venv_indicators)

    def _find_libraries_in_venv(self, venv_path: Path) -> List[Path]:
        """Находит библиотеки в venv - ищет только в папке lib."""
        libraries = []

        # Ищем папку lib
        lib_path = venv_path / 'lib'
        if not lib_path.exists():
            logger.error(f"Папка lib не найдена в {venv_path}")
            return libraries

        # Ищем site-packages в lib
        site_packages_paths = []

        # Проверяем возможные пути внутри lib
        possible_paths = [
            lib_path / 'site-packages',
        ]

        # Добавляем пути с версиями Python (python3.*/site-packages)
        if lib_path.exists():
            for item in lib_path.iterdir():
                if item.is_dir() and item.name.startswith('python'):
                    site_packages = item / 'site-packages'
                    if site_packages.exists():
                        possible_paths.append(site_packages)

        # Проверяем все возможные пути
        for site_packages_path in possible_paths:
            if site_packages_path.exists() and site_packages_path.is_dir():
                logger.info(f"Найдена папка site-packages: {site_packages_path}")

                # Добавляем все директории как библиотеки
                for item in site_packages_path.iterdir():
                    if item.is_dir() and not item.name.startswith('.'):
                        libraries.append(item)

                # Если нашли библиотеки, прекращаем поиск
                if libraries:
                    break

        if not libraries:
            logger.error(f"Не найдены библиотеки в {lib_path}")

        return libraries


def extract_docstrings(path: str) -> List[String]:
    """
    Извлекает docstrings из указанного пути.

    Args:
        path: Путь к venv или к директории с библиотекой

    Returns:
        List[String]: Список извлеченных docstrings
    """
    extractor = LibCSTDocstringExtractor()
    directory_path = Path(path)

    if not directory_path.exists():
        logger.error(f"Директория не найдена: {path}")
        return []

    if extractor._is_venv_directory(directory_path):
        # Если это venv, ищем библиотеки ТОЛЬКО в папке lib
        logger.info(f"Обнаружен venv: {path}")
        libraries = extractor._find_libraries_in_venv(directory_path)

        if not libraries:
            logger.error("Не найдены библиотеки в папке lib venv")
            return []

        for library_path in libraries:
            logger.info(f"Обработка библиотеки: {library_path.name}")
            extractor.extract_from_directory(library_path, library_path.name)
    else:
        # Если это обычная директория, обрабатываем как одну библиотеку
        logger.info(f"Обнаружена директория с библиотекой: {path}")
        library_name = directory_path.name
        extractor.extract_from_directory(directory_path, library_name)

    # Фильтруем пустые строки (на всякий случай еще раз)
    result = [ds for ds in extractor.docstrings if ds.string and ds.string.strip()]

    logger.info(f"Всего извлечено: {len(result)} docstrings")
    return result

# Пример использования
if __name__ == "__main__":
    # Пример 1: Извлечение из venv
    docstrings = extract_docstrings("/workspace/venv/lib/python3.11/site-packages/numpy")
    print(docstrings[1].information.signature)
