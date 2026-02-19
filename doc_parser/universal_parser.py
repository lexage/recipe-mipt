import sys
import libcst as cst
from dataclasses import dataclass
from typing import List, Optional
import logging
from pathlib import Path
import os

project_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(project_root))

from src.agent_constructor.core import Chunk


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
    signature: str  # Полная сигнатура из кода (с def/class и до :)
    parent_class: Optional[str] = None  # Только для методов
    return_info: Optional[str] = None  # Строка(и) с return/yield из тела функции


@dataclass
class String:
    """Контейнер для docstring."""
    information: ObjectInfo
    string: str  # Сам docstring
    examples: List[str] = None # Примеры, которые относятся к docstrings


class ReturnStatementVisitor(cst.CSTVisitor):
    """Посетитель для поиска return и yield операторов."""

    def __init__(self):
        self.return_statements: List[str] = []

    def visit_Return(self, node: cst.Return) -> None:
        """Посещает return операторы."""
        try:
            if node.value:
                code = cst.Module([]).code_for_node(node.value).strip()
                self.return_statements.append(f"return {code}")
            else:
                self.return_statements.append("return")
        except:
            pass

    def visit_Yield(self, node: cst.Yield) -> None:
        """Посещает yield операторы."""
        try:
            if node.value:
                code = cst.Module([]).code_for_node(node.value).strip()
                self.return_statements.append(f"yield {code}")
            else:
                self.return_statements.append("yield")
        except:
            pass


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
                    content = raw_string[3:-3] if raw_string.endswith(raw_string[:3]) else raw_string[3:]
                else:
                    content = raw_string[1:-1]

                return content.strip()

        return None

    def _format_parameter(self, param: cst.Param) -> str:
        """Форматирует отдельный параметр с полной аннотацией типа."""
        try:
            param_str = param.name.value

            # Добавляем аннотацию типа если есть
            if param.annotation:
                try:
                    # Получаем узел аннотации и конвертируем в код
                    annotation_node = param.annotation.annotation
                    if annotation_node:
                        annotation = cst.Module([]).code_for_node(annotation_node).strip()
                        if annotation:
                            param_str += f": {annotation}"
                except Exception:
                    pass

            # Добавляем значение по умолчанию если есть
            if param.default:
                try:
                    default = cst.Module([]).code_for_node(param.default).strip()
                    if default:
                        param_str += f" = {default}"
                except Exception:
                    pass

            return param_str
        except Exception:
            return "..."

    def _get_param_name(self, param) -> Optional[str]:
        """Безопасно извлекает имя параметра."""
        try:
            if hasattr(param, 'name'):
                if hasattr(param.name, 'value'):
                    return param.name.value
            return None
        except:
            return None

    def _get_function_signature(self, node: cst.FunctionDef) -> str:
        """Извлекает полную сигнатуру функции с аннотациями типов."""
        try:
            # Определяем префикс async
            prefix = "async " if self._is_async_function(node) else ""

            # Получаем все параметры
            params = []

            # Обычные параметры
            for param in node.params.params:
                param_str = self._format_parameter(param)
                params.append(param_str)

            # *args параметр или разделитель
            if node.params.star_arg != cst.MaybeSentinel.DEFAULT:
                if isinstance(node.params.star_arg, cst.Param):
                    # Это *args параметр
                    param_name = node.params.star_arg.name.value
                    param_str = f"*{param_name}"

                    # Добавляем аннотацию если есть
                    if node.params.star_arg.annotation:
                        try:
                            annotation_node = node.params.star_arg.annotation.annotation
                            if annotation_node:
                                annotation = cst.Module([]).code_for_node(annotation_node).strip()
                                if annotation:
                                    param_str += f": {annotation}"
                        except Exception:
                            pass

                    params.append(param_str)
                else:
                    # Это просто звездочка-разделитель
                    params.append("*")

            # Keyword-only параметры
            for param in node.params.kwonly_params:
                param_str = self._format_parameter(param)
                params.append(param_str)

            # **kwargs параметр
            if node.params.star_kwarg != cst.MaybeSentinel.DEFAULT and isinstance(node.params.star_kwarg, cst.Param):
                param_name = node.params.star_kwarg.name.value
                param_str = f"**{param_name}"

                # Добавляем аннотацию если есть
                if node.params.star_kwarg.annotation:
                    try:
                        annotation_node = node.params.star_kwarg.annotation.annotation
                        if annotation_node:
                            annotation = cst.Module([]).code_for_node(annotation_node).strip()
                            if annotation:
                                param_str += f": {annotation}"
                    except Exception:
                        pass

                params.append(param_str)

            # Формируем базовую сигнатуру
            if params:
                signature = f"{prefix}def {node.name.value}(" + ", ".join(params) + ")"
            else:
                signature = f"{prefix}def {node.name.value}()"

            # Добавляем возвращаемый тип если есть
            if node.returns:
                try:
                    # Получаем узел возвращаемого типа
                    if hasattr(node.returns, 'annotation'):
                        returns_node = node.returns.annotation
                    else:
                        returns_node = node.returns

                    returns = cst.Module([]).code_for_node(returns_node).strip()
                    if returns:
                        signature += f" -> {returns}"
                except Exception:
                    pass

            # Добавляем двоеточие в конце
            signature += ":"

            return signature

        except Exception:
            # Возвращаем упрощенную сигнатуру в случае ошибки
            prefix = "async " if self._is_async_function(node) else ""
            return f"{prefix}def {node.name.value}(...):"

    def _get_class_signature(self, node: cst.ClassDef) -> str:
        """Извлекает полную сигнатуру класса."""
        try:
            signature = f"class {node.name.value}"

            # Добавляем базовые классы если есть
            if node.bases:
                bases = []
                for base in node.bases:
                    try:
                        base_str = cst.Module([]).code_for_node(base.value).strip()
                        if base_str:
                            bases.append(base_str)
                    except:
                        pass

                if bases:
                    signature += "(" + ", ".join(bases) + ")"

            # Добавляем двоеточие
            signature += ":"

            return signature

        except Exception:
            return f"class {node.name.value}:"

    def _get_signature_from_node(self, node: cst.CSTNode) -> str:
        """
        Извлекает сигнатуру непосредственно из узла CST.
        """
        try:
            if isinstance(node, cst.FunctionDef):
                return self._get_function_signature(node)
            elif isinstance(node, cst.ClassDef):
                return self._get_class_signature(node)
            elif isinstance(node, cst.Module):
                return "<module>"
            else:
                if hasattr(node, 'name') and hasattr(node.name, 'value'):
                    return f"{node.name.value}"
                return f"<{type(node).__name__}>"

        except Exception:
            if hasattr(node, 'name') and hasattr(node.name, 'value'):
                return f"{node.name.value}"
            return "<unknown>"

    def _get_return_statements(self, node: cst.FunctionDef) -> Optional[str]:
        """
        Извлекает все return и yield операторы из тела функции.
        Возвращает их в виде строки, разделенной переносами строк.
        """
        try:
            visitor = ReturnStatementVisitor()
            node.body.visit(visitor)

            if visitor.return_statements:
                return '\n'.join(visitor.return_statements)

            return None  # Нет явных return/yield операторов

        except Exception:
            return None

    def _node_to_code(self, node: cst.CSTNode) -> str:
        """Конвертирует узел CST в строку с кодом."""
        try:
            return cst.Module([]).code_for_node(node).strip()
        except:
            return str(node)

    def _is_async_function(self, node: cst.CSTNode) -> bool:
        """Проверяет, является ли функция асинхронной."""
        try:
            if hasattr(node, 'asynchronous') and node.asynchronous:
                return True
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

        # Получаем сигнатуру из узла
        signature = self._get_signature_from_node(node)

        # Получаем return/yield операторы ТОЛЬКО для функций/методов
        return_info = None
        if is_function_node and isinstance(node, cst.FunctionDef):
            return_info = self._get_return_statements(node)

        # Создаем информацию об объекте
        info = ObjectInfo(
            library=self.current_library,
            type=node_type,
            name=name,
            filepath=self.current_filepath,
            signature=signature,
            parent_class=self.current_class,
            return_info=return_info  # Для классов останется None
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

        # Обработка асинхронной функции
        elif hasattr(cst, 'AsyncFunctionDef') and isinstance(node, cst.AsyncFunctionDef):
            func_name = node.name.value

            # Определяем тип
            if self.current_class is not None:
                doc_type = DocType.ASYNC_METHOD
            else:
                doc_type = DocType.ASYNC_FUNCTION

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
        except Exception:
            logger.error(f"Ошибка при обработке {filepath}")

    # def extract_from_chunk(self, chunk: Chunk) -> None:
    #     """Извлекает docstrings из одного чанка."""
    #     try:
    #         # Парсим модуль
    #         module = cst.parse_module(chunk.text)

    #         # Сохраняем текущий контекст
    #         self.current_library = chunk.metadata["library_name"]
    #         self.current_filepath = None

    #         # Извлекаем docstring модуля
    #         module_doc = self._create_docstring(module, DocType.MODULE, "")
    #         if module_doc:
    #             self.docstrings.append(module_doc)

    #         # Рекурсивно обрабатываем все узлы
    #         for node in module.body:
    #             self._process_node(node)

    #     except Exception as e:
    #         logger.error(f"Ошибка при обработке чанка: {chunk}. {type(e).__name__}: {e}")

    def extract_from_chunk(self, chunk: Chunk) -> None:
        """Извлекает docstrings из одного чанка БЕЗ парсинга кода."""
        content = chunk.text.strip()
        if not content:
            return

        # Эвристическое определение типа объекта и имени из первой строки
        first_line = content.split('\n')[0].strip()
        
        # Определяем тип объекта
        doc_type = DocType.MODULE
        if '(' in first_line and ')' in first_line:
            # Похоже на функцию/метод (есть скобки сигнатуры)
            doc_type = DocType.METHOD if self.current_class else DocType.FUNCTION
        
        # Извлекаем имя объекта (последний элемент до скобок/пробела)
        name = ""
        if first_line:
            # Убираем комментарии после '#'
            clean_line = first_line.split('#')[0].strip()
            # Берём последний элемент пути (после точки)
            parts = [p for p in clean_line.split('.') if p]
            if parts:
                candidate = parts[-1].split('(')[0].split()[0].strip()
                if candidate and not candidate.startswith((' ', '#', '[', ']')):
                    name = candidate

        # Создаём объект информации
        info = ObjectInfo(
            library=chunk.metadata.get("library_name", "unknown"),
            type=doc_type,
            name=name,
            filepath=None,
            signature=None,  # Сигнатура недоступна в чистом тексте
            parent_class=self.current_class,
        )

        docstring = String(information=info, string=content)
        self.docstrings.append(docstring)

    def extract_from_directory(self, directory_path: Path, library_name: str) -> List[String]:
        """Извлекает docstrings из директории."""
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
        venv_indicators = [
            directory_path / 'pyvenv.cfg',
            directory_path / 'Scripts',
            directory_path / 'bin',
        ]
        return any(indicator.exists() for indicator in venv_indicators)

    def _find_libraries_in_venv(self, venv_path: Path) -> List[Path]:
        """Находит библиотеки в venv."""
        libraries = []

        lib_path = venv_path / 'lib'
        if not lib_path.exists():
            logger.error(f"Папка lib не найдена в {venv_path}")
            return libraries

        site_packages_paths = []
        possible_paths = [
            lib_path / 'site-packages',
        ]

        if lib_path.exists():
            for item in lib_path.iterdir():
                if item.is_dir() and item.name.startswith('python'):
                    site_packages = item / 'site-packages'
                    if site_packages.exists():
                        possible_paths.append(site_packages)

        for site_packages_path in possible_paths:
            if site_packages_path.exists() and site_packages_path.is_dir():
                logger.info(f"Найдена папка site-packages: {site_packages_path}")

                for item in site_packages_path.iterdir():
                    if item.is_dir() and not item.name.startswith('.'):
                        libraries.append(item)

                if libraries:
                    break

        if not libraries:
            logger.error(f"Не найдены библиотеки в {lib_path}")

        return libraries


def extract_docstrings(path: str = None, chunks: List[Chunk] = None) -> List[String]:
    """
    Извлекает docstrings из указанного пути.

    Args:
        path: Путь к venv или к директории с библиотекой, если путь None, то docstrings извлекаются из чанков

    Returns:
        List[String]: Список извлеченных docstrings
    """
    extractor = LibCSTDocstringExtractor()

    if path:
        directory_path = Path(path)

        if not directory_path.exists():
            logger.error(f"Директория не найдена: {path}")
            return []

    if extractor._is_venv_directory(directory_path):
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

    result = [ds for ds in extractor.docstrings if ds.string and ds.string.strip()]
    logger.info(f"Всего извлечено: {len(result)} docstrings")
    return result


# Пример использования
if __name__ == "__main__":
    docstrings = extract_docstrings("./venv/lib/python3.12/site-packages/pandas")

    # for i, ds in enumerate(docstrings[:5]):
    #     logger.info(f"\n--- Docstring {i+1} ---")
    #     logger.info(f"Library: {ds.information.library}")
    #     logger.info(f"Type: {ds.information.type}")
    #     logger.info(f"Name: {ds.information.name}")
    #     if ds.information.parent_class:
    #         logger.info(f"Parent class: {ds.information.parent_class}")
    #     logger.info(f"Signature: {ds.information.signature}")
    #     if ds.information.return_info is not None:
    #         logger.info(f"Return info: {ds.information.return_info}")
    #     logger.info(f"File: {ds.information.filepath}")
    #     logger.info(f"Docstring preview: {ds.string[:100]}...")

    for i, ds in enumerate(docstrings):
        logger.info(f"\n--- Docstring {i+1} ---")
        # logger.info(f"Type: {ds.information.type}")
        logger.info(f"Signature: {ds.information.signature}")
        if ds.information.return_info is not None:
            logger.info(f"Return info: {ds.information.return_info}")
