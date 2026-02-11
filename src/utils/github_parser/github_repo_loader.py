"""
Модуль для взаимодействия с GitHub API и загрузки содержимого репозиториев.

Обеспечивает функционал для аутентификации, поиска конкретных файлов по
расширениям, фильтрации по паттернам и извлечения README.
"""

import logging
import os
from typing import Iterator, Optional

from dotenv import load_dotenv
from github import Auth, Github, GithubException, Repository

from src.utils.github_parser.config import (
    GitHubLoaderConfig,
    RepoFile,
    RepoWalkResult,
)


# TODO: написать readme + для лексического фильтратора тоже
# TODO: написать тесты!


load_dotenv()
logger = logging.getLogger(__file__)


class TorchGitHubLoader:
    """
    Класс для взаимодействия с GitHub API.
    Осуществляет подключение к репозиториям, рекурсивный обход файловой
    структуры и скачивание сырого содержимого файлов.

    Attributes:
        auth (Auth.Token): Объект аутентификации GitHub.
        github_client (Github): Клиент для работы с GitHub API.
        config (GitHubLoaderConfig): Конфиг для загрузчика файлов с
            репозитория.
    """

    def __init__(
        self,
        config_for_github_loader: GitHubLoaderConfig,
        token: Optional[str] = None,
    ) -> None:
        """
        Инициализирует клиент GitHub.

        Args:
            config_for_github_loader (GitHubLoaderConfig): Конфиг для
                загрузчика файлов с репозитория.
            token (Optional[str]): Персональный токен доступа GitHub.
                Если не указан, пытается получить из переменной окружения
                    GITHUB_ACCESS_TOKEN.

        Raises:
            ValueError: Если токен не передан и не найден в переменных
                окружения.
        """
        token = token or os.environ["GITHUB_ACCESS_TOKEN"]
        if not token:
            raise ValueError(
                "GitHub токен не найден. Проверьте .env или передайте токен!"
            )
        self.auth = Auth.Token(token)
        self.github_client = Github(auth=self.auth)
        self.config = config_for_github_loader

    def _get_specific_repo(
        self, target_repo_name: str
    ) -> Repository.Repository:
        """
        Получает объект репозитория GitHub по его полному имени.

        Args:
            target_repo_name (str): Полное имя репозитория.

        Returns:
            Repository.Repository: Объект репозитория PyGithub.

        Raises:
            GithubException: Если репозиторий не найден или возникли
                проблемы с API.
        """
        try:
            target_repo = self.github_client.get_repo(target_repo_name)
            logger.info(
                "Установлено подключение к репозиторию: %s",
                target_repo.full_name,
            )
            return target_repo
        except GithubException as e:
            logger.error("Ошибка GitHub API: %s", e, exc_info=True)
            raise

    def get_readme_content(self, target_repo: Repository.Repository) -> str:
        """
        Скачивает и декодирует содержимое файла README из репозитория.

        Args:
            target_repo (Repository.Repository): Объект репозитория, из
                которого нужно получить README.

        Returns:
            str: Декодированное содержимое README. Возвращает пустую строку,
                если файл не найден.
        """
        try:
            readme = target_repo.get_readme()
            logger.info("Нашли файл README.md!")
            return readme.decoded_content.decode()
        except GithubException:
            logger.warning(
                f"README не найден в репозитории {target_repo.full_name}"
            )
            return ""

    def get_python_and_markdown_files(
        self,
        target_repo: Repository.Repository,
        specific_folder_for_search: str = "",
    ) -> Iterator[RepoFile]:
        """
        Рекурсивно обходит репозиторий и возвращает содержимое файлов.

        Args:
            target_repo (Repository.Repository): Объект репозитория для поиска.
            specific_folder_for_search (str): Путь к конкретной папке внутри
                репозитория. По умолчанию поиск идет от корня.

        Yields:
            Iterator[RepoFile]: Итератор объектов RepoFile, содержащих путь к
                файлу и его байтовое содержимое.

        Note:
            Метод игнорирует файлы, имена которых содержат паттерны из
                конфига.
        """
        try:
            target_repo_content = target_repo.get_contents(
                specific_folder_for_search
            )
        except GithubException:
            logger.error(
                "Не удалось получить содержимое пути %s",
                specific_folder_for_search,
                exc_info=True,
            )
            return

        # обычно возвращается список объектов типа ContentFile,
        # но на всякий случай оборачиваем в список,
        # если пришел единичный объект
        if not isinstance(target_repo_content, list):
            target_repo_content = [target_repo_content]

        while target_repo_content:
            file_content = target_repo_content.pop(0)

            if file_content.type == "dir":
                try:
                    content = target_repo.get_contents(file_content.path)
                    target_repo_content.extend(
                        content if isinstance(content, list) else [content]
                    )
                except GithubException:
                    logger.error(
                        "Не удалось получить содержимое пути %s, "
                        "пропускаем его",
                        file_content.path,
                        exc_info=True,
                    )
                    continue

            elif file_content.type == "file" and file_content.path.endswith(
                self.config.allowed_file_extensions
            ):
                if any(
                    skip_pattern in file_content.name
                    for skip_pattern in self.config.skip_file_patterns
                ):
                    continue

                try:
                    yield RepoFile(
                        path=file_content.path,
                        content=file_content.decoded_content,
                    )
                except Exception:
                    logger.warning(
                        "Ошибка чтения файла %s",
                        file_content.path,
                        exc_info=True,
                    )

    def repo_walk(
        self, target_repo_name: str, specific_folder_for_search: str
    ) -> RepoWalkResult:
        """
        Выполняет комплексный обход репозитория: извлекает README и
        создает итератор по всем подходящим файлам.

        Args:
            target_repo_name (str): Полное имя репозитория.
            specific_folder_for_search (str): Стартовая папка/файл для поиска.

        Returns:
            RepoWalkResult: Структура данных, содержащая строку README,
                объект репозитория и итератор файлов.
        """
        target_repo = self._get_specific_repo(target_repo_name)
        readme = self.get_readme_content(target_repo)
        files_iterator = self.get_python_and_markdown_files(
            target_repo, specific_folder_for_search
        )
        return RepoWalkResult(
            readme=readme,
            target_repo=target_repo,
            files_iterator=files_iterator,
        )
