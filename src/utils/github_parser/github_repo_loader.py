import logging
import os
from typing import Iterator, Optional

from dotenv import load_dotenv
from github import Auth, Github, GithubException, Repository

from src.utils.github_parser.github_parser_config import (
    GitHubExampleFetcherConfig,
    RepoFile,
    RepoWalkResult,
)


# TODO: написать readme + для лексического фильтратора тоже
# TODO: написать тесты!

load_dotenv()
GITHUB_ACCESS_TOKEN = os.environ["GITHUB_ACCESS_TOKEN"]
logger = logging.getLogger(__file__)


# TODO: докстринги! + порядок импортов
# TODO: подумать, как обрабатывать .md!
# TODO: проверить init


class TorchGitHubLoader:
    """
    Класс для взаимодействия с GitHub API. Осуществляет
    подключение и скачивание сырых файлов из выбранного репозитория.
    """

    def __init__(
        self, config: GitHubExampleFetcherConfig, token: Optional[str] = None
    ) -> None:
        """
        Docstring for __init__
        """
        token = token or os.environ["GITHUB_ACCESS_TOKEN"]
        if not token:
            raise ValueError(
                "GitHub Token не найден. Проверьте .env или передайте токен."
            )
        self.auth = Auth.Token(token)
        self.github_client = Github(auth=self.auth)
        self.config = config.config_for_github_loader

    def _get_specific_repo(
        self, target_repo_name: str
    ) -> Repository.Repository:
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
        """Скачивает и декодирует содержимое README.md."""
        try:
            readme = target_repo.get_readme()
            logger.info(
                "Нашли файл README.md, который имеет %s строк",
                len(readme.line_numbers),
            )
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
        Если specific_folder_for_search пустая, то вернет содержимое всей репы.
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
        Docstring for __call__

        :param self: Description
        :return: Description
        :rtype: Any
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
