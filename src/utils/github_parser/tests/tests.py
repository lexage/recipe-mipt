import pytest
from typing import cast
from unittest.mock import MagicMock

from src.utils.github_parser.config import GitHubLoaderConfig
from src.utils.github_parser.github_repo_loader import TorchGitHubLoader


def test_init_raises_error_without_token(
    monkeypatch: pytest.MonkeyPatch,
    config_for_github_loader: GitHubLoaderConfig,
) -> None:
    monkeypatch.delenv("GITHUB_ACCESS_TOKEN")
    with pytest.raises(
        ValueError,
        match="GitHub токен не найден. Проверьте .env или передайте токен!",
    ):
        TorchGitHubLoader(config_for_github_loader)


def test_get_specific_repo(
    mock_env_token: str,
    mock_github_client: MagicMock,
    config_for_github_loader: GitHubLoaderConfig,
):
    github_loader = TorchGitHubLoader(config_for_github_loader, mock_env_token)
    mock_repo = MagicMock()
    mock_repo.full_name = "test/repo"
    cast(MagicMock, github_loader.github_client).get_repo.return_value = (
        mock_repo
    )
    result = github_loader._get_specific_repo("test/repo")  # type: ignore
    assert result == mock_repo
