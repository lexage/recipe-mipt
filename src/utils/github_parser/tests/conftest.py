import pytest
from typing import Generator
from unittest.mock import patch

from src.utils.github_parser.config import GitHubLoaderConfig


@pytest.fixture
def config_for_github_loader() -> GitHubLoaderConfig:
    return GitHubLoaderConfig()


@pytest.fixture
def mock_github_access_token(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GITHUB_ACCESS_TOKEN", "fake_token_for_tests")


@pytest.fixture
def mock_github_client() -> Generator[pytest.MonkeyPatch]:
    with patch(
        "src.utils.github_parser.github_repo_loader.Github"
    ) as mock_github_client:
        yield mock_github_client
