"""
Пакет для работы с базой данных GitHub примеров.
"""

from .base_idb import IDB
from .github_docs_db import GitHubDocsDB
from .ds1000_wrapper import DS1000Wrapper
from .github_sqlite_adapter import GitHubSQLiteAdapter

__all__ = [
    "IDB",
    "GitHubDocsDB",
    "DS1000Wrapper",
    "GitHubSQLiteAdapter",
]