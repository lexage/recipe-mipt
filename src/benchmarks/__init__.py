from importlib import import_module

from .swe_rebench import DatasetSWERebench, SWERebenchDataError, SWERebenchTask

__all__ = [
    "DS1000",
    "DataItemDS1000",
    "ResultsDS1000",
    "DatasetSWERebench",
    "SWERebenchDataError",
    "SWERebenchTask",
]


def __getattr__(name: str):
    if name in {"DS1000", "DataItemDS1000", "ResultsDS1000"}:
        return getattr(import_module(".ds1000", __name__), name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
