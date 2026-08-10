import logging
import multiprocessing
import threading
import re
from pathlib import Path


class _AutoRouteHandler(logging.Handler):
    """
    Logging handler that auto-routes messages based on thread name.

    Routes logs from ThreadPoolExecutor workers and multiprocessing workers to
    separate files, while parent-process logs go to a main file.
    """

    def __init__(self, log_dir: str, base_name: str, n_workers: int):
        super().__init__()
        self.dir = Path(log_dir)
        self.dir.mkdir(parents=True, exist_ok=True)  # ✅ Создаём папку
        self.base = base_name
        self.n = n_workers
        self._cache = {}
        self._lock = threading.Lock()
        self._main = self._make_handler(None)
        self._pattern = re.compile(r"ThreadPoolExecutor-\d+_(\d+)")
        self.setFormatter(logging.Formatter("%(message)s"))

    def _make_handler(self, wid: int | str | None):
        """Create a FileHandler for main log or specific worker."""
        if isinstance(wid, int):
            fname = f"{self.base}_worker_{wid}.log"
        elif isinstance(wid, str):
            fname = f"{self.base}_process_{wid}.log"
        else:
            fname = f"{self.base}_main.log"
        h = logging.FileHandler(self.dir / fname, "a")
        h.setFormatter(self.formatter)
        return h

    def _get_wid(self):
        """Extract worker ID from current thread name."""
        process_name = multiprocessing.current_process().name
        if process_name != "MainProcess":
            return re.sub(r"[^a-zA-Z0-9_.-]+", "-", process_name).strip("-_.")
        match = self._pattern.match(threading.current_thread().name)
        return int(match.group(1)) % self.n if match else None

    def emit(self, record):
        """Route and emit a log record to the appropriate file handler."""
        wid = self._get_wid()
        if wid not in self._cache:
            with self._lock:
                if wid not in self._cache:
                    self._cache[wid] = (
                        self._main if wid is None else self._make_handler(wid)
                    )
        self._cache[wid].emit(record)

    def close(self):
        """Close all cached file handlers and the main handler."""
        with self._lock:
            for h in self._cache.values():
                if h is not self._main:
                    h.close()
            if self._main:
                self._main.close()
        super().close()


def create_logging(
    log_path: str, tag: str = None, n_workers: int = None, route: bool = False
):
    """
    Configure root logger with file and console handlers.

    Supports two modes:
    - route=False: All logs written to a single {name}_main.log file.
    - route=True: Thread workers are routed to worker_N.log, multiprocessing
      workers to process_<name>.log, and parent logs to {name}_main.log.

    Accepts log_path as either a directory or a file path (auto-detected).
    Creates the log directory if it does not exist.

    Args:
        log_path: Path to log directory or file.
        tag: Optional suffix for log file names (e.g., config name).
        n_workers: Number of workers (required if route=True).
        route: Enable auto-routing of thread and process worker logs.

    Example:
        >>> create_logging("/logs", tag="exp1", n_workers=4, route=True)
        # Creates: /logs/log_exp1_main.log, /logs/log_exp1_worker_0.log, ...
    """
    root = logging.getLogger()
    for h in root.handlers[:]:
        h.close()
        root.removeHandler(h)

    path = Path(log_path)
    log_dir = path.parent if path.suffix else path
    base = path.stem if path.suffix else "log"
    name = f"{base}_{tag}" if tag else base

    log_dir.mkdir(parents=True, exist_ok=True)

    handlers = (
        [_AutoRouteHandler(str(log_dir), name, n_workers)]
        if route and n_workers
        else [logging.FileHandler(log_dir / f"{name}_main.log", "a")]
    )

    handlers.append(logging.StreamHandler())

    logging.basicConfig(
        level=logging.INFO, format="%(message)s", handlers=handlers, force=True
    )
