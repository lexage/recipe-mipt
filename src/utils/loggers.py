import os
import logging
import distutils.dir_util
import threading


class _ThreadFileHandler(logging.Handler):
    """Per-thread file handler with auto PID_TID suffix and optional tag."""
    def __init__(self, path, mode='a', enc=None, tag=None):
        super().__init__()
        self._path, self._mode, self._enc, self._tag = path, mode, enc, tag
        self._cache, self._lock = {}, threading.Lock()
    
    def _get_file(self):
        tid = threading.get_ident()
        if tid not in self._cache:
            with self._lock:
                if tid not in self._cache:
                    base, ext = os.path.splitext(self._path)
                    # Формируем имя: base[_tag]_pid_tid.ext
                    parts = [base]
                    if self._tag: parts.append(self._tag)
                    parts.append(os.getpid())
                    parts.append(tid)
                    fname = f"{'_'.join(map(str, parts))}{ext}"
                    distutils.dir_util.mkpath(os.path.dirname(fname) or '.')
                    self._cache[tid] = logging.FileHandler(fname, self._mode, self._enc)
        return self._cache[tid]
    
    def emit(self, record): self._get_file().emit(record)
    def close(self):
        with self._lock:
            for h in self._cache.values(): h.close()
        super().close()

def create_logging(log_filename=None, level=logging.INFO, tag=None):
    root = logging.getLogger()
    if root.hasHandlers(): root.handlers.clear()
    
    if log_filename:
        handlers = [_ThreadFileHandler(log_filename, 'a', tag=tag), logging.StreamHandler()]
        logging.basicConfig(level=level, format="%(message)s", handlers=handlers)
    else:
        logging.basicConfig(level=level, format="%(message)s", handlers=[logging.StreamHandler()])