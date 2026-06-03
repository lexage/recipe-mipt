"""Lightweight token accountant for OpenAI-compatible LLM calls.

Installs a one-time monkey-patch on the OpenAI client so every
chat.completions.create / completions.create response is inspected
and its usage added to the *currently active* TokenTracker (if any).

Usage in the runner:

    from src.utils.token_tracker import install_patch, TokenTracker, set_active

    install_patch()
    filter_tracker = TokenTracker(); set_active(filter_tracker)
    pipeline = builder.build(cfg)        # filter_apply happens here
    set_active(None)

    eval_tracker = TokenTracker();  set_active(eval_tracker)
    bench.eval(...)
    set_active(None)
"""

import logging
import threading
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class TokenTracker:
    input_tokens: int = 0
    output_tokens: int = 0
    n_calls: int = 0

    def add(self, prompt: int, completion: int) -> None:
        self.input_tokens += int(prompt or 0)
        self.output_tokens += int(completion or 0)
        self.n_calls += 1

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens

    def as_dict(self) -> dict:
        return {
            "input_tokens":  self.input_tokens,
            "output_tokens": self.output_tokens,
            "total_tokens":  self.total_tokens,
            "n_calls":       self.n_calls,
        }


# Thread-local active tracker so concurrent eval-workers don't stomp each other.
_state = threading.local()


def set_active(tracker):
    _state.active = tracker


def _record(response) -> None:
    tracker = getattr(_state, "active", None)
    if tracker is None:
        return
    usage = getattr(response, "usage", None)
    if usage is None:
        return
    prompt = getattr(usage, "prompt_tokens", None)
    completion = getattr(usage, "completion_tokens", None)
    tracker.add(prompt or 0, completion or 0)


_patched = False


def install_patch() -> None:
    """Wrap OpenAI's `create` methods once.  Subsequent calls are no-ops."""
    global _patched
    if _patched:
        return

    try:
        from openai.resources.chat.completions import Completions as ChatCompletions
        from openai.resources.completions import Completions as RawCompletions
    except ImportError:
        logger.warning("openai SDK not available — token tracker disabled.")
        return

    def _wrap(cls):
        original = cls.create

        def patched_create(self, *args, **kwargs):
            response = original(self, *args, **kwargs)
            # Skip streaming responses — they don't have a single .usage.
            if hasattr(response, "usage"):
                try:
                    _record(response)
                except Exception as e:
                    logger.debug("token tracker recording failed: %s", e)
            return response

        cls.create = patched_create

    _wrap(ChatCompletions)
    _wrap(RawCompletions)
    _patched = True
    logger.info("Token tracker monkey-patch installed.")
