"""The tag-based Caine agent, split out of the old main.py monolith.

Modules access shared paths as `constants.X` attributes (never by-name
imports) so tests can redirect a path in symbio.constants once and every
module sees it.
"""

from symbio.app.chat import chat_loop
from symbio.app.config import load_config
from symbio.app.training import run_training

__all__ = ["chat_loop", "load_config", "run_training"]

try:
    from symbio.app.telegram import TelegramBot  # noqa: F401
    __all__.append("TelegramBot")
except ImportError:
    TelegramBot = None
