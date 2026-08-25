"""The tag-based Caine agent, split out of the old main.py monolith.

Modules access shared paths as `constants.X` attributes (never by-name
imports) so tests can redirect a path in symbio.constants once and every
module sees it.
"""

# First: patches the installed mlx-lm so load() accepts checkpoints it does
# not yet know about. Must precede any import that reaches mlx_lm.load().
from symbio.app import mlx_compat  # noqa: F401
from symbio.app.chat import chat_loop
from symbio.app.config import load_config
from symbio.app.training import run_training

__all__ = ["chat_loop", "load_config", "run_training"]

try:
    from symbio.app.telegram import TelegramBot  # noqa: F401
    __all__.append("TelegramBot")
except ImportError:
    TelegramBot = None
