"""The tag-based Caine agent, split out of the old main.py monolith.

Modules access shared paths as `constants.X` attributes (never by-name
imports) so tests can redirect a path in symbio.constants once and every
module sees it.
"""

# Registers the mlx_compat prepare hook with the mlx_gate (nothing here
# imports the engine at module scope — that stayed true only because the
# compat module itself is lazy now). Importing `symbio.app` therefore
# guarantees the patches apply before the first mlx_lm.load(), wherever the
# gate is resolved from.
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
