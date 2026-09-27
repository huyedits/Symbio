"""Guards for suite-wide machinery in conftest.py that a merge can drop silently.

The MLX skip hook was lost once when a branch rewrote conftest.py from an older
copy; nothing failed locally on a Mac (the hook is inert there), and CI on
Linux went from green to 126 failures. This makes the loss visible on any host.
"""
import conftest


def test_engine_skip_hook():
    assert callable(getattr(conftest, "pytest_runtest_makereport", None)), (
        "tests/conftest.py lost its MLX skip hook; restore it "
        "(see the '---- hosts without the MLX engine ----' block)")
    assert callable(getattr(conftest, "_needs_engine", None))
