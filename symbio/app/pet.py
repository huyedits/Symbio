"""`symb pet`: the desktop cat, started and stopped like the daemon.

The pet is its own process, `python -m symbio_pet`, and deliberately not this
one: by the time this CLI runs it has imported the agent package (~105 MB),
and a pet that sits on the desktop all day must not carry that. `start`
launches it detached, `run` replaces this process with it, and the pet keeps
its own pid file, so one started as `symbio-pet` is found here too.
"""

from __future__ import annotations

import os
import signal
import subprocess
import sys
import time
from pathlib import Path

from symbio import constants

# How long `start` waits to see the pet write its pid file before saying it is
# out. Drawing the first frame takes under a second; importing AppKit cold
# can take a few.
_START_WAIT_S = 8.0


def pet_pid() -> int | None:
    """The running pet's pid, or None.

    A pid file outlives a killed process, and after a reboot its number is
    soon somebody else's, so the process has to be alive and has to be the
    pet.
    """
    try:
        pid = int(constants.PET_PID_FILE.read_text(encoding="utf-8").strip())
    except (OSError, ValueError):
        return None
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return None
    except PermissionError:
        pass
    try:
        command = subprocess.run(["ps", "-o", "command=", "-p", str(pid)],
                                 capture_output=True, text=True, timeout=5).stdout
    except (OSError, subprocess.SubprocessError):
        return None
    return pid if "symbio_pet" in command else None


def _argv(demo: bool, port: int) -> list[str]:
    argv = [sys.executable, "-m", "symbio_pet"]
    if demo:
        argv.append("--demo")
    if port != 8742:
        argv += ["--port", str(port)]
    return argv


def _package_root() -> str:
    """Where `-m symbio_pet` resolves: the checkout, or site-packages."""
    return str(Path(__file__).resolve().parent.parent.parent)


def _env() -> dict[str, str]:
    """The pet watches exactly the install this CLI resolved. It runs from the
    package root, so a relative SYMBIO_HOME would otherwise resolve somewhere
    else in the child."""
    env = dict(os.environ)
    env["SYMBIO_HOME"] = str(constants.PROJECT_DIR)
    return env


def start_pet(demo: bool = False, port: int = 8742) -> int:
    pid = pet_pid()
    if pid is not None:
        print(f"The pet is already out (PID {pid}). `symb pet stop` calls it in.")
        return 0
    constants.PET_PID_FILE.unlink(missing_ok=True)      # stale, per pet_pid
    log_path = constants.LOG_DIR / "pet.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with open(log_path, "a", encoding="utf-8") as log:
        process = subprocess.Popen(
            _argv(demo, port), cwd=_package_root(), env=_env(), start_new_session=True,
            stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT)
    deadline = time.monotonic() + _START_WAIT_S
    while time.monotonic() < deadline:
        if process.poll() is not None:
            # It said why before it left (no PyObjC, another pet already out).
            tail = log_path.read_text(encoding="utf-8", errors="replace").strip()
            print("The pet did not start:")
            print("\n".join(tail.splitlines()[-6:]))
            return 1
        if pet_pid() == process.pid:
            what = "the demo" if demo else f"watching {constants.PROJECT_DIR}"
            print(f"The pet is out (PID {process.pid}), {what}.")
            print("Double-click it to chat, drag it anywhere, right-click for "
                  "more. `symb pet stop` calls it in.")
            return 0
        time.sleep(0.1)
    print(f"The pet is starting (PID {process.pid}); it has not drawn yet. "
          f"See {log_path} if it never appears.")
    return 0


def stop_pet() -> int:
    pid = pet_pid()
    if pid is None:
        constants.PET_PID_FILE.unlink(missing_ok=True)
        print("No pet is out.")
        return 0
    try:
        os.kill(pid, signal.SIGTERM)
    except ProcessLookupError:
        pass
    except PermissionError:
        print(f"Permission denied: cannot stop process {pid}.")
        return 1
    # It saves where it sat and removes its own pid file on the way out.
    deadline = time.monotonic() + 5.0
    while time.monotonic() < deadline and pet_pid() == pid:
        time.sleep(0.1)
    if pet_pid() == pid:
        print(f"Asked the pet (PID {pid}) to go; it is still there.")
        return 1
    constants.PET_PID_FILE.unlink(missing_ok=True)
    print(f"The pet (PID {pid}) went home.")
    return 0


def pet_status() -> int:
    pid = pet_pid()
    print(f"Pet out: {'yes' if pid else 'no'}")
    if pid:
        print(f"PID: {pid}")
    print(f"Watching: {constants.PROJECT_DIR}")
    return 0


def run_pet(demo: bool = False, port: int = 8742) -> int:
    """This terminal becomes the pet: exec, so none of this process stays."""
    sys.stdout.flush()
    os.chdir(_package_root())
    argv = _argv(demo, port)
    os.execve(argv[0], argv, _env())
    return 1   # not reached
