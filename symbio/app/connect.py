"""`symb connect`: hook Symbio into another app.

Claude Desktop hosts MCP servers, not ACP agents, so it gets Symbio through
symbio_desktop.mcp_bridge, registered under `mcpServers` in its config file.
Everything else already in that file is kept exactly as it was, the file is
backed up before it is changed, and it is written through a rename so Claude
Desktop never reads half of one.

Hermes Agent is an ACP server but not an ACP client, so it takes the same MCP
bridge. Its config.yaml is registered through Hermes's own `hermes mcp add`
rather than written from here: Hermes keeps that file's comments through a
round-trip writer, and a plain YAML dump would strip them.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

from symbio import constants

CLAUDE_DESKTOP_CONFIG = (Path.home() / "Library" / "Application Support" / "Claude"
                         / "claude_desktop_config.json")
SERVER_NAME = "symbio"
# Hermes gives an MCP tool call 300s by default. ask_symbio can hold a /train
# for up to the bridge's own turn limit (symbio_desktop.mcp_bridge.TURN_S).
HERMES_TOOL_TIMEOUT_S = 900
_ANSI = re.compile(r"\x1b\[[0-9;?]*[A-Za-z]")


def package_root() -> str:
    """This code's own root, so the host runs this version of the bridge and
    not whatever an editable install elsewhere happens to point at."""
    return str(Path(__file__).resolve().parent.parent.parent)


def mcp_entry() -> dict:
    return {
        "command": sys.executable,
        "args": ["-m", "symbio_desktop.mcp_bridge"],
        "env": {"SYMBIO_HOME": str(constants.PROJECT_DIR), "PYTHONPATH": package_root()},
    }


def claude_desktop(remove: bool = False, config_path: Path = CLAUDE_DESKTOP_CONFIG) -> int:
    try:
        data = (json.loads(config_path.read_text(encoding="utf-8"))
                if config_path.exists() else {})
    except (OSError, ValueError) as e:
        print(f"Could not read {config_path} ({e}); leaving it alone.")
        return 1
    if not isinstance(data, dict):
        print(f"{config_path} is not a JSON object; leaving it alone.")
        return 1
    servers = data.get("mcpServers")
    if not isinstance(servers, dict):
        servers = {}
    if remove:
        if SERVER_NAME not in servers:
            print("Symbio is not connected to Claude Desktop.")
            return 0
        servers.pop(SERVER_NAME)
    else:
        servers[SERVER_NAME] = mcp_entry()
    data["mcpServers"] = servers

    config_path.parent.mkdir(parents=True, exist_ok=True)
    if config_path.exists():
        shutil.copy2(config_path, config_path.with_name(config_path.name + ".bak-symbio"))
    temporary = config_path.with_name(config_path.name + ".tmp-symbio")
    temporary.write_text(json.dumps(data, indent=2), encoding="utf-8")
    os.replace(temporary, config_path)

    if remove:
        print("Removed Symbio from Claude Desktop. Restart Claude Desktop to drop it.")
    else:
        print(f"Connected: Claude Desktop will start Symbio's MCP bridge "
              f"({SERVER_NAME}: ask_symbio, symbio_status), watching {constants.PROJECT_DIR}.")
        print("Quit and reopen Claude Desktop, then ask Claude to use Symbio.")
    print(f"(Backup of the previous config: {config_path.name}.bak-symbio)")
    return 0


def hermes_home() -> Path:
    home = os.environ.get("HERMES_HOME", "").strip()
    return Path(home).expanduser() if home else Path.home() / ".hermes"


def hermes_binary() -> str | None:
    """`hermes` on PATH, or where its two installers put it: the desktop
    installer under installs/, install.sh under hermes-agent/venv."""
    found = shutil.which("hermes")
    if found:
        return found
    candidates: list[Path] = []
    for home in dict.fromkeys((hermes_home(), Path.home() / ".hermes")):
        candidates += sorted(home.glob("installs/*/environments/*/venv/bin/hermes"),
                             key=lambda p: p.stat().st_mtime, reverse=True)
        candidates.append(home / "hermes-agent" / "venv" / "bin" / "hermes")
    for candidate in candidates:
        if os.access(candidate, os.X_OK):
            return str(candidate)
    return None


def hermes(remove: bool = False, binary: str | None = None) -> int:
    binary = binary or hermes_binary()
    if not binary:
        print("Hermes Agent was not found: no `hermes` on PATH or under "
              f"{hermes_home()}. Install it, then run this again.")
        return 1

    def run(*args: str, answers: str = "") -> tuple[int, str]:
        # Hermes asks before it overwrites or removes, and which tools to
        # enable; every question here gets a yes.
        try:
            done = subprocess.run([binary, *args], input=answers, capture_output=True,
                                  text=True, timeout=180, check=False)
        except (OSError, subprocess.TimeoutExpired) as e:
            return 1, str(e)
        return done.returncode, _ANSI.sub("", done.stdout + done.stderr)

    if remove:
        code, said = run("mcp", "remove", SERVER_NAME, answers="y\n")
        if code != 0:
            print(said.strip())
            return 1
        print("Symbio is not connected to Hermes." if "not found" in said
              else "Removed Symbio from Hermes. New Hermes sessions will not have it.")
        return 0

    entry = mcp_entry()
    code, said = run("mcp", "add", SERVER_NAME, "--command", entry["command"],
                     "--env", *(f"{k}={v}" for k, v in entry["env"].items()),
                     "--args", *entry["args"], answers="y\ny\n")
    if code != 0 or f"Saved '{SERVER_NAME}'" not in said:
        print("Hermes did not register Symbio:")
        print(said.strip())
        return 1
    code, said = run("config", "set", f"mcp_servers.{SERVER_NAME}.timeout",
                     str(HERMES_TOOL_TIMEOUT_S))
    if code != 0:
        print(f"Registered, but the tool timeout stayed at Hermes's default "
              f"(a /train through ask_symbio may be cut off): {said.strip()}")
    print(f"Connected: Hermes will start Symbio's MCP bridge ({SERVER_NAME}: "
          f"ask_symbio, symbio_status), watching {constants.PROJECT_DIR}.")
    print("Start a new Hermes session (or /reload-mcp) and ask Hermes to use Symbio.")
    return 0
