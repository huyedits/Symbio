"""`symb connect`: hook Symbio into another app.

Claude Desktop hosts MCP servers, not ACP agents, so it gets Symbio through
symbio_desktop.mcp_bridge, registered under `mcpServers` in its config file.
Everything else already in that file is kept exactly as it was, the file is
backed up before it is changed, and it is written through a rename so Claude
Desktop never reads half of one.
"""

from __future__ import annotations

import json
import os
import shutil
import sys
from pathlib import Path

from symbio import constants

CLAUDE_DESKTOP_CONFIG = (Path.home() / "Library" / "Application Support" / "Claude"
                         / "claude_desktop_config.json")
SERVER_NAME = "symbio"


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
